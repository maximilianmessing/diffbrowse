"""Pure fixed-template compilation for training-free structured diffusion reads.

Design references and immutable upstream revisions: docs/structured-diffusion-plan.md.
Combines seeded read-only slots, template pinning, and hierarchical candidate partitioning.
No Cartesian enumeration or generated JSON is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class AnswerSlot:
    name: str
    choices: Tuple[str, ...]
    labels: Tuple[str, ...]
    token_ids: Tuple[int, ...]
    position: Optional[int]
    token_ids_mx: Optional[Any] = None

    @property
    def choice_to_label(self) -> Dict[str, str]:
        return dict(zip(self.choices, self.labels))

    @property
    def label_to_choice(self) -> Dict[str, str]:
        return dict(zip(self.labels, self.choices))

    @property
    def token_to_choice(self) -> Dict[int, str]:
        return dict(zip(self.token_ids, self.choices))

    @property
    def choice_to_token(self) -> Dict[str, int]:
        return dict(zip(self.choices, self.token_ids))


@dataclass(frozen=True)
class CanvasLayout:
    tokens: Tuple[int, ...]
    slots: Tuple[AnswerSlot, ...]
    fixed: Tuple[bool, ...]
    width: int
    hierarchical_map: Optional[Dict[str, Dict[str, Any]]] = None
    questions: Optional[Dict[str, Any]] = None

    @property
    def slot_map(self) -> Dict[str, AnswerSlot]:
        return {slot.name: slot for slot in self.slots}


def chat_prompt(tokenizer: Any, content: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )


def answer_tokens(tokenizer: Any, answer: str) -> Tuple[int, ...]:
    """Derive assistant framing and turn close from the checkpoint's chat template."""
    messages = [{"role": "user", "content": "Structured read"}]
    prefix = chat_prompt(tokenizer, messages[0]["content"])
    try:
        complete = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        complete = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
    if not complete.startswith(prefix):
        raise ValueError("Checkpoint chat template does not expose a stable assistant continuation")
    return tuple(tokenizer.encode(complete[len(prefix):], add_special_tokens=False))


def partition_candidates_by_role(criteria: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Partition target candidates into categorical DOM groups:
    - G1_INPUTS: textbox, combobox, searchbox, checkbox, radio, slider, spinbutton
    - G2_ACTIONS: button, menuitem, tab, switch
    - G3_NAVIGATION: link, treeitem, breadcrumb
    - G4_CONTENT: options, list items, calendar cells, other interactables
    """
    g1, g2, g3, g4 = {}, {}, {}, {}
    for key, val in criteria.items():
        role = ""
        if isinstance(val, dict):
            role = str(val.get("role", "")).lower()
            if not role:
                label = str(val.get("element", "")).lower()
                if any(w in label for w in ("button", "submit", "menu")):
                    role = "button"
                elif any(w in label for w in ("input", "type", "search", "box", "text")):
                    role = "textbox"
                elif "link" in label:
                    role = "link"
        elif isinstance(val, str):
            label = val.lower()
            if any(w in label for w in ("button", "submit", "menu")):
                role = "button"
            elif any(w in label for w in ("input", "type", "search", "box", "text")):
                role = "textbox"
            elif "link" in label:
                role = "link"

        if role in ("textbox", "combobox", "searchbox", "checkbox", "radio", "slider", "spinbutton"):
            g1[key] = val
        elif role in ("button", "menuitem", "tab", "switch"):
            g2[key] = val
        elif role in ("link", "treeitem", "breadcrumb"):
            g3[key] = val
        else:
            g4[key] = val

    groups = {}
    if g1:
        groups["G1_INPUTS"] = g1
    if g2:
        groups["G2_ACTIONS"] = g2
    if g3:
        groups["G3_NAVIGATION"] = g3
    if g4:
        groups["G4_CONTENT"] = g4

    # Sub-partition any group exceeding 35 items, or if all items landed in a single group
    final_groups: Dict[str, Dict[str, Any]] = {}
    for g_name, items in groups.items():
        if len(items) <= 35 and len(groups) > 1:
            final_groups[g_name] = items
        else:
            item_list = list(items.items())
            chunk_size = 25
            for chunk_idx in range(0, len(item_list), chunk_size):
                sub_items = dict(item_list[chunk_idx : chunk_idx + chunk_size])
                suffix = f"_{chunk_idx // chunk_size + 1}" if len(item_list) > chunk_size else ""
                final_groups[f"{g_name}{suffix}"] = sub_items

    if not final_groups and criteria:
        item_list = list(criteria.items())
        chunk_size = 25
        for chunk_idx in range(0, len(item_list), chunk_size):
            sub_items = dict(item_list[chunk_idx : chunk_idx + chunk_size])
            final_groups[f"G{chunk_idx // chunk_size + 1}_ITEMS"] = sub_items

    return final_groups


def build_cross_head_attention_mask(
    layout: CanvasLayout,
    encoder_len: int,
    base_mask: Optional[Any] = None,
    framework: str = "mlx",
) -> Any:
    """Build a 2D cross-head attention mask that isolates sibling target slots.

    - All queries can attend to the prompt observation tokens [0, encoder_len - 1].
    - All queries can attend to fixed template tokens.
    - All queries can attend to the operation slot.
    - Target slots (e.g. click_target, type_text_target) cannot cross-attend to each other.
    """
    canvas_len = layout.width
    key_len = encoder_len + canvas_len

    # Identify target slot positions
    target_slots: List[AnswerSlot] = [
        s for s in layout.slots if "target" in s.name and s.position is not None
    ]
    target_positions = [s.position for s in target_slots if s.position is not None]

    grid = [[True] * key_len for _ in range(canvas_len)]
    for p_a in target_positions:
        for p_b in target_positions:
            if p_a != p_b:
                grid[p_a][encoder_len + p_b] = False

    if framework == "torch" or (mx is None and torch is not None):
        block = torch.tensor(grid, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    elif mx is not None:
        block = mx.array(grid, dtype=mx.bool_)[None, None, :, :]
    else:
        block = grid
    if base_mask is None:
        return block
    return base_mask & block


class CanvasCompiler:
    """One tokenizer per bounded compiler cache; all alternatives checked in context."""

    def __init__(self, tokenizer: Any, max_width: int = 256):
        if max_width not in (32, 64, 128, 256):
            raise ValueError("max_width must be 32, 64, 128 or 256")
        self.tokenizer = tokenizer
        self.max_width = max_width
        self._cache: Dict[Any, CanvasLayout] = {}
        self._labels: Optional[Tuple[str, ...]] = None

    def reset(self) -> None:
        self._cache.clear()

    def labels(self) -> Tuple[str, ...]:
        if self._labels is None:
            # Vocabulary-derived labels are only proposals; compile validates context.
            # Prefer upper-case letters first (A, B, C...), then lower-case, then digits.
            proposals = list(ascii_uppercase + ascii_lowercase + digits)
            special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            vocab_size = getattr(self.tokenizer, "vocab_size", 0)
            for tid in range(vocab_size):
                if tid in special:
                    continue
                try:
                    label = self.tokenizer.decode([tid])
                except Exception:
                    continue
                if label.isalnum() and label.isascii() and len(label) <= 8:
                    proposals.append(label)
                if len(proposals) >= 2048:
                    break
            self._labels = tuple(dict.fromkeys(proposals))
        return self._labels

    def compile(self, questions: Dict[str, Any], hierarchical: bool = True) -> CanvasLayout:
        # Check if hierarchical partitioning is needed for large candidate sets
        hierarchical_map: Dict[str, Dict[str, Any]] = {}
        processed_questions: Dict[str, Any] = {}

        for name, q in questions.items():
            criteria = q.get("criteria", {})
            if hierarchical and "target" in name and len(criteria) > 35:
                groups = partition_candidates_by_role(criteria)
                if len(groups) > 1:
                    group_head_name = f"{name.replace('_target', '')}_group"
                    processed_questions[group_head_name] = {
                        "type": "choice",
                        "criteria": {
                            g: f"Category {g} ({len(items)} items)"
                            for g, items in groups.items()
                        },
                        "instructions": {"goal": f"Select category for {name}"},
                    }
                    sub_heads = {}
                    for g_name, g_crit in groups.items():
                        sub_name = f"{name}_{g_name.lower()}"
                        processed_questions[sub_name] = {
                            "type": "choice",
                            "criteria": g_crit,
                            "instructions": {"goal": f"Select target in {g_name}"},
                        }
                        sub_heads[g_name] = sub_name
                    hierarchical_map[name] = {
                        "group_head": group_head_name,
                        "sub_heads": sub_heads,
                    }
                    continue
            processed_questions[name] = q

        key = tuple((name, tuple(q["criteria"])) for name, q in processed_questions.items())
        if key in self._cache:
            return self._cache[key]
        if not key or any(not choices for _, choices in key):
            raise ValueError("Every answer head needs at least one choice")
        names = [name for name, _ in key]
        if any(not name.replace("_", "").isalnum() for name in names):
            raise ValueError("Invalid answer head name")
        base_labels = ["A"] * len(names)

        def render(labels: list[str]) -> Tuple[int, ...]:
            return answer_tokens(
                self.tokenizer,
                "\n".join(f"{n}: {v}" for n, v in zip(names, labels)),
            )

        base = render(base_labels)
        slots = []
        for index, (name, choices) in enumerate(key):
            if len(choices) == 1:
                slots.append(AnswerSlot(name, choices, ("A",), (), None))
                continue
            position = None
            valid = []
            seen = set()
            for label in self.labels():
                if label == "A":
                    continue
                trial = base_labels.copy()
                trial[index] = label
                ids = render(trial)
                if len(ids) != len(base):
                    continue
                differences = [i for i, (a, b) in enumerate(zip(base, ids)) if a != b]
                if len(differences) != 1:
                    continue
                p = differences[0]
                if position is None:
                    position = p
                    valid = [("A", base[p])]
                    seen.add(base[p])
                special_ids = getattr(self.tokenizer, "all_special_ids", []) or []
                if p != position or ids[p] in seen or ids[p] in special_ids:
                    continue
                valid.append((label, ids[p]))
                seen.add(ids[p])
                if len(valid) == len(choices):
                    break
            if len(valid) != len(choices):
                raise ValueError(f"Cannot represent all {len(choices)} choices for {name} in one token slot")
            token_ids_tuple = tuple(v[1] for v in valid)
            if mx is not None:
                token_ids_tensor = mx.array(token_ids_tuple, dtype=mx.int32)
            elif torch is not None:
                token_ids_tensor = torch.tensor(token_ids_tuple, dtype=torch.int32)
            else:
                token_ids_tensor = token_ids_tuple
            slots.append(
                AnswerSlot(
                    name,
                    choices,
                    tuple(v[0] for v in valid),
                    token_ids_tuple,
                    position,
                    token_ids_tensor,
                )
            )
        width = next((w for w in (32, 64, 128, 256) if len(base) <= w <= self.max_width), None)
        if width is None:
            raise ValueError(f"Answer template needs {len(base)} tokens, exceeds configured canvas")
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is None:
            raise ValueError("Checkpoint tokenizer needs a pad or eos token")
        free = {s.position for s in slots if s.position is not None}
        layout = CanvasLayout(
            base + (pad,) * (width - len(base)),
            tuple(slots),
            tuple(i not in free for i in range(width)),
            width,
            hierarchical_map=hierarchical_map if hierarchical_map else None,
            questions=processed_questions,
        )
        if len(self._cache) >= 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = layout
        return layout
