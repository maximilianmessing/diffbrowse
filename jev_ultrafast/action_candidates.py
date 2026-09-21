"""Flattened browser action candidates and single-token label pool for MLX models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateAction:
    """A single executable browser action choice."""

    index: int
    candidate_id: str  # e.g. "A0", "A1", ...
    action_id: str  # e.g. "e1", "scroll_down", "wait", "DONE", "BLOCKED"
    operation: str  # "CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"
    target: Optional[str] = None  # e.g. "1", "2:1"
    label: str = ""  # Human-readable description
    raw_action: Optional[Dict[str, Any]] = None


def flatten_actions(actions: List[Dict[str, Any]]) -> List[CandidateAction]:
    """Flatten dynamic browser state actions into an ordered list of executable candidates.

    Preserves exact correspondence to jev-ultrafast's action_space logic:
    - CLICK on clickable controls
    - TYPE_TEXT on editable controls
    - SELECT on select options
    - SCROLL_DOWN, SCROLL_UP, WAIT
    - Terminal DONE and BLOCKED actions
    """
    from .model import action_space

    elements, targets, controls = action_space(actions)
    candidates: List[CandidateAction] = []
    idx = 0

    # 1. Operation-specific target actions
    raw_targets = []
    for op in ("CLICK", "TYPE_TEXT", "SELECT"):
        if op not in targets:
            continue
        for target_id, act in targets[op].items():
            elem_idx = target_id.split(":")[0]
            elem = elements[int(elem_idx) - 1] if elem_idx.isdigit() and int(elem_idx) <= len(elements) else {}
            role = elem.get("role", act.get("role", "element"))
            label_text = act.get("label", elem.get("label", ""))
            is_option = role == "option"
            is_selected = (
                act.get("selected") == "true"
                or elem.get("selected") == "true"
                or act.get("checked") == "true"
            )
            # Sort order:
            # 0: Active unselected options (e.g. autocomplete suggestions, unselected radio/dropdown options)
            # 1: TYPE_TEXT editable fields (form inputs)
            # 2: Standard CLICK/SELECT actions
            # 3: Already selected options
            if is_option and not is_selected:
                priority = 0
            elif op == "TYPE_TEXT":
                priority = 1
            elif not is_selected:
                priority = 2
            else:
                priority = 3
            raw_targets.append((priority, op, target_id, act, elem_idx, role, label_text, is_selected))

    raw_targets.sort(key=lambda x: x[0])

    for _, op, target_id, act, elem_idx, role, label_text, is_selected in raw_targets:
        if op == "CLICK":
            sel_str = " [selected]" if is_selected else ""
            desc = f"CLICK [{elem_idx}] {label_text} ({role}){sel_str}"
        elif op == "TYPE_TEXT":
            cur_val = act.get("current_value", act.get("value", ""))
            val_str = f' [current: "{cur_val}"]' if cur_val else ""
            desc = f"TYPE_TEXT [{elem_idx}] {label_text} ({role}){val_str}"
        elif op == "SELECT":
            desc = f"SELECT [{target_id}] {label_text}"
        else:
            desc = f"{op} [{target_id}] {label_text}"

        candidates.append(
            CandidateAction(
                index=idx,
                candidate_id=f"A{idx}",
                action_id=act["id"],
                operation=op,
                target=target_id,
                label=desc,
                raw_action=act,
            )
        )
        idx += 1

    # 2. Control actions (scroll, wait)
    for control_id, act in controls.items():
        candidates.append(
            CandidateAction(
                index=idx,
                candidate_id=f"A{idx}",
                action_id=act["id"],
                operation=control_id,
                target=None,
                label=f"{control_id}: {act.get('label', control_id)}",
                raw_action=act,
            )
        )
        idx += 1

    # 3. Terminal actions (DONE, BLOCKED)
    candidates.append(
        CandidateAction(
            index=idx,
            candidate_id=f"A{idx}",
            action_id="DONE",
            operation="DONE",
            target=None,
            label="DONE: Every requirement is visibly satisfied.",
            raw_action=None,
        )
    )
    idx += 1

    candidates.append(
        CandidateAction(
            index=idx,
            candidate_id=f"A{idx}",
            action_id="BLOCKED",
            operation="BLOCKED",
            target=None,
            label="BLOCKED: No supported operation can make progress.",
            raw_action=None,
        )
    )
    return candidates


class LabelPool:
    """Safe single-token label pool satisfying exact 1-token encode/decode criteria."""

    def __init__(self, tokenizer: Any):
        if hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "encode"):
            self.tokenizer = tokenizer.tokenizer
        else:
            self.tokenizer = tokenizer
        self.special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        self._cache: Dict[str, List[Tuple[str, int]]] = {}

    def get_pool(self, strategy: str = "index", size: int = 256) -> List[Tuple[str, int]]:
        """Return at least `size` (label_str, token_id) tuples for the specified strategy.

        Strategies:
        - 'index': Human-readable single characters (A-Z, a-z, 0-9, then clean short words).
        - 'arbitrary': Arbitrary safe single tokens from across the vocabulary.
        - 'semantic': Compact tokens prioritizing letter codes or semantic words.
        """
        if strategy in self._cache and len(self._cache[strategy]) >= size:
            return self._cache[strategy][:size]

        valid_tokens: List[Tuple[str, int]] = []
        seen_token_ids = set()

        if strategy in ("index", "semantic"):
            # Try single ASCII letters A-Z, a-z, 0-9 first
            candidate_chars = (
                [chr(c) for c in range(ord("A"), ord("Z") + 1)]
                + [chr(c) for c in range(ord("a"), ord("z") + 1)]
                + [chr(c) for c in range(ord("0"), ord("9") + 1)]
            )
            for char in candidate_chars:
                ids = self.tokenizer.encode(char, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in self.special_ids and ids[0] not in seen_token_ids:
                    if self.tokenizer.decode(ids) == char:
                        valid_tokens.append((char, ids[0]))
                        seen_token_ids.add(ids[0])

        # Fill remaining slots with clean, alphanumeric single tokens from vocabulary
        vocab_size = getattr(self.tokenizer, "vocab_size", 262144)
        for token_id in range(vocab_size):
            if len(valid_tokens) >= max(size, 300):
                break
            if token_id in self.special_ids or token_id in seen_token_ids:
                continue
            text = self.tokenizer.decode([token_id])
            if not text or text.isspace() or "\ufffd" in text:
                continue
            # Must roundtrip exactly as one token
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded == [token_id] and text.strip() == text and text.isalnum():
                valid_tokens.append((text, token_id))
                seen_token_ids.add(token_id)

        if len(valid_tokens) < size:
            raise RuntimeError(f"Could only find {len(valid_tokens)} safe single tokens, needed {size}")

        self._cache[strategy] = valid_tokens
        return valid_tokens[:size]

    def map_candidates(
        self,
        candidates: List[CandidateAction],
        strategy: str = "index",
    ) -> Tuple[List[int], Dict[int, CandidateAction], Dict[str, Tuple[str, int]], str]:
        """Map candidate actions to single label tokens.

        Returns:
            candidate_token_ids: list of int token IDs corresponding to each candidate
            token_to_candidate: mapping token_id -> CandidateAction
            candidate_to_label: mapping candidate_id -> (label_str, token_id)
            prompt_section: formatted candidate list string for prompt injection
        """
        pool = self.get_pool(strategy=strategy, size=max(len(candidates), 256))
        candidate_token_ids: List[int] = []
        token_to_candidate: Dict[int, CandidateAction] = {}
        candidate_to_label: Dict[str, Tuple[str, int]] = {}
        lines: List[str] = ["Executable Actions:"]

        for cand, (label_str, token_id) in zip(candidates, pool):
            candidate_token_ids.append(token_id)
            token_to_candidate[token_id] = cand
            candidate_to_label[cand.candidate_id] = (label_str, token_id)
            lines.append(f"[{label_str}] {cand.label}")

        prompt_section = "\n".join(lines)
        return candidate_token_ids, token_to_candidate, candidate_to_label, prompt_section
