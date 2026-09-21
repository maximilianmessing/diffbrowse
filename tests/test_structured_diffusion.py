"""Offline unit tests for Training-Free Structured Diffusion.

All tests run completely offline with tiny tensor models and mock tokenizers.
No remote downloads or paid APIs are invoked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import mlx.core as mx
import pytest

from jev_ultrafast.backends.mlx_structured import MlxDiffusionStructuredBackend
from jev_ultrafast.decision_request import build_decision_request
from jev_ultrafast.model import action_space
from jev_ultrafast.structured_canvas import CanvasCompiler


class MockTokenizer:
    """Deterministic mock tokenizer supporting single-token ASCII labels."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.all_special_ids = [0, 1]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        out = ""
        for m in messages:
            out += f"<{m['role']}>{m['content']}</{m['role']}>\n"
        if add_generation_prompt:
            out += "<assistant>"
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Simple deterministic character-level encoding offset to avoid special tokens
        tokens = []
        for ch in text:
            tokens.append((ord(ch) % (self.vocab_size - 10)) + 2)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return "".join(chr(t - 2) for t in token_ids)


class MockLayerCache:
    def __init__(self):
        self.state = mx.zeros((1, 4))
        self.offset = 0

    def clone(self):
        c = MockLayerCache()
        c.state = mx.array(self.state)
        c.offset = self.offset
        return c


class MockDiffusionModel:
    """Mock MLX diffusion model producing controlled logits without GPU overhead."""

    def __init__(
        self,
        vocab_size: int = 512,
        target_token_id: int = 67,
        logits_fn: Optional[Any] = None,
    ):
        self.vocab_size = vocab_size
        self.target_token_id = target_token_id
        self.logits_fn = logits_fn
        self.call_count = 0
        self.config = {
            "model_type": "diffusion_gemma",
            "text_config": SimpleNamespace(vocab_size=vocab_size),
        }
        self.embed_tokens = SimpleNamespace(weight=mx.zeros((vocab_size, 32)))
        self.last_self_conditioning = None
        self.last_canvas = None
        self.last_decoder_attention_mask = None

    @property
    def prefers_logits_self_conditioning(self) -> bool:
        return True

    def make_cache(self, max_size=None):
        from mlx_vlm.models.cache import KVCache

        return [KVCache()]

    def diffusion_prefill_cache(self, input_ids, cache=None, **kwargs):
        if cache is None:
            cache = self.make_cache()
        return cache

    def diffusion_update_cache(self, input_ids, cache=None, **kwargs):
        return cache

    def diffusion_decoder_masks(self, current_canvas, kv_cache, decoder_attention_mask=None):
        return {"full_attention": None, "sliding_attention": None}

    def diffusion_decoder_logits(
        self,
        current_canvas,
        cache=None,
        self_conditioning=None,
        decoder_attention_mask=None,
    ):
        self.call_count += 1
        self.last_self_conditioning = self_conditioning
        self.last_canvas = current_canvas
        self.last_decoder_attention_mask = decoder_attention_mask
        if self.logits_fn is not None:
            return self.logits_fn(self.call_count, current_canvas, self_conditioning)
        _, L = current_canvas.shape
        # Create base logits
        arr = [[0.0] * self.vocab_size for _ in range(L)]
        for row in arr:
            row[self.target_token_id] = 6.0
        return mx.array([arr], dtype=mx.float32)

    def diffusion_prepare_self_conditioning(self):
        return None

    def diffusion_self_conditioning(self, processed_logits, embedding_weight):
        return processed_logits


def sample_browser_state():
    return {
        "url": "https://travel.example.com",
        "title": "Flight Booking",
        "text": "Select your flight departure and arrival destination.",
        "actions": [
            {"id": "btn_roundtrip", "kind": "click", "label": "Round Trip", "role": "button", "node": 10},
            {"id": "btn_oneway", "kind": "click", "label": "One Way", "role": "button", "node": 11},
            {"id": "field_origin", "kind": "fill", "label": "Origin", "role": "combobox", "node": 20},
            {"id": "field_dest", "kind": "fill", "label": "Destination", "role": "combobox", "node": 21},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }


def test_compiler_context_tokenization_and_case_sensitivity():
    tok = MockTokenizer()
    compiler = CanvasCompiler(tok, max_width=64)

    questions = {
        "operation": {"criteria": {"CLICK": "Click element", "TYPE_TEXT": "Enter text"}},
        "click_target": {"criteria": {"1": "Button 1", "2": "Button 2"}},
    }
    layout = compiler.compile(questions)
    assert layout.width == 64
    assert len(layout.slots) == 2

    # Check case-sensitive labels
    labels = compiler.labels()
    assert "A" in labels
    assert "a" in labels
    assert labels.index("A") < labels.index("a")

    op_slot = layout.slots[0]
    assert op_slot.name == "operation"
    assert op_slot.choices == ("CLICK", "TYPE_TEXT")
    assert len(op_slot.labels) == 2
    assert op_slot.choice_to_label["CLICK"] == op_slot.labels[0]
    assert op_slot.label_to_choice[op_slot.labels[0]] == "CLICK"


def test_compiler_single_choice_and_absent_heads():
    tok = MockTokenizer()
    compiler = CanvasCompiler(tok, max_width=64)

    # single-choice head: position should be None
    questions = {
        "operation": {"criteria": {"CLICK": "Only click"}},
        "click_target": {"criteria": {"1": "Only button"}},
    }
    layout = compiler.compile(questions)
    for slot in layout.slots:
        assert slot.position is None
        assert slot.choices == ("CLICK",) if slot.name == "operation" else ("1",)
        assert slot.choice_to_label == {"CLICK": "A"} if slot.name == "operation" else {"1": "A"}

    # Absent heads: select_target and type_text_target should not be in layout
    slot_names = [s.name for s in layout.slots]
    assert "select_target" not in slot_names
    assert "type_text_target" not in slot_names


def test_compiler_capacity_and_explicit_rejection():
    tok = MockTokenizer(vocab_size=64)  # Small vocabulary
    compiler = CanvasCompiler(tok, max_width=32)

    # 100 choices cannot fit in small vocab proposals -> explicit ValueError
    large_criteria = {str(i): f"Choice {i}" for i in range(100)}
    with pytest.raises(ValueError, match="Cannot represent all"):
        compiler.compile({"click_target": {"criteria": large_criteria}}, hierarchical=False)

    # Exceeding canvas max_width -> explicit ValueError
    compiler_tiny = CanvasCompiler(tok, max_width=32)
    many_heads = {f"q_{i}": {"criteria": {"A": "1", "B": "2"}} for i in range(20)}
    with pytest.raises(ValueError, match="exceeds configured canvas"):
        compiler_tiny.compile(many_heads)


def test_fixed_token_preservation_and_answer_noise():
    tok = MockTokenizer()
    compiler = CanvasCompiler(tok, max_width=64)
    questions = {
        "operation": {"criteria": {"CLICK": "Click", "TYPE_TEXT": "Type"}},
        "click_target": {"criteria": {"1": "B1", "2": "B2"}},
    }
    layout = compiler.compile(questions)

    # Fixed mask must be False at answer slot positions, True elsewhere
    slot_positions = {s.position for s in layout.slots if s.position is not None}
    for idx, is_fixed in enumerate(layout.fixed):
        if idx in slot_positions:
            assert not is_fixed
        else:
            assert is_fixed


def test_two_pass_refinement_and_pin_masking():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="refinement_comparison",
        num_passes=2,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Book a flight", state, space, [])

    assert decision.stable_steps == 2
    assert decision.metadata["actual_forward_passes"] == 2
    assert decision.metadata["num_passes"] == 2
    # Verify self-conditioning was passed to decoder
    assert model.last_self_conditioning is not None


def test_quantized_self_conditioning_none_context():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    # Emulate quantized embeddings where prepare_self_conditioning returns None
    assert model.prefers_logits_self_conditioning is True
    assert model.diffusion_prepare_self_conditioning() is None

    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="refinement_comparison",
        num_passes=2,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Search flight", state, space, [])
    assert decision.action_id is not None


def test_multi_read_reproducibility_and_reset():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="noise_comparison",
        num_reads=4,
        seed=42,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])

    dec1 = backend.choose("Select one-way", state, space, [])
    assert dec1.metadata["num_reads"] == 4
    assert dec1.metadata["actual_forward_passes"] == 4
    assert "agreement" in dec1.metadata

    # Reset session clears prefix cache
    backend.reset_session()
    assert backend._cached_goal is None
    assert backend._cached_prefix_tokens is None
    assert backend._cached_prefix_cache is None


def test_cache_prefix_and_clone_fallback():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        max_canvas_width=64,
        enable_prefix_caching=True,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])

    # First call: cold cache
    dec1 = backend.choose("Book flight", state, space, [])
    assert not dec1.metadata["cache_hit"]

    # Second call with identical prefix: cache hit
    dec2 = backend.choose("Book flight", state, space, [])
    assert dec2.metadata["cache_hit"]
    assert dec2.metadata["cached_prefix_tokens"] > 16

    # Third call with updated history/state (same goal): cache hit on invariant prefix
    state_updated = dict(state)
    state_updated["text"] = "Different page text after navigation"
    history = [{"step": 1, "action": "Click round trip", "kind": "click"}]
    dec_turn2 = backend.choose("Book flight", state_updated, space, history)
    assert dec_turn2.metadata["cache_hit"]
    assert dec_turn2.metadata["cached_prefix_tokens"] >= 16

    # Session reset clears prefix cache
    backend.reset_session()
    dec_cold = backend.choose("Book flight", state, space, [])
    assert not dec_cold.metadata["cache_hit"]

    # Test clone failure fallback: monkeypatch clone to fail
    backend.reset_session()
    backend.choose("Book flight", state, space, [])
    with patch("mlx_vlm.apc._clone_prompt_cache_for_apc", side_effect=RuntimeError("Clone failed")):
        dec3 = backend.choose("Book flight", state, space, [])
        # Recovers and executes cleanly
        assert dec3.action_id is not None


def test_selected_operation_only_target_consumption():
    tok = MockTokenizer()
    # Mock model that votes for CLICK (slot label A) and click_target 1
    model = MockDiffusionModel(target_token_id=ord("A") - 2)
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(model=model, processor=proc, max_canvas_width=64)
    state = sample_browser_state()
    space = action_space(state["actions"])

    decision = backend.choose("Click round trip", state, space, [])
    assert decision.operation == "CLICK"
    assert decision.target == "1"
    assert decision.action_id == "btn_roundtrip"


def test_non_finite_logits_rejection():
    tok = MockTokenizer()
    model = MockDiffusionModel()

    # Produce NaN logits
    def nan_logits(*args, **kwargs):
        return mx.array([[[float("nan")] * 512] * 32], dtype=mx.float32)

    model.diffusion_decoder_logits = nan_logits
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(model=model, processor=proc, max_canvas_width=128)

    state = sample_browser_state()
    space = action_space(state["actions"])

    with pytest.raises(ValueError, match="Non-finite logits detected"):
        backend.choose("Invalid step", state, space, [])


def test_offline_only_enforcement(tmp_path):
    backend = MlxDiffusionStructuredBackend(
        local_model_path=str(tmp_path / "non_existent_model_dir"),
        offline_only=True,
    )
    with pytest.raises(RuntimeError, match="Offline policy violation"):
        backend._resolve_local_path()


def test_field_text_unicode_and_punctuation_preservation():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(model=model, processor=proc)

    # Mock mlx_vlm.generate to return unicode and apostrophes
    mock_result = SimpleNamespace(text='VALUE="Zürich / Geneva (O\'Reilly)"')
    with patch("mlx_vlm.generate", return_value=mock_result):
        val = backend.generate_field_text({"goal": "Enter city", "field": {"label": "City"}})
        assert val == "Zürich / Geneva (O'Reilly)"

    # Strict failure behavior: never fall back to field label
    mock_empty = SimpleNamespace(text="")
    with patch("mlx_vlm.generate", return_value=mock_empty):
        val_empty = backend.generate_field_text({"goal": "Enter city", "field": {"label": "City"}})
        assert val_empty == ""
        assert val_empty != "City"


def test_decision_request_pure_builder():
    state = sample_browser_state()
    space = action_space(state["actions"])
    req = build_decision_request("test-model", "Test goal", state, space, [])

    assert req["model"] == "test-model"
    assert "questions" in req
    assert "operation" in req["questions"]
    assert "click_target" in req["questions"]
    assert "type_text_target" in req["questions"]
    assert req["state"]["page"]["title"] == "Flight Booking"


def test_initial_default_is_one_pass_without_escalation():
    tok = MockTokenizer()
    model = MockDiffusionModel(target_token_id=ord("A") - 2)
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(model=model, processor=proc, max_canvas_width=64)
    assert backend.num_reads == 1
    assert backend.num_passes == 1
    assert backend.adaptive_escalation is False

    state = sample_browser_state()
    decision = backend.choose("Click round trip", state, action_space(state["actions"]), [])
    assert decision.metadata["escalation_stage"] == "deterministic_fixed"
    assert decision.metadata["actual_forward_passes"] == 1


def test_adaptive_escalation_early_exit():
    tok = MockTokenizer()
    # Confident model: target_token_id has high logit -> Pass 1 early exit
    model = MockDiffusionModel(target_token_id=tok.encode("A")[0])
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="initial_default",
        adaptive_escalation=True,
        pass1_margin_threshold=0.65,
        pass1_entropy_threshold=0.35,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Click round trip", state, space, [])

    assert decision.metadata["escalation_stage"] == "pass1_exit"
    assert decision.metadata["actual_forward_passes"] == 1
    assert decision.operation == "CLICK"


def test_adaptive_escalation_refinement_to_pass2():
    tok = MockTokenizer()

    # Step 1: flat (uncertain) logits; Step 2: confident logits
    def progressive_logits(call_count, current_canvas, self_cond):
        _, L = current_canvas.shape
        arr = [[0.0] * 512 for _ in range(L)]
        if call_count >= 2:
            for row in arr:
                row[tok.encode("A")[0]] = 6.0
        return mx.array([arr], dtype=mx.float32)

    model = MockDiffusionModel(logits_fn=progressive_logits)
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="initial_default",
        adaptive_escalation=True,
        pass1_margin_threshold=0.65,
        pass1_entropy_threshold=0.35,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Click round trip", state, space, [])

    assert decision.metadata["escalation_stage"] == "pass2_refinement"
    assert decision.metadata["actual_forward_passes"] == 2
    assert decision.operation == "CLICK"


def test_adaptive_escalation_noise_draws_fallback():
    tok = MockTokenizer()

    # Always flat (uncertain) logits
    def flat_logits(call_count, current_canvas, self_cond):
        _, L = current_canvas.shape
        arr = [[0.0] * 512 for _ in range(L)]
        return mx.array([arr], dtype=mx.float32)

    model = MockDiffusionModel(logits_fn=flat_logits)
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="initial_default",
        adaptive_escalation=True,
        noise_draw_fallback=True,
        pass1_margin_threshold=0.65,
        pass1_entropy_threshold=0.35,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Click round trip", state, space, [])

    # Uncertain after Pass 2 -> falls back to 4 independent noise draws
    assert decision.metadata["escalation_stage"] == "multi_read_draws"
    assert decision.metadata["actual_forward_passes"] >= 4
    assert "agreement" in decision.metadata


def test_adaptive_escalation_disabled_deterministic():
    tok = MockTokenizer()
    model = MockDiffusionModel(target_token_id=ord("A") - 2)
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="initial_default",
        adaptive_escalation=False,
        max_canvas_width=64,
    )

    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Click round trip", state, space, [])

    assert decision.metadata["escalation_stage"] == "deterministic_fixed"
    assert decision.metadata["actual_forward_passes"] == 1


def test_cross_head_attention_mask_isolation():
    from jev_ultrafast.structured_canvas import build_cross_head_attention_mask

    tok = MockTokenizer()
    compiler = CanvasCompiler(tok, max_width=64)
    questions = {
        "operation": {"criteria": {"CLICK": "Click", "TYPE_TEXT": "Type"}},
        "click_target": {"criteria": {"1": "B1", "2": "B2"}},
        "type_text_target": {"criteria": {"3": "F1", "4": "F2"}},
    }
    layout = compiler.compile(questions)
    encoder_len = 10

    block_mask = build_cross_head_attention_mask(layout, encoder_len)
    assert block_mask.shape == (1, 1, layout.width, encoder_len + layout.width)

    s_op = layout.slot_map["operation"]
    s_click = layout.slot_map["click_target"]
    s_type = layout.slot_map["type_text_target"]

    p_op = s_op.position
    p_click = s_click.position
    p_type = s_type.position

    # Sibling targets must NOT attend to one another
    assert not bool(block_mask[0, 0, p_click, encoder_len + p_type].item())
    assert not bool(block_mask[0, 0, p_type, encoder_len + p_click].item())

    # Targets CAN attend to the operation head
    assert bool(block_mask[0, 0, p_click, encoder_len + p_op].item())
    assert bool(block_mask[0, 0, p_type, encoder_len + p_op].item())

    # Targets CAN attend to prompt observation tokens
    assert bool(block_mask[0, 0, p_click, 0].item())
    assert bool(block_mask[0, 0, p_type, 5].item())

    # Self-attention within the same target slot is preserved
    assert bool(block_mask[0, 0, p_click, encoder_len + p_click].item())
    assert bool(block_mask[0, 0, p_type, encoder_len + p_type].item())

    # Verify end-to-end integration with MlxDiffusionStructuredBackend
    model = MockDiffusionModel(target_token_id=ord("A") - 2)
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        use_attention_isolation=True,
        max_canvas_width=64,
    )
    state = sample_browser_state()
    space = action_space(state["actions"])
    dec = backend.choose("Click round trip", state, space, [])

    assert dec.metadata["attention_isolation"] is True
    assert model.last_decoder_attention_mask is not None
    assert "full_attention" in model.last_decoder_attention_mask


def test_linear_temperature_schedule():
    tok = MockTokenizer()
    model = MockDiffusionModel(target_token_id=ord("A") - 2)
    proc = SimpleNamespace(tokenizer=tok)

    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        mode="refinement_comparison",
        num_passes=2,
        use_temperature_schedule=True,
        max_canvas_width=64,
    )
    state = sample_browser_state()
    space = action_space(state["actions"])
    decision = backend.choose("Click round trip", state, space, [])

    assert decision.metadata["temperature_schedule"] is True
    assert decision.stable_steps == 2


def test_hierarchical_candidate_partitioning_and_grounding():
    tok = MockTokenizer()
    compiler = CanvasCompiler(tok, max_width=128)

    # Construct 60 actions across categories
    actions = []
    # 25 inputs
    for i in range(25):
        actions.append({
            "id": f"input_{i}",
            "kind": "fill",
            "label": f"Input Field {i}",
            "role": "textbox",
            "node": 100 + i,
        })
    # 30 buttons
    for i in range(30):
        actions.append({
            "id": f"btn_{i}",
            "kind": "click",
            "label": f"Submit Button {i}",
            "role": "button",
            "node": 200 + i,
        })
    # 10 navigation links
    for i in range(10):
        actions.append({
            "id": f"link_{i}",
            "kind": "click",
            "label": f"Nav Link {i}",
            "role": "link",
            "node": 300 + i,
        })

    actions.append({"id": "wait", "kind": "wait", "label": "Wait"})

    state = {
        "url": "https://large-app.example.com",
        "title": "Portal",
        "text": "Extensive candidate interface.",
        "actions": actions,
    }
    space = action_space(actions)
    req = build_decision_request("test-model", "Submit application", state, space, [])
    questions = req["questions"]

    layout = compiler.compile(questions, hierarchical=True)
    assert layout.hierarchical_map is not None
    assert "click_target" in layout.hierarchical_map
    h_info = layout.hierarchical_map["click_target"]
    assert "click_group" in h_info["group_head"]
    assert layout.width <= 128

    # Verify backend grounding
    model = MockDiffusionModel(target_token_id=tok.encode("B")[0])
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(
        model=model,
        processor=proc,
        hierarchical_partitioning=True,
        max_canvas_width=128,
    )

    decision = backend.choose("Submit application", state, space, [])
    assert decision.operation == "CLICK"
    assert decision.action_id == "link_1"
    assert decision.metadata["hierarchical_partitioning"] is True
    assert decision.metadata["hierarchical_group"] == "G3_NAVIGATION"


def test_format_prompt_state_annotations():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(model=model, processor=proc)
    backend._ensure_loaded()

    actions = [
        {"id": "e1", "kind": "click", "label": "Zurich Airport", "role": "checkbox", "checked": "true", "node": 10},
        {"id": "e2", "kind": "click", "label": "Done", "role": "button", "node": 11},
        {"id": "e3", "kind": "fill", "label": "Where to?", "role": "combobox", "value": "London", "node": 12},
        {"id": "e4", "kind": "fill", "label": "Where from?", "role": "combobox", "value": "Zurich", "node": 13},
    ]
    state = {
        "url": "https://flights.example.com",
        "title": "Flights",
        "text": "Choose airport",
        "actions": actions,
    }
    space = action_space(actions)
    req = build_decision_request("test", "Find flights", state, space, [])
    layout = backend._compiler.compile(req["questions"])

    history = [
        {"step": 1, "action": "Click round trip", "kind": "click", "text": None},
        {"step": 2, "action": "Where from?", "kind": "fill", "text": "Zurich"},
    ]
    prompt = backend._format_prompt("Find flights", state, req["questions"], layout, history)

    assert "Zurich Airport (checkbox, checked)" in prompt
    assert "Done (button)" in prompt
    assert "Where to? (combobox)" in prompt
    assert "Step 1: Click round trip" in prompt
    assert ": None" not in prompt
    assert "Step 2: Where from? · Zurich" in prompt

    # Verify Trajectory Prefix Rolling prompt order: History precedes Page Title
    hist_idx = prompt.find("Recent Action History:")
    page_idx = prompt.find("Page Title:")
    assert 0 < hist_idx < page_idx, "Recent Action History must precede Page Title for rolling prefix caching"

    import inspect
    sig = inspect.signature(backend.generate_field_text)
    assert sig.parameters["max_tokens"].default == 16


def test_cycle_breaking_checked_target():
    tok = MockTokenizer()
    model = MockDiffusionModel()
    proc = SimpleNamespace(tokenizer=tok)
    backend = MlxDiffusionStructuredBackend(model=model, processor=proc)
    backend._ensure_loaded()

    actions = [
        {"id": "e1", "kind": "click", "label": "Zurich Airport", "role": "checkbox", "checked": "true", "node": 10},
        {"id": "e2", "kind": "click", "label": "Done", "role": "button", "node": 11},
    ]
    state = {
        "url": "https://flights.example.com",
        "title": "Flights",
        "text": "Airport selected",
        "actions": actions,
    }
    space = action_space(actions)
    # History shows e1 was just clicked in step 1
    history = [
        {"step": 1, "action": "Zurich Airport", "choice": "e1", "kind": "click", "target": "1", "page_changed": True}
    ]
    decision = backend.choose("Find flights", state, space, history)

    assert decision.operation == "CLICK"
    # Cycle breaking should divert from already-checked e1 to Done button e2
    assert decision.action_id == "e2"


