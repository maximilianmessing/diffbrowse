"""Unit tests for backend abstractions, hybrid routing, and replay runner."""

from unittest.mock import Mock

from jev_ultrafast.backends import Decision, HybridBackend, JevBackend
from jev_ultrafast.recorder import DecisionRecorder
from scripts.replay_decisions import run_replay


def sample_page():
    return {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search Flights",
        "fingerprint": "fp_123",
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }


def test_jev_backend_computes_entropy_and_margin(monkeypatch):
    from jev_ultrafast import model

    def mock_post(_url, _key, _body):
        return {
            "model": "jev-test",
            "answers": {
                "operation": {
                    "choice": "CLICK",
                    "confidence": 0.9,
                    "probabilities": {
                        "CLICK": 0.9,
                        "TYPE_TEXT": 0.0,
                        "WAIT": 0.1,
                        "DONE": 0.0,
                        "BLOCKED": 0.0,
                    },
                },
                "click_target": {"choice": "2", "confidence": 1.0, "probabilities": {"2": 1.0}},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test_key")
    monkeypatch.setattr(model, "post_json", mock_post)

    backend = JevBackend()
    p = sample_page()
    space = model.action_space(p["actions"])
    decision = backend.choose("Search flights", p, space, [])

    assert decision.action_id == "e2"
    assert decision.operation == "CLICK"
    assert decision.target == "2"
    assert decision.confidence == 0.9
    assert decision["choice"] == "e2"  # dict access compatibility
    assert decision.raw_top_probability == 1.0
    assert decision.top2_margin == 1.0  # Only 1 candidate under CLICK
    assert decision.entropy == 0.0


def test_hybrid_backend_routes_locally_when_confident():
    local_backend = Mock()
    local_decision = Decision(
        action_id="e2",
        operation="CLICK",
        target="2",
        confidence=0.95,
        probabilities={"e2": 0.95, "e1": 0.05},
        raw_top_probability=0.95,
        entropy=0.10,  # Below threshold
        top2_margin=0.90,  # Above threshold
        stable_steps=2,
        backend="mlx_direct",
    )
    local_backend.choose.return_value = local_decision

    remote_backend = Mock()
    hybrid = HybridBackend(
        local_backend=local_backend,
        remote_backend=remote_backend,
        entropy_threshold=0.35,
        margin_threshold=0.50,
        min_stable_steps=2,
    )

    p = sample_page()
    res = hybrid.choose("Search", p, (), [])
    assert res.action_id == "e2"
    assert res.metadata.get("hybrid_route") == "local"
    local_backend.choose.assert_called_once()
    remote_backend.choose.assert_not_called()


def test_hybrid_backend_falls_back_to_remote_when_uncertain():
    local_backend = Mock()
    # High entropy / low margin decision
    local_decision = Decision(
        action_id="e1",
        operation="TYPE_TEXT",
        target="1",
        confidence=0.52,
        probabilities={"e1": 0.52, "e2": 0.48},
        raw_top_probability=0.52,
        entropy=0.69,  # High entropy
        top2_margin=0.04,  # Low margin
        stable_steps=1,
        backend="mlx_direct",
    )
    local_backend.choose.return_value = local_decision

    remote_backend = Mock()
    remote_decision = Decision(
        action_id="e2",
        operation="CLICK",
        target="2",
        confidence=0.99,
        probabilities={"e2": 0.99, "e1": 0.01},
        backend="jev",
    )
    remote_backend.choose.return_value = remote_decision

    hybrid = HybridBackend(
        local_backend=local_backend,
        remote_backend=remote_backend,
        entropy_threshold=0.35,
        margin_threshold=0.50,
        min_stable_steps=2,
    )

    p = sample_page()
    res = hybrid.choose("Search", p, (), [])
    assert res.action_id == "e2"
    assert res.metadata.get("hybrid_route") == "fallback_remote"
    local_backend.choose.assert_called_once()
    remote_backend.choose.assert_called_once()


def test_decision_recorder_and_replay_runner(tmp_path):
    record_file = tmp_path / "decisions.jsonl"
    recorder = DecisionRecorder(record_file)
    p = sample_page()

    dec = {
        "choice": "e2",
        "operation": "CLICK",
        "target": "2",
        "confidence": 0.95,
        "probabilities": {"e2": 0.95, "e1": 0.05},
        "latency_ms": 50,
    }
    recorder.record_decision(
        task_id="t1",
        goal="Click go",
        page=p,
        history=[],
        decision=dec,
        executed_action="e2",
    )

    loaded = DecisionRecorder.load_dataset(record_file)
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "t1"
    assert loaded[0]["executed_action"] == "e2"

    # Test replay runner
    mock_backend = Mock()
    mock_backend.name = "mock"
    mock_backend.choose.return_value = Decision(
        action_id="e2",
        operation="CLICK",
        target="2",
        confidence=0.95,
        probabilities={"e2": 0.95, "e1": 0.05},
        total_decision_ms=25.0,
        prefill_ms=10.0,
        decoder_ms=15.0,
    )

    summary = run_replay(mock_backend, loaded)
    assert summary["total_decisions"] == 1
    assert summary["action_accuracy"] == 1.0
    assert summary["agreement_with_jev"] == 1.0
    assert summary["p50_total_ms"] == 25.0


def test_torch_backend_device_resolution(monkeypatch):
    from jev_ultrafast.backends.torch_direct import TorchDiffusionDirectBackend
    from jev_ultrafast.backends.torch_structured import (
        TorchDiffusionStructuredBackend,
        _clone_past_key_values,
    )

    backend = TorchDiffusionDirectBackend(device="cuda:0")
    assert backend._resolve_device() == "cuda:0"

    backend_auto = TorchDiffusionDirectBackend()
    dev = backend_auto._resolve_device()
    assert dev in ("cuda", "cpu")

    # Structured PyTorch backend
    struct_backend = TorchDiffusionStructuredBackend(device="cuda:0")
    assert struct_backend._resolve_device() == "cuda:0"
    assert struct_backend.name == "torch_structured"

    struct_auto = TorchDiffusionStructuredBackend()
    assert struct_auto._resolve_device() in ("cuda", "cpu")

    # Test reset_session
    struct_backend._cached_goal = "Goal"
    struct_backend._cached_prefix_tokens = [1, 2, 3]
    struct_backend.reset_session()
    assert struct_backend._cached_goal is None
    assert struct_backend._cached_prefix_tokens is None

    # Test _clone_past_key_values
    assert _clone_past_key_values(None) is None
    mock_kv = Mock()
    mock_kv.copy.return_value = "cloned_cache"
    assert _clone_past_key_values(mock_kv) == "cloned_cache"


def test_agent_backend_resolution(monkeypatch):
    from jev_ultrafast.agent import Agent

    monkeypatch.setattr("jev_ultrafast.agent.Browser", Mock())

    # Direct string backend specification
    agent_torch = Agent("https://example.test", "Goal", backend="torch_structured")
    assert agent_torch.backend.name == "torch_structured"

    agent_mlx = Agent("https://example.test", "Goal", backend="mlx_structured")
    assert agent_mlx.backend.name == "mlx_structured"


