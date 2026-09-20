"""Unit tests for multi-turn KV session caching and LoRA fine-tuning utilities."""

import json
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from jev_ultrafast.action_candidates import LabelPool
from jev_ultrafast.backends.mlx_direct import MlxDiffusionDirectBackend
from scripts.export_lora_dataset import format_decision_for_lora
from scripts.train_lora import LoRALinear, run_training


def test_session_prefix_cache_reset():
    backend = MlxDiffusionDirectBackend(enable_prefix_caching=True)
    backend._cached_goal = "Find flights to London"
    backend._cached_prefix_tokens = [1, 2, 3, 4, 5]
    backend._cached_prefix_cache = [MagicMock(), MagicMock()]

    assert backend._cached_goal is not None
    assert backend._cached_prefix_tokens is not None
    assert backend._cached_prefix_cache is not None

    backend.reset_session()

    assert backend._cached_goal is None
    assert backend._cached_prefix_tokens is None
    assert backend._cached_prefix_cache is None


def test_format_decision_for_lora():
    vocab = {chr(i): i for i in range(ord("A"), ord("Z") + 1)}
    vocab.update({chr(i): i + 30 for i in range(ord("a"), ord("z") + 1)})
    vocab.update({chr(i): i + 60 for i in range(ord("0"), ord("9") + 1)})
    for idx in range(300):
        vocab[f"w{idx}"] = 200 + idx
    inv_vocab = {v: k for k, v in vocab.items()}

    class DummyTokenizer:
        def __init__(self):
            self.all_special_ids = [0]
            self.vocab_size = 600

        def encode(self, text, add_special_tokens=False):
            return [vocab[text]] if text in vocab else [500]

        def decode(self, ids):
            return "".join(inv_vocab.get(i, "") for i in ids)

    pool = LabelPool(DummyTokenizer())
    rec = {
        "task_id": "test_1",
        "goal": "Click on flight search",
        "title": "Google Flights",
        "url": "https://flights.google.com",
        "text": "Find cheap flights",
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Origin", "role": "textbox", "node": 10},
            {"id": "e2", "kind": "click", "label": "Search Flights", "role": "button", "node": 20},
        ],
        "executed_action": "e2",
        "history": [],
    }

    formatted = format_decision_for_lora(rec, pool)
    assert formatted is not None
    assert "Goal: Click on flight search" in formatted["prompt"]
    assert formatted["completion"].startswith("ACTION=")
    assert formatted["expected_action"] == "e2"


def test_lora_linear_forward():
    in_dim = 16
    out_dim = 32
    r = 4
    base_linear = nn.Linear(in_dim, out_dim)
    lora = LoRALinear(base_linear, r=r, scale=2.0)

    x = mx.random.normal(shape=(2, in_dim))
    out = lora(x)

    assert out.shape == (2, out_dim)
    # Check that gradients flow to lora_a and lora_b
    def loss_fn(mod, inp):
        return mx.mean(mod(inp))

    loss_and_grad = nn.value_and_grad(lora, loss_fn)
    loss, grads = loss_and_grad(lora, x)
    assert "lora_a" in grads
    assert "lora_b" in grads


def test_lora_dry_run_training(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "adapters"
    data_dir.mkdir()

    # Create tiny dummy train/valid sets
    sample = {
        "prompt": "Goal: Book flight\nACTION=1",
        "completion": "ACTION=1",
        "expected_action": "e1",
    }
    with open(data_dir / "train.jsonl", "w") as f:
        f.write(json.dumps(sample) + "\n")
    with open(data_dir / "valid.jsonl", "w") as f:
        f.write(json.dumps(sample) + "\n")

    run_training(
        model_name="mlx-community/diffusiongemma-26B-A4B-it-4bit",
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        r=4,
        dry_run=True,
    )

    config_path = output_dir / "adapter_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["lora_rank"] == 4
    assert "mlx-community" in config["model_name"]


def test_field_text_with_local_backend(monkeypatch):
    from jev_ultrafast.model import field_text

    # Ensure no remote API key is present
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)

    mock_backend = MagicMock()
    mock_backend.name = "mlx_direct"
    mock_backend.model_name = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
    mock_backend.generate_field_text.return_value = "Zurich"

    context = {
        "goal": "Find flights from Zurich to London",
        "field": {"label": "Departure airport", "role": "textbox"},
        "page": {"title": "Flights", "text": "Search"},
        "recent_actions": [],
    }

    val, meta = field_text(context, backend=mock_backend)
    assert val == "Zurich"
    assert "local" in meta["model"]
    assert meta["latency_ms"] >= 0
    mock_backend.generate_field_text.assert_called_once_with(context)


def test_hybrid_backend_delegates_field_text():
    from jev_ultrafast.backends.hybrid import HybridBackend

    mock_local = MagicMock()
    mock_local.generate_field_text.return_value = "London"
    mock_remote = MagicMock()

    hybrid = HybridBackend(local_backend=mock_local, remote_backend=mock_remote)
    context = {"goal": "Fly to London", "field": {"label": "Destination"}}
    val = hybrid.generate_field_text(context)

    assert val == "London"
    mock_local.generate_field_text.assert_called_once_with(context, max_tokens=32)


def test_pass1_early_exit_attributes():
    backend = MlxDiffusionDirectBackend(
        pass1_margin_threshold=0.70,
        pass1_entropy_threshold=0.20,
    )
    assert backend.pass1_margin_threshold == 0.70
    assert backend.pass1_entropy_threshold == 0.20

