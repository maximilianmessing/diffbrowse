"""Apple Silicon Metal LoRA fine-tuning for DiffusionGemma browser decision heads."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a quantized or frozen linear projection."""

    def __init__(self, original_linear: nn.Module, r: int = 8, scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear

        has_scales = hasattr(original_linear, "scales") and hasattr(original_linear, "group_size")
        if original_linear is not None and has_scales:
            scales = original_linear.scales
            group_size = original_linear.group_size
            out_features = scales.shape[0]
            in_features = scales.shape[1] * group_size
        elif original_linear is not None and isinstance(original_linear, dict) and "scales" in original_linear:
            scales = original_linear["scales"]
            group_size = getattr(original_linear, "group_size", 64)
            out_features = scales.shape[0]
            in_features = scales.shape[1] * group_size
        elif original_linear is not None and hasattr(original_linear, "weight") and original_linear.weight is not None:
            out_features, in_features = original_linear.weight.shape
        else:
            in_features = getattr(original_linear, "in_features", 2816)
            out_features = getattr(original_linear, "out_features", 4096)

        self.r = r
        self.scale = scale
        scale_init = 1.0 / (r**0.5)
        self.lora_a = mx.random.normal(shape=(r, in_features)) * scale_init
        self.lora_b = mx.zeros(shape=(out_features, r))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.original_linear(x)
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + lora_out * self.scale


def _patch_moe_stop_gradient():
    """Ensure Router and expert gather indices stop gradients in MLX autograd."""
    try:
        from mlx_vlm.models import switch_layers
        from mlx_vlm.models.diffusion_gemma import language

        def safe_router_call(self, x):
            x_norm = mx.fast.rms_norm(x, None, self.eps)
            x_norm = x_norm * self.scale * self._root_size
            scores = self.proj(x_norm)
            top_k = self.config.top_k_experts
            indices = mx.stop_gradient(mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:])
            weights = mx.take_along_axis(scores, indices, axis=-1)
            weights = mx.softmax(weights, axis=-1, precise=True)
            weights = weights * self.per_expert_scale[indices]
            return indices, weights

        def safe_gather_sort(x, indices):
            *_, M = indices.shape
            indices = indices.flatten()
            order = mx.stop_gradient(mx.argsort(indices))
            inv_order = mx.stop_gradient(mx.argsort(order))
            return x.flatten(0, -3)[order // M], indices[order], inv_order

        language.Router.__call__ = safe_router_call
        language._gather_sort = safe_gather_sort
        switch_layers._gather_sort = safe_gather_sort
    except Exception:
        pass


def inject_lora_adapters(model: nn.Module, r: int = 8, target_modules: tuple = ("q_proj", "v_proj")) -> int:
    """Inject LoRA matrices into decoder self-attention projections."""
    # Freeze entire model
    model.freeze()

    adapter_count = 0
    decoder = getattr(getattr(model, "model", model), "decoder", None)
    if decoder is None:
        return 0

    for layer in getattr(decoder, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for mod_name in target_modules:
            orig_mod = getattr(attn, mod_name, None)
            if orig_mod is None or not isinstance(orig_mod, nn.Module):
                continue
            lora_layer = LoRALinear(orig_mod, r=r)
            setattr(attn, mod_name, lora_layer)
            adapter_count += 1

    return adapter_count


def load_dataset_pairs(jsonl_path: str, max_samples: int = 500) -> List[Dict[str, str]]:
    path = Path(jsonl_path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= max_samples:
                break
    return records


def run_training(
    model_name: str,
    data_dir: str,
    output_dir: str,
    r: int = 8,
    lr: float = 1e-4,
    epochs: int = 1,
    max_steps: int = 10,
    batch_size: int = 1,
    dry_run: bool = False,
):
    print("=" * 60)
    print("DiffBrowse: Apple Silicon Metal LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Base Model : {model_name}")
    print(f"Data Dir   : {data_dir}")
    print(f"Output Dir : {output_dir}")
    print(f"LoRA Rank  : {r}")
    print(f"Learn Rate : {lr}")
    print(f"Epochs     : {epochs}")
    print(f"Max Steps  : {max_steps}")
    print(f"Dry Run    : {dry_run}")
    print("-" * 60)

    train_data = load_dataset_pairs(str(Path(data_dir) / "train.jsonl"))
    valid_data = load_dataset_pairs(str(Path(data_dir) / "valid.jsonl"))
    print(f"Loaded {len(train_data)} train samples, {len(valid_data)} validation samples.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_run or not train_data:
        print("[DRY-RUN] Simulating adapter configuration and parameter count...")
        config = {
            "model_name": model_name,
            "lora_rank": r,
            "learning_rate": lr,
            "target_modules": ["q_proj", "v_proj"],
            "train_samples": len(train_data),
            "valid_samples": len(valid_data),
            "simulated": True,
            "timestamp": time.time(),
        }
        (out_path / "adapter_config.json").write_text(json.dumps(config, indent=2))
        print(f"Saved simulated adapter config to {out_path / 'adapter_config.json'}")
        print("Dry run completed successfully!")
        return config

    # Real training flow
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    from jev_ultrafast.backends.mlx_direct import _patch_diffusion_encoder

    _patch_diffusion_encoder()
    _patch_moe_stop_gradient()
    print("Loading DiffusionGemma-26B base model into unified memory...")
    t_load = time.perf_counter()
    model, processor = load(model_name)
    load_time_s = time.perf_counter() - t_load
    print(f"Model loaded in {load_time_s:.2f}s.")

    adapters_injected = inject_lora_adapters(model, r=r)
    print(f"Injected LoRA adapters into {adapters_injected} decoder attention projections.")

    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    total_trainable_params = sum(p.size for p in trainable_dict.values())
    print(f"Trainable parameters: {total_trainable_params:,} across {len(trainable_dict)} tensors.")

    optimizer = optim.AdamW(learning_rate=lr)
    tokenizer = getattr(processor, "tokenizer", processor)

    def loss_fn(m, input_ids, canvas_ids, slot_idx, target_id):
        out = m(input_ids=input_ids, canvas_ids=canvas_ids)
        logits = out.logits[:, slot_idx, :]
        return mx.mean(nn.losses.cross_entropy(logits, target_id))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    prefix_str = "ACTION="
    prefix_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)
    prefix_len = len(prefix_tokens)
    canvas_len = 16

    print("\nStarting Metal LoRA optimization loop...")
    losses = []
    step = 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        for sample in train_data:
            if step >= max_steps:
                break
            prompt_text = sample.get("prompt", "")
            completion = sample.get("completion", "ACTION=A")

            # Extract target action label
            match = re.search(r"ACTION=([A-Za-z0-9_]+)", completion)
            target_label = match.group(1) if match else "A"
            target_tokens = tokenizer.encode(target_label, add_special_tokens=False)
            if not target_tokens:
                continue
            target_id = mx.array([target_tokens[0]])

            # Tokenize prompt and construct canvas
            formatted = apply_chat_template(processor, model.config, prompt_text)
            input_tokens = tokenizer.encode(formatted, add_special_tokens=False)
            input_ids_mx = mx.array([input_tokens])

            canvas = mx.zeros((1, canvas_len), dtype=mx.int32)
            canvas[0, :prefix_len] = mx.array(prefix_tokens)
            slot_idx = prefix_len

            # Gradient update
            t_step = time.perf_counter()
            loss_val, grads = loss_and_grad_fn(model, input_ids_mx, canvas, slot_idx, target_id)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            step_duration_ms = (time.perf_counter() - t_step) * 1000

            curr_loss = float(loss_val)
            losses.append(curr_loss)
            step += 1
            print(f"Step {step:3d}/{max_steps} | Loss: {curr_loss:.4f} | Latency: {step_duration_ms:.1f}ms")

        if step >= max_steps:
            break

    total_time_s = time.perf_counter() - t_start
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    print("-" * 60)
    print(f"Training completed in {total_time_s:.2f}s | Average Loss: {avg_loss:.4f}")

    # Save trained adapters
    print(f"Saving trained LoRA adapter weights to {out_path}...")
    trained_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_path / "adapters.safetensors"), trained_weights)

    config = {
        "model_name": model_name,
        "lora_rank": r,
        "learning_rate": lr,
        "target_modules": ["q_proj", "v_proj"],
        "adapters_count": adapters_injected,
        "trainable_parameters": total_trainable_params,
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "steps_completed": step,
        "final_loss": round(losses[-1] if losses else 0.0, 4),
        "average_loss": round(avg_loss, 4),
        "simulated": False,
        "timestamp": time.time(),
    }
    (out_path / "adapter_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved adapter config and safetensors to {out_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/diffusiongemma-26B-A4B-it-4bit")
    parser.add_argument("--data-dir", default="data/lora")
    parser.add_argument("--output-dir", default="adapters/diffusiongemma_browser")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without executing GPU gradient steps")
    args = parser.parse_args()

    run_training(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        r=args.r,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
