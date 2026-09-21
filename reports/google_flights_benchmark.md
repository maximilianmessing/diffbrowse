# Bounded Local Evaluation: Training-Free Structured Diffusion

- **Dataset**: `fixtures/recorded_decisions.jsonl` (50 states)
- **Evaluation Date**: 2026-09-21 15:37:40
- **Tuning Set**: 25 states | **Held-out Set**: 25 states
- **Engine**: DiffusionGemma-26B-A4B-it-4bit (Local MLX Metal)

## Summary Comparison

| Configuration | Reads | Passes | Overall Acc | Tuning Acc | Held-out Acc | Invalid Rate | False Done | p50 Latency | p95 Latency | Mean Prefill | Mean Decode | Passes/Dec | Peak Metal MB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **initial_default** | 1 | 1 | 90.0% | 92.0% | 88.0% | 0.0% | 4.0% | 483.61 ms | 21126.27 ms | 1496.13 ms | 974.14 ms | 1.0 | 17352.38 MB |
| **refinement_comparison** | 1 | 2 | 82.0% | 76.0% | 88.0% | 0.0% | 8.0% | 605.76 ms | 635.83 ms | 384.39 ms | 246.84 ms | 2.0 | 17352.38 MB |
| **noise_comparison** | 4 | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 826.67 ms | 45987.08 ms | 3107.83 ms | 3627.32 ms | 4.0 | 17352.38 MB |
| **adaptive_escalation** | 1 | 1-2 (dyn) | 98.0% | 100.0% | 96.0% | 0.0% | 2.0% | 491.5 ms | 1345.02 ms | 386.93 ms | 185.48 ms | 1.56 | 17352.38 MB |

## Google Flights Stage Breakdown

| Stage | **initial_default** | **refinement_comparison** | **noise_comparison** | **adaptive_escalation** |
| :--- | :---: | :---: | :---: | :---: |
| **Airport Selection** | 100.0% (6/6) | 100.0% (6/6) | 100.0% (6/6) | 100.0% (6/6) |
| **Done Verification** | 100.0% (7/7) | 100.0% (7/7) | 100.0% (7/7) | 100.0% (7/7) |
| **Origin / Destination** | 100.0% (18/18) | 94.4% (17/18) | 100.0% (18/18) | 100.0% (18/18) |
| **Other Navigation** | 25.0% (1/4) | 50.0% (2/4) | 100.0% (4/4) | 75.0% (3/4) |
| **Ticket Type** | 86.7% (13/15) | 60.0% (9/15) | 100.0% (15/15) | 100.0% (15/15) |

## Cross-Head Interference Diagnostic

- **Head Agreement Rate**: 100.0%
- **Interference Disagreement Rate**: 0.0%
- **Total Evaluated**: 50 states

### Interpretation & Architectural Grounding
- **Initial Default ($1 \times 1$)**: Runs a single forward pass over the noisy canvas with logit slicing. Fastest latency with zero iterative overhead.
- **Refinement Comparison ($1 \times 2$)**: Adds a second pass with pinned token restoration and masked self-conditioning. Zeroes out self-conditioning at fixed template positions while allowing answers to refine.
- **Noise Comparison ($4 \times 1$)**: Averages 4 independent random noisy draws sharing the prefilled KV cache, capturing cross-sample variance.
- **Interference Diagnostic**: Validates that shared attention across question heads does not induce catastrophic head corruption relative to isolated reads.

> [!NOTE]
> Latencies and accuracies represent bounded offline replays over authentic browser fixtures without remote cloud transit.