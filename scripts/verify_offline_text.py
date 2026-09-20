"""Verify offline text generation using resident DiffusionGemma weights."""

import time

from jev_ultrafast.backends import MlxDiffusionDirectBackend


def main():
    print("=" * 60)
    print("Verifying 100% Air-Gapped Field Text Generation on Metal")
    print("=" * 60)
    backend = MlxDiffusionDirectBackend()

    context = {
        "goal": "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy.",
        "field": {"label": "Where from? (Departure airport)", "role": "textbox", "value": ""},
        "page": {"title": "Google Flights", "text": "Find and book flights with Google Flights"},
        "recent_actions": [],
    }

    print(f"Goal : {context['goal']}")
    print(f"Field: {context['field']['label']}")
    print("\nGenerating field text using DiffusionGemma...")
    tic = time.perf_counter()
    value = backend.generate_field_text(context)
    elapsed_ms = (time.perf_counter() - tic) * 1000

    print("-" * 60)
    print(f"Generated Text Value : '{value}'")
    print(f"Generation Latency   : {elapsed_ms:.1f} ms (includes cold weight load)")
    print("-" * 60)

    # Second warm call in resident memory
    context2 = {
        "goal": "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy.",
        "field": {"label": "Where to? (Destination airport)", "role": "textbox", "value": ""},
        "page": {"title": "Google Flights", "text": "Find and book flights with Google Flights"},
        "recent_actions": [{"action": "Type in Where from?", "text": "Zurich"}],
    }
    print(f"\nWarm Field: {context2['field']['label']}")
    tic2 = time.perf_counter()
    value2 = backend.generate_field_text(context2)
    warm_ms = (time.perf_counter() - tic2) * 1000
    print(f"Generated Text Value : '{value2}'")
    print(f"Warm Generation Lat  : {warm_ms:.1f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
