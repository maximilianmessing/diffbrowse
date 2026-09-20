"""Script to interact with the live demo server on http://127.0.0.1:8766 and extract telemetry."""

import json
import re
import urllib.request

PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"

def get_token():
    req = urllib.request.Request(BASE + "/")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    m = re.search(r'<meta name="demo-token" content="([^"]+)"', html)
    if not m:
        raise ValueError("Could not find demo token in HTML")
    return m.group(1)

def post_api(endpoint, body, token):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/{endpoint}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Demo-Token": token,
            "Host": f"127.0.0.1:{PORT}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    token = get_token()
    print(f"Acquired demo token: {token[:8]}...")

    print("\n" + "=" * 60)
    print("STEP 1: Reset demo with backend='mlx_direct' (Local DiffusionGemma)")
    print("=" * 60)
    reset_res = post_api(
        "reset",
        {
            "scenario": "travel",
            "goal": "Find a Design stay in Lisbon with Free cancellation and open Casa Flora.",
            "backend": "mlx_direct",
        },
        token,
    )
    print(f"Status       : {reset_res.get('status')}")
    print(f"Backend      : {reset_res.get('backend')}")
    print(f"Page URL     : {reset_res.get('page', {}).get('url')}")
    print(f"Elements     : {len(reset_res.get('elements', []))}")

    print("\n" + "=" * 60)
    print("STEP 2: Predict Decision 1 (Cold Prefill)")
    print("=" * 60)
    step1_res = post_api("predict", {}, token)
    d1 = step1_res.get("decision", {})
    print(f"Choice       : {d1.get('choice')} ({d1.get('operation')} -> {d1.get('target')})")
    print(f"Confidence   : {d1.get('confidence')}")
    print(f"Entropy      : {d1.get('entropy')}")
    print(f"Top-2 Margin : {d1.get('top2_margin')}")
    print(f"Prefill Lat  : {d1.get('prefill_ms')} ms")
    print(f"Decoder Lat  : {d1.get('decoder_ms')} ms")
    print(f"Total Model  : {d1.get('total_model_ms')} ms")
    print(f"Cache Hit    : {d1.get('metadata', {}).get('cache_hit')}")

    print("\n" + "=" * 60)
    print("STEP 3: Execute Action 1")
    print("=" * 60)
    act_res = post_api("act", {"fingerprint": step1_res["page"]["fingerprint"]}, token)
    print(f"Executed     : {act_res['history'][-1]['action']}")
    print(f"Page Changed : {act_res['history'][-1]['page_changed']}")

    print("\n" + "=" * 60)
    print("STEP 4: Predict Decision 2 (Multi-Turn KV Cache Hit)")
    print("=" * 60)
    step2_res = post_api("predict", {}, token)
    d2 = step2_res.get("decision", {})
    print(f"Choice       : {d2.get('choice')} ({d2.get('operation')} -> {d2.get('target')})")
    print(f"Confidence   : {d2.get('confidence')}")
    print(f"Entropy      : {d2.get('entropy')}")
    print(f"Top-2 Margin : {d2.get('top2_margin')}")
    print(f"Prefill Lat  : {d2.get('prefill_ms')} ms")
    print(f"Decoder Lat  : {d2.get('decoder_ms')} ms")
    print(f"Total Model  : {d2.get('total_model_ms')} ms")
    print(f"Cache Hit    : {d2.get('metadata', {}).get('cache_hit')}")

    # Save telemetry results for reporting
    telemetry = {
        "step1": d1,
        "step2": d2,
        "page_elements": len(reset_res.get("elements", [])),
    }
    with open("results/live_demo_telemetry.json", "w") as f:
        json.dump(telemetry, f, indent=2)
    print("\nLive demo telemetry saved to results/live_demo_telemetry.json")

if __name__ == "__main__":
    main()
