# service/test_client.py (CLEAN VERSION for EC2 or Docker)

import requests
import time
import statistics
import argparse
import json
import os
import sys
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:5000"

TEST_CASES = [
    {"text": "Please process my refund; the item arrived damaged.", "label": "refund_request"},
    {"text": "I can't log in to my account. It says my password is wrong.", "label": "account_access"},
    {"text": "When will my order ship? It's been five days.", "label": "shipping_delay"},
    {"text": "I have a question about the features of the new X-Pro drone.", "label": "product_question"},
    {"text": "The latest patch broke my connectivity. I need help fixing it.", "label": "tech_support"},
    {"text": "My statement has two charges for the same item.", "label": "billing_issue"},
]

# ---------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------

def test_health_check():
    print("\n--- Testing Health Check (GET /healthz) ---")
    try:
        response = requests.get(BASE_URL + "/healthz", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Status Code: {response.status_code}")
        assert data.get("status") == "ok"
        print("PASS: Health check OK")
    except Exception as e:
        print(f"FAIL: {e}")


def test_missing_text_error():
    print("\n--- Testing Missing Text Error (POST /infer) ---")
    payload = {"wrong_key": "abc"}

    try:
        response = requests.post(BASE_URL + "/infer", json=payload, timeout=10)
        data = response.json()
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 400
        print("PASS: Correct 400 error received")
    except Exception as e:
        print(f"FAIL: {e}")


def test_successful_classification():
    print("\n--- Testing Successful Classification (POST /infer) ---")
    payload = {"text": "How do I install the software?"}

    try:
        response = requests.post(BASE_URL + "/infer", json=payload, timeout=30)
        data = response.json()
        print("Status:", response.status_code)
        print("Output:", data)
        assert "output" in data
        print("PASS: Classification OK")
    except Exception as e:
        print(f"FAIL: {e}")

# ---------------------------------------------------------------
# Full Evaluation
# ---------------------------------------------------------------

def call_once(url: str, payload: Dict[str, Any]):
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=90)
        latency = (time.time() - t0) * 1000
        return r.status_code, r.json(), latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return 500, {"error": str(e)}, latency


def run_evaluation(url: str, num_runs: int):
    print(f"\n--- Running evaluation: {num_runs} requests ---")

    results = []
    expanded_cases = (TEST_CASES * ((num_runs // len(TEST_CASES)) + 1))[:num_runs]

    for i, case in enumerate(expanded_cases):
        status, body, lat = call_once(url, {"text": case["text"]})

        pred = body.get("output", "ERROR") if status == 200 else "ERROR"
        correct = pred == case["label"]

        results.append({
            "id": i+1,
            "text": case["text"],
            "ground_truth": case["label"],
            "prediction": pred,
            "correct": correct,
            "latency_ms": lat
        })

        print(f"Req {i+1:03d}: {lat:.1f} ms | GT={case['label']} | Pred={pred}")

    successful = [r for r in results if r["prediction"] != "ERROR"]

    if not successful:
        print("No successful requests!")
        return

    latencies = [r["latency_ms"] for r in successful]

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print("\n--- SUMMARY ---")
    print("Accuracy:", sum(r["correct"] for r in successful) / len(successful))
    print("P50:", p50)
    print("P95:", p95)

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to eval/results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=f"{BASE_URL}/infer")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    # Smoke tests
    test_health_check()
    test_missing_text_error()
    test_successful_classification()

    # Evaluation
    run_evaluation(args.url, args.n)


if __name__ == "__main__":
    print("Waiting 15s for model to load...")
    time.sleep(15)
    main()
