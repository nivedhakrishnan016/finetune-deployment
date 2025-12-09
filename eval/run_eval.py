import argparse
import requests
import json
import time
from typing import List, Dict


def load_test_data(path: str) -> List[Dict[str, str]]:
    """Load evaluation questions from JSONL or JSON."""
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)

    else:
        raise ValueError("Test data must be JSON or JSONL")


def call_model(url: str, text: str) -> str:
    """POST to model inference API."""
    try:
        payload = {"text": text}
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code != 200:
            return f"[ERROR] HTTP {response.status_code}: {response.text}"

        data = response.json()
        return data.get("response") or data.get("output") or data.get("text") or str(data)

    except Exception as e:
        return f"[EXCEPTION] {str(e)}"


def evaluate(test_data: List[Dict], ft_url: str, base_url: str):
    """Run evaluation."""
    results = []

    for idx, item in enumerate(test_data, start=1):
        prompt = item.get("prompt") or item.get("question") or item.get("input")

        print(f"\n=== Test {idx} ===")
        print("Prompt:", prompt)

        # Baseline
        t1 = time.time()
        baseline_out = call_model(base_url, prompt)
        t2 = time.time()

        # Fine-tuned
        t3 = time.time()
        ft_out = call_model(ft_url, prompt)
        t4 = time.time()

        result = {
            "prompt": prompt,
            "baseline_response": baseline_out,
            "fine_tuned_response": ft_out,
            "baseline_latency": round(t2 - t1, 3),
            "fine_tuned_latency": round(t4 - t3, 3),
        }
        results.append(result)

        print("Baseline:", baseline_out[:180])
        print("Fine-Tuned:", ft_out[:180])
        print("Baseline Latency:", result["baseline_latency"])
        print("Fine-Tuned Latency:", result["fine_tuned_latency"])

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fine-tuned-url", required=True,
                        help="URL of fine-tuned inference endpoint")

    parser.add_argument("--baseline-url", required=True,
                        help="URL of baseline inference endpoint")

    parser.add_argument("--test-data-path", default="test_data.jsonl",
                        help="Path to prompts JSON/JSONL")

    args = parser.parse_args()

    print("Loading test data...")
    test_data = load_test_data(args.test_data_path)

    print(f"Loaded {len(test_data)} test cases.")
    print("Starting evaluation...")

    results = evaluate(test_data, args.fine_tuned_url, args.baseline_url)

    # Write output file
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results → eval_results.json")
    print("Done!")


if __name__ == "__main__":
    main()
