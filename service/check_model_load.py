import requests
import json
import uuid

API_URL = "http://127.0.0.1:5000/infer"   # Change if needed

def test_model_load():
    payload = {
        "text": "test message for model load checking",
        "request_id": str(uuid.uuid4())
    }

    print("\nSending request to:", API_URL)

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
    except Exception as e:
        print("\n❌ ERROR: Could not connect to API")
        print("Details:", e)
        return

    if response.status_code != 200:
        print("\n❌ API returned error:", response.status_code)
        print(response.text)
        return

    result = response.json()
    print("\n✅ API Response:")
    print(json.dumps(result, indent=2))

    if "output" in result:
        print("\n🎉 SUCCESS: Model is loaded and responding.")
    else:
        print("\n⚠️ WARNING: API responded but model output missing.")

if __name__ == "__main__":
    test_model_load()
