# test_api_enhanced.py
import requests
from pprint import pprint

API_URL = "https://de89-197-211-58-38.ngrok-free.app/predict"
TEST_DATA = {
    "STATE": "Anambra",
    "LGA": "Ogbaru",
    "DAY_OF_YEAR": 180,
    "MONTH_SIN": 0.5,
    "MONTH_COS": 0.866,
    "SEASON": "Rainy",
    "model": "cnn",
}


def test_endpoint():
    try:
        print(f"Testing endpoint: {API_URL}")
        response = requests.post(API_URL, json=TEST_DATA, timeout=10)

        print("\n=== Response Details ===")
        print(f"Status Code: {response.status_code}")
        print("Headers:")
        pprint(dict(response.headers))

        try:
            json_data = response.json()
            print("\nJSON Response:")
            pprint(json_data)
        except ValueError:
            print("\nRaw Response (not JSON):")
            print(response.text)

        return response.ok

    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_endpoint()
    print("\nTest", "passed" if success else "failed")
