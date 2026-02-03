import requests
import sys
import time

API_URL = "http://localhost:8000"

def smoke_test():
    print("Starting Smoke Tests...")
    
    # 1. Health Check
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health Check Passed")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Health Check Failed: Connection Error {e}")
        sys.exit(1)

    # 2. Prediction Check
    try:
        # Use a dummy image or existing mock image
        # We need a file to send
        files = {'file': ('test.jpg', open('data/raw/cats/cat_0.jpg', 'rb'), 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "label" in data and "score" in data:
                print(f"✅ Prediction Check Passed: {data}")
            else:
                print(f"❌ Prediction Check Failed: Invalid Response {data}")
                sys.exit(1)
        else:
            print(f"❌ Prediction Check Failed: {response.status_code}")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ Smoke Test Failed: Test image not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Prediction Check Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Optional: Wait for service to be ready
    # time.sleep(5) 
    smoke_test()
