import time
import requests
import random
import os

# Configuration
API_URL = "http://localhost:8000"
IMAGE_DIR = "data/raw"  # Ensure this path exists and has images
NUM_REQUESTS = 10

def simulate_traffic():
    print(f"Starting traffic simulation to {API_URL}...")
    
    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory {IMAGE_DIR} not found. Retrieving a sample image online.")
        # Download a sample image for testing if local data is missing
        sample_image_url = "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
        img_data = requests.get(sample_image_url).content
        with open("sample_dog.jpg", "wb") as f:
            f.write(img_data)
        image_files = ["sample_dog.jpg"]
    else:
        image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
             print(f"No images found in {IMAGE_DIR}. Downloading sample.")
             sample_image_url = "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
             img_data = requests.get(sample_image_url).content
             with open("sample_dog.jpg", "wb") as f:
                 f.write(img_data)
             image_files = ["sample_dog.jpg"]

    for i in range(NUM_REQUESTS):
        img_path = random.choice(image_files)
        print(f"Request {i+1}/{NUM_REQUESTS}: Sending {img_path}")
        
        try:
            with open(img_path, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Response: {result}")
                
                # Simulate feedback
                # Randomly agree or disagree with prediction for simulation purposes
                actual_label = result["label"] if random.random() > 0.2 else ("Cat" if result["label"] == "Dog" else "Dog")
                
                feedback_data = {
                    "image_id": os.path.basename(img_path),
                    "actual_label": actual_label,
                    "predicted_label": result["label"]
                }
                requests.post(f"{API_URL}/feedback", params=feedback_data)
                print(f"  Feedback sent: {feedback_data}")
            else:
                print(f"  Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"  Exception: {e}")
        
        time.sleep(1)

    print("Simulation complete.")

if __name__ == "__main__":
    simulate_traffic()
