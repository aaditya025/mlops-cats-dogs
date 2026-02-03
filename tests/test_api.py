from fastapi.testclient import TestClient
from src.api import app
import os
from PIL import Image
import numpy as np
import io
import pytest

client = TestClient(app)

def test_read_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    # Only test if model exists, otherwise mock
    if os.path.exists("models/model.h5"):
        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "score" in data
            assert data["label"] in ["Cat", "Dog"]
    else:
        pytest.skip("Model not found, skipping prediction test")
