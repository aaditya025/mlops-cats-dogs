import io
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import time
from starlette.middleware.base import BaseHTTPMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from prometheus_fastapi_instrumentator import Instrumentator

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Define a custom Dense layer that ignores 'quantization_config'
    @tf.keras.utils.register_keras_serializable(package='Custom', name='Dense')
    class PatchedDense(tf.keras.layers.Dense):
        def __init__(self, quantization_config=None, **kwargs):
            super().__init__(**kwargs)

    # Load model on startup
    global model
    model_path = "models/model.h5"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'Dense': PatchedDense})
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Fallback or re-raise if critical
            raise e
    else:
        print("Model file not found! Predictions will fail.")
    yield
    # Clean up (if needed)

app = FastAPI(title="Cats vs Dogs Inference Service", lifespan=lifespan)

# Add CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s")
        return response

# Add Logging Middleware
app.add_middleware(LoggingMiddleware)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health_check():
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

def preprocess_image(image_bytes: bytes):
    """Preprocess image bytes to match training input (224x224, normalized)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    # Rescaling (1./255) is already part of the model layers in train.py!
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        
        prediction = model.predict(processed_img)
        score = float(prediction[0][0])
        
        # Binary classification: <0.5 Cat, >=0.5 Dog (assuming 0=Cat, 1=Dog based on alphabetical)
        # tf.keras.utils.image_dataset_from_directory sorts alphabetically by default
        label = "Dog" if score >= 0.5 else "Cat"
        confidence = score if score >= 0.5 else 1 - score
        
        return {
            "label": label,
            "score": score,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(image_id: str, actual_label: str, predicted_label: str):
    logger.info(f"FEEDBACK: image_id={image_id}, actual={actual_label}, predicted={predicted_label}")
    return {"status": "received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
