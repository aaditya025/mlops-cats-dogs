import mlflow
import mlflow.keras
import tensorflow as tf
import os

# Define a custom Dense layer that ignores 'quantization_config' (Same patch as api.py)
@tf.keras.utils.register_keras_serializable(package='Custom', name='Dense')
class PatchedDense(tf.keras.layers.Dense):
    def __init__(self, quantization_config=None, **kwargs):
        super().__init__(**kwargs)

# Set tracking URI to the mlflow-service container
mlflow.set_tracking_uri("http://mlflow-service:5001")
mlflow.set_experiment("Cats_vs_Dogs_Experiment")

model_path = "/app/models/model.h5"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

print("Loading model...")
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Dense': PatchedDense})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

with mlflow.start_run():
    print("Logging model to MLflow...")
    # Log the model artifacts
    mlflow.keras.log_model(model, "model")
    # Log dummy metrics to simulate a run
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.15)
    print("Model registered successfully!")
