import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from src.data_loader import load_data

# Define model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001

def build_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    mlflow.set_experiment("cats_vs_dogs_baseline")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        })
        
        # Load Data
        dataset = load_data()
        
        # Build Model
        model = build_model()
        
        # Train
        history = model.fit(dataset, epochs=EPOCHS)
        
        # Log Metrics
        # logging last epoch results
        acc = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("loss", loss)
        
        # Save Model
        os.makedirs("models", exist_ok=True)
        model.save("models/model.h5")
        mlflow.log_artifact("models/model.h5")
        
        print(f"Training Complete. Accuracy: {acc}, Loss: {loss}")

if __name__ == "__main__":
    train()
