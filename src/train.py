import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from src.data_loader import load_data

# Define model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

def build_model():
    # Load the pre-trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the new model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    # Preprocessing for MobileNetV2 ([-1, 1])
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
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
