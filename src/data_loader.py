import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def generate_mock_data(num_samples=50):
    """Generates mock data for testing the pipeline."""
    os.makedirs("data/raw/cats", exist_ok=True)
    os.makedirs("data/raw/dogs", exist_ok=True)
    
    for i in range(num_samples):
        # Generate random noise image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # Save half as cats, half as dogs
        if i % 2 == 0:
            img.save(f"data/raw/cats/cat_{i}.jpg")
        else:
            img.save(f"data/raw/dogs/dog_{i}.jpg")
    print(f"Generated {num_samples} mock images in data/raw/")

def load_data(data_dir="data/raw"):
    """
    Loads images from data_dir, resizes, and returns train/val/test splits.
    Assumes structure: 
    data_dir/
      cats/
      dogs/
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Data directory empty or missing. Generating mock data...")
        generate_mock_data()
        
    # Use keras utility for loading dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    # Simple split (not optimal for tf.data.Dataset but works for baseline)
    # For a real pipeline, we might serialize processed numpy arrays to disk
    # allowing for easier splitting and DVC tracking of split files.
    
    # For now, let's just return the dataset as is for training
    # And maybe iterate to create X, y if we want sklearn splitting, 
    # but for CNNs, tf.dataset is better.
    
    return dataset

if __name__ == "__main__":
    generate_mock_data()
