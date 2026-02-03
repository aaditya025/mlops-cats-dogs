import pytest
import shutil
from src.data_loader import load_data, generate_mock_data
import os

def test_load_data():
    """Test if data loader returns a dataset."""
    # Ensure data exists
    if not os.path.exists("data/raw"):
        generate_mock_data(10)
        
    dataset = load_data("data/raw")
    assert dataset is not None, "Dataset should not be None"
    
    # Check if we can get a batch
    for images, labels in dataset.take(1):
        assert images.shape[1:] == (224, 224, 3), "Image shape mismatch"
        assert labels.shape[0] > 0, "Batch size should be > 0"
