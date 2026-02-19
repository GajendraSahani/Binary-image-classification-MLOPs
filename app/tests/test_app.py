import pytest
import numpy as np
import os
from src.preprocess import preprocess_image
from app.main import app
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(app)

def test_preprocess_logic():
    # Creating a temporary test image
    test_img_path = "test_input.jpg"
    dummy_img = Image.new('RGB', (500, 500), color='red')
    dummy_img.save(test_img_path)
    
    processed_img = preprocess_image(test_img_path)
    
    assert processed_img.shape == (224, 224, 3)
    assert processed_img.max() <= 1.0
    assert processed_img.min() >= 0.0
    
    os.remove(test_img_path)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"} [cite: 24]
