# tests/test_self_checkout.py
import pytest
from src.main import app, detector, transformer
import cv2
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}

def test_checkout(client, mocker):
    # Mock camera capture
    mocker.patch('cv2.VideoCapture')
    mocker.patch('cv2.VideoCapture.read', return_value=(True, np.zeros((480, 640, 3), dtype=np.uint8)))
    
    # Mock object detection
    mocker.patch.object(detector, 'detect', return_value=[([100, 100, 50, 50], 0.9, 'apple')])
    
    # Mock coordinate transformation
    mocker.patch.object(transformer, 'transform_left', return_value=np.array([[0, 0, 0]]))
    mocker.patch.object(transformer, 'transform_right', return_value=np.array([[1, 1, 1]]))

    response = client.post('/checkout')
    assert response.status_code == 200
    assert "Checkout successful" in response.json["message"]
    assert "total" in response.json
    assert "items" in response.json

def test_checkout_camera_failure(client, mocker):
    mocker.patch('cv2.VideoCapture')
    mocker.patch('cv2.VideoCapture.read', return_value=(False, None))

    response = client.post('/checkout')
    assert response.status_code == 500
    assert "Failed to capture images" in response.json["error"]

# Add more tests for edge cases and error handling