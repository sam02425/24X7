# src/main.py
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
import os
import cv2
import numpy as np
from object_detector import ObjectDetector
from coordinate_transformer import CoordinateTransformer

app = Flask(__name__)
producer = KafkaProducer(bootstrap_servers=os.environ.get('KAFKA_BROKERS', 'localhost:9092'),
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Initialize object detector and coordinate transformer
detector = ObjectDetector('config/yolo.cfg', 'config/yolo.weights', 'config/coco.names')
transformer = CoordinateTransformer('data/left_calibration.pkl', 'data/right_calibration.pkl', 'data/stereo_calibration.pkl')

LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/checkout', methods=['POST'])
def checkout():
    # Capture images from both cameras
    left_camera = cv2.VideoCapture(LEFT_CAMERA_ID)
    right_camera = cv2.VideoCapture(RIGHT_CAMERA_ID)
    
    ret_left, left_frame = left_camera.read()
    ret_right, right_frame = right_camera.read()
    
    left_camera.release()
    right_camera.release()
    
    if not ret_left or not ret_right:
        return jsonify({"error": "Failed to capture images"}), 500

    # Detect objects in both frames
    left_objects = detector.detect(left_frame)
    right_objects = detector.detect(right_frame)

    # Transform coordinates
    left_world_coords = transformer.transform_left(left_objects)
    right_world_coords = transformer.transform_right(right_objects)

    # Combine and process detected items
    all_items = process_detected_items(left_world_coords, right_world_coords)

    # Calculate total
    total = sum(item['price'] * item['quantity'] for item in all_items)

    # Send event to Kafka
    producer.send('shelf_checkout_events', {
        'items': all_items,
        'total': total
    })

    return jsonify({"message": "Checkout successful", "total": total, "items": all_items})

def process_detected_items(left_coords, right_coords):
    # Implement logic to process detected items, match with database, etc.
    # This is a placeholder implementation
    all_coords = left_coords + right_coords
    items = []
    for i, coord in enumerate(all_coords):
        items.append({
            "id": i + 1,
            "name": f"Detected Item {i + 1}",
            "price": 10.0,  # placeholder price
            "quantity": 1,
            "coordinates": coord.tolist()
        })
    return items

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))