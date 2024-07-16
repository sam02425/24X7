# src/object_detector.py
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, config_path, weights_path, coco_names_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = []
        with open(coco_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return [(boxes[i[0]], confidences[i[0]], self.classes[class_ids[i[0]]]) for i in indices]