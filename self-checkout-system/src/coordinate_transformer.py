# src/coordinate_transformer.py
import cv2
import numpy as np
import pickle

class CoordinateTransformer:
    def __init__(self, left_calib_file, right_calib_file, stereo_calib_file):
        with open(left_calib_file, 'rb') as f:
            self.left_matrix, self.left_dist = pickle.load(f)
        with open(right_calib_file, 'rb') as f:
            self.right_matrix, self.right_dist = pickle.load(f)
        with open(stereo_calib_file, 'rb') as f:
            self.R, self.T = pickle.load(f)

    def transform_coordinates(self, points, camera_matrix, dist_coeffs, offset):
        undistorted_points = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
        homogeneous_points = cv2.convertPointsToHomogeneous(undistorted_points)
        
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self.R
        transformation_matrix[:3, 3] = self.T.ravel() + offset
        
        world_points = []
        for point in homogeneous_points:
            world_point = np.dot(transformation_matrix, point.T).T
            world_points.append(world_point[:3])
        
        return np.array(world_points)

    def transform_left(self, objects):
        points = np.array([[obj[0], obj[1]] for obj in objects], dtype=np.float32).reshape(-1, 1, 2)
        return self.transform_coordinates(points, self.left_matrix, self.left_dist, np.array([-120, 0, -60]))

    def transform_right(self, objects):
        points = np.array([[obj[0], obj[1]] for obj in objects], dtype=np.float32).reshape(-1, 1, 2)
        return self.transform_coordinates(points, self.right_matrix, self.right_dist, np.array([120, 0, -60]))