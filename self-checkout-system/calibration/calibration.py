# calibration.py
import numpy as np
import cv2 as cv
import glob
import pickle
import os

# Constants
CHESSBOARD_SIZE = (9, 6)
FRAME_SIZE = (640, 480)
SQUARE_SIZE_MM = 26

def calibrate_camera(images_folder):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []
    images = glob.glob('self-checkout-system/calibration/cali.png')

    for image_path in images:
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, FRAME_SIZE, None, None)
    
    return camera_matrix, dist_coeffs

def main():
    os.makedirs('left_images', exist_ok=True)
    os.makedirs('right_images', exist_ok=True)

    left_matrix, left_dist = calibrate_camera('left_images')
    right_matrix, right_dist = calibrate_camera('right_images')

    pickle.dump((left_matrix, left_dist), open("left_calibration.pkl", "wb"))
    pickle.dump((right_matrix, right_dist), open("right_calibration.pkl", "wb"))

    print("Calibration complete. Results saved to left_calibration.pkl and right_calibration.pkl")

if __name__ == '__main__':
    main()