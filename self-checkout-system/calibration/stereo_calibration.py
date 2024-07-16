# stereo_calibration.py
import cv2
import numpy as np
import pickle
import glob

CHESSBOARD_SIZE = (9, 6)
FRAME_SIZE = (640, 480)

def stereo_calibrate():
    with open('left_calibration.pkl', 'rb') as file:
        left_matrix, left_dist = pickle.load(file)
    with open('right_calibration.pkl', 'rb') as file:
        right_matrix, right_dist = pickle.load(file)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    objpoints = []
    left_imgpoints = []
    right_imgpoints = []

    left_images = sorted(glob.glob('left_images/*.png'))
    right_images = sorted(glob.glob('right_images/*.png'))

    for left_img, right_img in zip(left_images, right_images):
        left = cv2.imread(left_img)
        right = cv2.imread(right_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None)

        if ret_left and ret_right:
            objpoints.append(objp)
            left_imgpoints.append(corners_left)
            right_imgpoints.append(corners_right)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        left_matrix, left_dist, right_matrix, right_dist,
        FRAME_SIZE, criteria=criteria, flags=flags)

    pickle.dump((R, T), open("stereo_calibration.pkl", "wb"))
    print("Stereo calibration complete. Results saved to stereo_calibration.pkl")

if __name__ == '__main__':
    stereo_calibrate()