import numpy as np
import cv2 as cv
import glob
import pickle
import os

def get_user_input():
    chessboard_rows = int(input("Rows verticis (input number of black and white box a row minus 1): "))
    chessboard_cols = int(input("Columns verticis (input number of black and white box in a column minus 1): "))
    square_size = float(input("Size of a black box mm: "))
    return (chessboard_cols, chessboard_rows), square_size

def main():
    # Get user input for chessboard size and square size
    chessboardSize, size_of_chessboard_squares_mm = get_user_input()
    frameSize = (640, 480)

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objp *= size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob('T2C_PickAndPlace/Data/Image4Cal/Freshly_Captured/*.png')

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(10)

    cv.destroyAllWindows()

    # Calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # Save the camera calibration result for later use
    pickle.dump((cameraMatrix, dist), open("T2C_PickAndPlace/Data/Calibration_result_test/calibration.pkl", "wb"))
    pickle.dump(cameraMatrix, open("T2C_PickAndPlace/Data/Calibration_result_test/cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open("T2C_PickAndPlace/Data/Calibration_result_test/dist.pkl", "wb"))

    # Calculate and display reprojection error
    mean_error = 0

    for i in range(len(objpoints)):
        # Project 3D object points to the 2D image plane
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        
        # Calculate the Euclidean norm error between actual and projected points
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    # Average reprojection error across all images
    mean_error /= len(objpoints)

    print(f"\nTotal reprojection error: {mean_error:.6f}")
    reprojection_meaning = """\nWhat is a Good Reprojection Error?
Ideally, a reprojection error less than 0.5 pixels is considered good.
Errors higher than 1 pixel suggest that the calibration might need refinement, such as:
Using higher-quality images.
Ensuring good coverage of different perspectives of the chessboard.
Increasing the number of calibration images.
\n
"""
    print(reprojection_meaning)

if __name__ == "__main__":
    main()