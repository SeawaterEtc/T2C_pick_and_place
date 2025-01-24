import numpy as np
import cv2 as cv
import glob
import pickle
import os

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (6,4)  # verticies
frameSize = (640,480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 40  # squares size
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('T2C_PickAndPlace/Data/Image4Cal/Freshly_Captured/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()


############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist, rvecs, tvecs), open("T2C_PickAndPlace/Data/calReWithRvecTvec/calibration.pkl", "wb"))
pickle.dump(cameraMatrix, open("T2C_PickAndPlace/Data/calReWithRvecTvec/cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("T2C_PickAndPlace/Data/calReWithRvecTvec/dist.pkl", "wb"))
pickle.dump(rvecs, open("T2C_PickAndPlace/Data/calReWithRvecTvec/rvecs.pkl", "wb"))
pickle.dump(tvecs, open("T2C_PickAndPlace/Data/calReWithRvecTvec/tvecs.pkl", "wb"))

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

Please check undistorted_images folder to see output of the calibration: 

The undistortion process applies the calibration parameters to a specific image.
When you test the undistortion, you are checking:
How well the distortion is corrected for that image.
Whether the field of view and cropping are as expected.
If the undistortion works correctly for one image, it’s safe to assume it will work for others (provided they were taken with the same camera).\n
"""
print(reprojection_meaning)
