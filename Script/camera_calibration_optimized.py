import numpy as np
import cv2 as cv
import glob
import pickle
import concurrent.futures

def get_user_input():
    chessboard_rows = int(input("Rows vertices (input number of black and white boxes in a row minus 1): "))
    chessboard_cols = int(input("Columns vertices (input number of black and white boxes in a column minus 1): "))
    square_size = float(input("Size of a black box mm: "))
    return (chessboard_cols, chessboard_rows), square_size

def process_image(image_path, chessboardSize, criteria):
    """ Processes a single image: detects and refines chessboard corners. """
    img = cv.imread(image_path)
    if img is None:
        return None  # Skip if image is invalid
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if not ret:
        return None  # Skip if no corners found
    
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners2

def main():
    
    
    
    chessboardSize, square_size = get_user_input()
    frameSize = (640, 480)
    
    # High-precision termination criteria for subpixel accuracy
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

    # Generate 3D object points
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by real-world size

    objpoints = []
    imgpoints = []
    
    images = glob.glob('T2C_PickAndPlace/Data/Image4Cal/Freshly_Captured/*.png')  # Use all images

    # Parallelize image processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda img: process_image(img, chessboardSize, criteria), images))

    for result in results:
        if result is not None:
            objpoints.append(objp)
            imgpoints.append(result)

    if not objpoints or not imgpoints:
        print("No valid chessboard images found. Exiting.")
        return

    print(f"Using {len(objpoints)} images for calibration...")

    # Compute an initial camera matrix for better optimization
    cameraMatrix = cv.initCameraMatrix2D(objpoints, imgpoints, frameSize)
    
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frameSize, cameraMatrix, None,
        flags=cv.CALIB_USE_INTRINSIC_GUESS  
    )

    # Save calibration results
    with open("T2C_PickAndPlace/Data/Calibration_result_test1/calibration.pkl", "wb") as f:
        pickle.dump((cameraMatrix, dist), f)

    with open("T2C_PickAndPlace/Data/Calibration_result_test1/cameraMatrix.pkl", "wb") as f:
        pickle.dump(cameraMatrix, f)

    with open("T2C_PickAndPlace/Data/Calibration_result_test1/dist.pkl", "wb") as f:
        pickle.dump(dist, f)

    # Compute reprojection error for accuracy assessment
    mean_error = sum(cv.norm(imgpoints[i], cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)[0], cv.NORM_L2) / len(imgpoints[i])
                     for i in range(len(objpoints))) / len(objpoints)

    print(f"\nTotal reprojection error: {mean_error:.6f}")

if __name__ == "__main__":
    main()
