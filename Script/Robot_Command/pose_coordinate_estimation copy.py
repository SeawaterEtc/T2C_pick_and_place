import numpy as np
import cv2
import pickle
from cv2 import aruco

# Load the intrinsic camera matrix from cameraMatrix.pkl
with open('T2C_PickAndPlace/Data/Calibration_result/cameraMatrix.pkl', 'rb') as f:
    intrinsic_camera = pickle.load(f)

# Load the distortion coefficients from dist.pkl
with open('T2C_PickAndPlace/Data/Calibration_result/dist.pkl', 'rb') as f:
    distortion = pickle.load(f)

# Dictionary of ArUco markers
ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    # Add other ArUco types if needed
}

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, marker_size):
    """
    Detect ArUco markers in the frame and estimate their pose in real-world coordinates.

    Args:
        frame: Image frame from the camera.
        aruco_dict_type: Type of ArUco marker dictionary.
        matrix_coefficients: Intrinsic camera matrix.
        distortion_coefficients: Distortion coefficients for the camera.
        marker_size: Size of the marker in meters.

    Returns:
        frame: The frame with drawn markers and their axes.
        real_world_points: Real-world coordinates of the markers.
        center_x: x-coordinate of the frame center.
        center_y: y-coordinate of the frame center.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()

    corners, ids, rejected_img_points = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    
    real_world_points = []  # Store real-world coordinates of the markers
    
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of the marker
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
                corners[i], marker_size, matrix_coefficients, distortion_coefficients
            )
            
            # Draw detected markers and their axes
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            # Real-world coordinates from the translation vector (tvec)
            real_x = tvec[0][0][0] * 1000  # Convert to millimeters
            real_y = tvec[0][0][1] * 1000  # Convert to millimeters
            real_z = tvec[0][0][2] * 1000  # Convert to millimeters
            
            # Append real-world coordinates
            real_world_points.append((real_x, real_y, real_z))

            # Display real-world coordinates on the frame
            text = f'ID: {ids[i][0]}, X: {real_x:.2f} mm, Y: {real_y:.2f} mm, Z: {real_z:.2f} mm'
            cv2.putText(frame, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, real_world_points, center_x, center_y


def draw_cross_lines(frame, center_x, center_y):
    """
    Draw cross lines (vertical, horizontal, and diagonals) at the center of the frame.

    Args:
        frame: The image frame.
        center_x: x-coordinate of the frame center.
        center_y: y-coordinate of the frame center.
    """
    height, width = frame.shape[:2]
    
    # Define color (blue) and thickness
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 1  # Line thickness
    
    # Draw vertical and horizontal lines
    cv2.line(frame, (center_x, 0), (center_x, height), color, thickness)
    cv2.line(frame, (0, center_y), (width, center_y), color, thickness)

    # Draw diagonals
    cv2.line(frame, (0, 0), (width, height), color, thickness)
    cv2.line(frame, (width, 0), (0, height), color, thickness)


def main():
    """
    Main function to capture video feed from the camera, detect ArUco markers,
    estimate their pose, and display real-world coordinates along with cross lines.
    """
    aruco_type = "DICT_4X4_100"  # Type of ArUco marker
    marker_size = 0.055  # Marker size in meters
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Initialize camera
    print("Camera is now on...")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 0 to disable autofocus

    try:
        while cap.isOpened():
            ret, img = cap.read()  # Capture frame
            if not ret:
                print("Failed to grab frame")
                break

            # Estimate pose and get marker center
            output, real_world_points, center_x, center_y = pose_estimation(
                img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, marker_size
            )
            
            # Draw cross lines at the center of the frame
            draw_cross_lines(output, center_x, center_y)
            
            cv2.imshow('Estimated Pose with Cross Lines', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key to process frame
                print("Frame captured and processing...")
            if key == ord('q'):  # Quit loop
                break

    finally:
        cap.release()  # Release camera
        cv2.destroyAllWindows()  # Close all windows
  
if __name__ == "__main__":
    main()