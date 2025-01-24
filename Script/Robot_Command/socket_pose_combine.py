import numpy as np
import cv2
import pickle
from Socket_ABB import *
from pose_coordinate_estimation import pose_estimation,draw_cross_lines, intrinsic_camera,distortion,ARUCO_DICT 


def combined_aruco_socket():
    aruco_type = "DICT_4X4_100"
    marker_size = 0.055  # Size of the marker in meters
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("Camera on _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    HOST = '127.0.0.1'  # Replace with your robot controller's IP address
    PORT = 55000  # The same port as used in the RAPID code

    # Connect to the robot controller
    robot_socket = connect_to_robot(HOST, PORT)
    if robot_socket is None:
        print("Failed to connect to the robot controller")
        return

    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Estimate pose and get marker center
            output, real_world_points, center_x, center_y= pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, marker_size)
            
            # Draw the 4 cross lines at the center of the frame
            draw_cross_lines(output, center_x, center_y)
            
            cv2.imshow('Estimated Pose with Cross Lines', output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space key press
                print("Frame captured and processing...")

                if real_world_points:
                    # Sending pixel offsets

                    print(real_world_points)
                    x,y,z = real_world_points[0]
                      
                    data = f"[{round(x, 2)}, {round(y, 2)}, 0.00]"  # Assume z_world offset is 0
                    
    
                    send_message(robot_socket, data)
                    print(f"Sent: {data}")
                    print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n")

            if key == ord('q'):  # Exit loop
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot_socket.close()
        print("Camera and connection closed")

# Call the combined function to run the process
combined_aruco_socket()
