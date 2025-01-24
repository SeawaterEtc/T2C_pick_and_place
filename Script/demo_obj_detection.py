import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("T2C_PickAndPlace/Data/AImodel/74.pt")
names = model.model.names

# Adjust frame size variables
DESIRED_FRAME_WIDTH = 1000
DESIRED_FRAME_HEIGHT = 800

# Frame rate variable
FRAME_RATE = 30  # Desired frames per second (fps)

def annotate_object(frame, box, cls, frame_center):
    x, y, w, h = box  # Unpack four values

    # Calculate the center of the detected object
    obj_center_x = x
    obj_center_y = y

    # Calculate relative position
    relative_x = obj_center_x - frame_center[0]
    relative_y = obj_center_y - frame_center[1]

    # Convert class label to string
    cls_text = names[int(cls)]

    # Display class name above the object
    cv2.putText(frame, cls_text, (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display relative position below the object
    position_text = f"Pos_In_frame: ({relative_x:.1f}, {relative_y:.1f})"
    cv2.putText(frame, position_text, (int(x - w / 2), int(y + h / 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def draw_oriented_bounding_box(image, box, cls, frame_center, color=(0, 255, 0), thickness=2):
    x, y, w, h, angle = box
    center = (int(x), int(y))
    size = (int(w), int(h))
    rect = cv2.boxPoints((center, size, angle))
    rect = np.int0(rect)
    
    # Draw the center point
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    
    # Calculate relative position
    relative_x = center[0] - frame_center[0]
    relative_y = center[1] - frame_center[1]
    
    # Convert class label to string
    cls_text = names[int(cls)]
    
    # Display class name above the center point
    cv2.putText(image, cls_text, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display relative position below the center point
    position_text = f"Pos: ({relative_x}, {relative_y})"
    cv2.putText(image, position_text, (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_FRAME_HEIGHT)

    # Calculate frame delay based on desired frame rate
    frame_delay = int(2000 / FRAME_RATE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame_center = (width // 2, height // 2)
        
        # results = model.predict(frame)
        
        # # Define confidence and NMS thresholds and show result
        # conf_threshold = 0.5  #Only detections with a confidence score of % or higher will be kept.  
        # nms_threshold = 0 # NMS is a technique used to eliminate redundant bounding boxes for the same object. The threshold of 0.4 means that if the Intersection over Union (IoU) between two boxes is greater than 0.4, the box with the lower confidence score will be suppressed.
        # results = model.predict(frame, conf=conf_threshold, iou=nms_threshold)

        # Define confidence and NMS thresholds and show result
        conf_threshold = 0.7  #Only detections with a confidence score of % or higher will be kept.
        results = model.predict(frame, conf=conf_threshold)

        if results[0].masks is not None:
            for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
                draw_oriented_bounding_box(frame, (*box, 0), cls, frame_center)

        cv2.imshow("Object Detection", frame)

        # Add frame delay to control frame rate
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# # HELP # combine draw_oriented_bounding_box and show_object_position_relative_to_center function into one. The goal is to detect the object, draw box around it but make the box invisible, show the class name on top and the position at the bottom. 

# import numpy as np
# import cv2
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("T2C_PickAndPlace/Data/74.pt")
# names = model.model.names

# # Adjust frame size variables
# DESIRED_FRAME_WIDTH = 640
# DESIRED_FRAME_HEIGHT = 480

# # Frame rate variable
# FRAME_RATE = 30  # Desired frames per second (fps)

# def draw_oriented_bounding_box(frame, box, cls):
#     x, y, w, h = box  # Unpack four values

#     # Calculate the rectangle points
#     top_left = (int(x - w / 2), int(y - h / 2))
#     bottom_right = (int(x + w / 2), int(y + h / 2))

#     # Draw rectangle on the frame
#     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    
#     # Convert class label to string before displaying
#     cls_text = names[int(cls)]
#     cv2.putText(frame, cls_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# def show_object_position_relative_to_center(frame, box, frame_center):
#     x, y, w, h = box  # Unpack box values
#     obj_center_x = x
#     obj_center_y = y

#     # Calculate relative position
#     relative_x = obj_center_x - frame_center[0]
#     relative_y = obj_center_y - frame_center[1]

#     # Display relative position on the frame
#     position_text = f"Rel Pos: ({relative_x}, {relative_y})"
#     cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# def main():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_FRAME_HEIGHT)

#     # Calculate frame delay based on desired frame rate
#     frame_delay = int(1000 / FRAME_RATE)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         height, width = frame.shape[:2]
#         frame_center = (width // 2, height // 2)

#         results = model.predict(frame)
#         if results[0].masks is not None:
#             for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
#                 draw_oriented_bounding_box(frame, box[:4], cls)
#                 show_object_position_relative_to_center(frame, box, frame_center)

#         cv2.imshow("Object Detection", frame)

#         # Add frame delay to control frame rate
#         if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# # HELP # remove the draw_cross_line function. add the variables to adjust frame desired width and height of the frame then implement it in main. Add the function to show the detected object position relative to the center of the frame (0,0), and implement it in main. Add the option to adjust frame rate to main.

# import numpy as np
# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

# # Load YOLO model
# model = YOLO("T2C_PickAndPlace/Data/74.pt")
# names = model.model.names

# def draw_oriented_bounding_box(frame, box, cls):
#     x, y, w, h = box  # Unpack four values
#     angle = 0         # Default angle to 0 if not provided
    
#     # Calculate the rectangle points (if rotation is required in the future)
#     top_left = (int(x - w / 2), int(y - h / 2))
#     bottom_right = (int(x + w / 2), int(y + h / 2))

#     # Draw rectangle on the frame
#     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    
#     # Convert class label to string before displaying
#     cls_text = names[int(cls)]
#     cv2.putText(frame, cls_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


# def draw_cross_lines(frame, center_x, center_y):
#     height, width = frame.shape[:2]
#     cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
#     cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)

# def main():
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         height, width = frame.shape[:2]
#         center_x, center_y = width // 2, height // 2

#         results = model.predict(frame)
#         boxes = []
#         if results[0].masks is not None:
#             for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
#                 boxes.append((*box, 0))
#                 draw_oriented_bounding_box(frame, box[:4], cls)

#         draw_cross_lines(frame, center_x, center_y)
#         cv2.imshow("Object Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


# # HELP # demo_obj_detection.py script need modifcation

# import numpy as np
# import cv2
# import pickle
# from cv2 import aruco
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
# import time

# # HELP # No need to load calibration data
# # Load camera calibration data
# with open('D:/CamTech_Related/V3_waste_detection_experiment/camera_calibration/cameraMatrix.pkl', 'rb') as f:
#     intrinsic_camera = pickle.load(f)

# with open('D:/CamTech_Related/V3_waste_detection_experiment/camera_calibration/dist.pkl', 'rb') as f:
#     distortion = pickle.load(f)

# # HELP # change Load YOLO model path to T2C_PickAndPlace/Data/100.pt
# model = YOLO("D:/CamTech_Related/V3_waste_detection_experiment/100.pt")
# names = model.model.names

# # HELP # modify to do only draw the bouding box on detection object
# def draw_oriented_bounding_box(image, box, cls, color=(0, 255, 0), thickness=2):
#     x, y, w, h, angle = box
#     center = (int(x), int(y))
#     size = (int(w), int(h))
#     rect = cv2.boxPoints((center, size, angle))
#     rect = np.intp(rect)
#     cv2.drawContours(image, [rect], 0, color, thickness)
    
#     cv2.circle(image, center, 5, (0, 0, 255), -1)
#     cv2.putText(image, f"({center[0]}, {center[1]})", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#     cv2.putText(image, names[int(cls)], (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# # HELP # no need position esitmation, but calculate the position of object relative to the center of the camera frame. The center of the camera from is (0,0) any object detected will be calculate accordingly
# def pose_estimation(frame, boxes, matrix_coefficients, distortion_coefficients):
#     real_world_points = [] 
#     height, width = frame.shape[:2]
#     center_x, center_y = width // 2, height // 2

#     for box in boxes:
#         x, y, w, h, angle = box
#         rect = ((x, y), (w, h), angle)
#         box_points = cv2.boxPoints(rect)
#         box_points = np.intp(box_points)

#         if len(box_points) >= 4:
#             # Convert box points to the correct format
#             box_points = np.array(box_points, dtype=np.float32)

#             # Estimate pose
#             rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
#                 [box_points], 0.055, matrix_coefficients, distortion_coefficients
#             )
            
#             cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

#             real_x = tvec[0][0][0] * 1000  
#             real_y = tvec[0][0][1] * 1000  
#             real_z = tvec[0][0][2] * 1000 
            
#             real_world_points.append((real_x, real_y, real_z))

#             text = f'X: {real_x:.2f} mm, Y: {real_y:.2f} mm, Z: {real_z:.2f} mm'
#             cv2.putText(frame, text, (int(x) + 10, int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Print the estimated position
#             print(f"Estimated Position - X: {real_x:.2f} mm, Y: {real_y:.2f} mm, Z: {real_z:.2f} mm")

#     return frame, real_world_points, center_x, center_y

# def draw_cross_lines(frame, center_x, center_y):
#     height, width = frame.shape[:2]
    
#     color = (255, 0, 0)  
#     thickness = 1  
    
#     cv2.line(frame, (center_x, 0), (center_x, height), color, thickness)
#     cv2.line(frame, (0, center_y), (width, center_y), color, thickness)

#     cv2.line(frame, (0, 0), (width, height), color, thickness)
#     cv2.line(frame, (width, 0), (0, height), color, thickness)

# def main():
#     cap = cv2.VideoCapture(0)
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#     out = cv2.VideoWriter('instance-segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

#     try:
#         while cap.isOpened():
#             ret, im0 = cap.read()
#             if not ret:
#                 print("Video frame is empty or video processing has been successfully completed.")
#                 break
            
#             # Define confidence and NMS thresholds and show result
#             conf_threshold = 0.9  #Only detections with a confidence score of % or higher will be kept.  
#             nms_threshold = 0.4 # NMS is a technique used to eliminate redundant bounding boxes for the same object. The threshold of 0.4 means that if the Intersection over Union (IoU) between two boxes is greater than 0.4, the box with the lower confidence score will be suppressed.
#             results = model.predict(im0, iou=nms_threshold)
#             annotator = Annotator(im0, line_width=2)

#             boxes = []
#             if results[0].masks is not None:
#                 clss = results[0].boxes.cls.cpu().tolist()
#                 confs = results[0].boxes.conf.cpu().tolist()
#                 masks = results[0].masks.xy
#                 for mask, cls, conf in zip(masks, clss, confs):
#                     if conf >= 0.8:
#                         rect = cv2.minAreaRect(np.array(mask))
#                         boxes.append((*rect[0], *rect[1], rect[2]))
#                         draw_oriented_bounding_box(im0, (*rect[0], *rect[1], rect[2]), cls, color=colors(int(cls), True))

#                         # Print the bounding box details
#                         print(f"Bounding Box - Class: {names[int(cls)]}, Center: ({rect[0][0]}, {rect[0][1]}), Size: ({rect[1][0]}, {rect[1][1]}), Angle: {rect[2]}")

#             output, real_world_points, center_x, center_y = pose_estimation(
#                 im0, boxes, intrinsic_camera, distortion
#             )
            
#             draw_cross_lines(output, center_x, center_y)
            
#             out.write(im0)
#             cv2.imshow("instance-segmentation", im0)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             # Wait for 1 second before capturing the next frame
#             time.sleep(0.5)

# # store position of the detected object


#     finally:
#         out.release()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
