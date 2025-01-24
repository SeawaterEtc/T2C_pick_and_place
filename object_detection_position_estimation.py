import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import pickle

# Load model
model = YOLO("T2C_PickAndPlace/Data/AImodel/1-15-50-50best.pt")  # segmentation model
names = model.model.names

# Load the intrinsic camera matrix from cameraMatrix.pkl
with open('T2C_PickAndPlace/Data/Calibration_result/cameraMatrix.pkl', 'rb') as f:
    intrinsic_camera = pickle.load(f)

# Load the distortion coefficients from dist.pkl
with open('T2C_PickAndPlace/Data/Calibration_result/dist.pkl', 'rb') as f:
    distortion = pickle.load(f)

# # Load the intrinsic camera matrix from old calibration

# with open('T2C_PickAndPlace/Data/camera_calibration/cameraMatrix.pkl', 'rb') as f:
#     intrinsic_camera = pickle.load(f)

# # Load the distortion coefficients from dist.pkl
# with open('T2C_PickAndPlace/Data/camera_calibration/dist.pkl', 'rb') as f:
#     distortion = pickle.load(f)


def draw_oriented_bounding_box(image, box, cls, color=(0, 255, 0), thickness=2):
    x, y, w, h, angle = box
    center = (int(x), int(y))
    size = (int(w), int(h))
    rect = cv2.boxPoints((center, size, angle))
    rect = np.int0(rect)
    cv2.drawContours(image, [rect], 0, color, thickness)
    
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    cv2.putText(image, f"({center[0]}, {center[1]})", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, names[int(cls)], (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def estimate_position(center, matrix_coefficients, distortion_coefficients, rvec, tvec):
    center = np.array([[center]], dtype=np.float32)
    center_undistorted = cv2.undistortPoints(center, matrix_coefficients, distortion_coefficients)
    world_points, _ = cv2.projectPoints(center_undistorted, rvec, tvec, matrix_coefficients, distortion_coefficients)
    real_x = world_points[0][0][0] * 12000
    real_y = world_points[0][0][1] * 12000
    return real_x, real_y

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('T2C_PickAndPlace/Script/object_detection_position_estimation.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, conf=0.75, iou=0.5)

    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            rect = cv2.minAreaRect(np.array(mask))
            draw_oriented_bounding_box(im0, (*rect[0], *rect[1], rect[2]), cls, color=colors(int(cls), True))
            real_x, real_y = estimate_position(rect[0], intrinsic_camera, distortion)
            cv2.putText(im0, f"Real X: {real_x:.2f} mm, Real Y: {real_y:.2f} mm", (int(rect[0][0]) + 10, int(rect[0][1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # out.write(im0)
    cv2.imshow("object_detection_position_estimation", im0)

    if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('x')]:
        break

    if cv2.getWindowProperty("object_detection_position_estimation", cv2.WND_PROP_VISIBLE) < 1:
        break

# out.release()
cap.release()
cv2.destroyAllWindows()