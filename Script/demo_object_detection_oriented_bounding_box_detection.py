import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load model
model = YOLO("T2C_PickAndPlace/Data/AImodel/1-15-50.pt")  # segmentation model
names = model.model.names

def draw_oriented_bounding_box(image, box, cls, color=(0, 255, 0), thickness=2):
    # Extract the coordinates of the box
    x, y, w, h, angle = box
    center = (int(x), int(y))
    size = (int(w), int(h))
    rect = cv2.boxPoints((center, size, angle))
    rect = np.int0(rect)
    cv2.drawContours(image, [rect], 0, color, thickness)
    
    # Draw the center coordinates and class label inside the bounding box
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    cv2.putText(image, f"({center[0]}, {center[1]})", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, names[int(cls)], (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce frame size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('T2C_PickAndPlace/Script/instance-segmentation.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, conf=0.75, iou=0.5)  # Adjust thresholds

    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            # Calculate the oriented bounding box
            rect = cv2.minAreaRect(np.array(mask))
            draw_oriented_bounding_box(im0, (*rect[0], *rect[1], rect[2]), cls,
                                       color=colors(int(cls), True))

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    # click on any button to break
    if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('x')]:  # 27 is the ASCII code for the ESC key
        break

    # Break the loop when the window is closed
    if cv2.getWindowProperty("instance-segmentation", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
