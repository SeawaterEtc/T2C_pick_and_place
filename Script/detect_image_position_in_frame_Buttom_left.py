import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load model
model = YOLO("T2C_PickAndPlace/Data/AImodel/100.pt")  # segmentation model
names = model.model.names

def adjust_brightness(image, brightness=1.0):
    """
    Adjust the brightness of the image.
    :param image: Input image
    :param brightness: Brightness factor (1.0 means no change, >1.0 means increase, <1.0 means decrease)
    :return: Brightness adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_box(image, rect, cls, color, thickness=2):
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Use np.intp instead of np.int0
    cv2.drawContours(image, [box], 0, color, thickness)
    center = (int(rect[0][0]), int(rect[0][1]))
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    relative_center = (center[0], h - center[1])  # Adjust for bottom-left origin
    cv2.putText(image, f"({relative_center[0]}, {relative_center[1]})", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, names[int(cls)], (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Load the image
image_path = 'C:/Users/USER/Learn_Coding/T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Sample/s-image3.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Adjust brightness
brightness_factor = 0.79  # Increase brightness by 20%
image = adjust_brightness(image, brightness_factor)


# Get image dimensions and calculate the center
h, w = image.shape[:2]

# Draw the x and y axes at the bottom-left of the image
cv2.line(image, (0, h), (w, h), (0, 255, 0), 2)  # x-axis
cv2.line(image, (0, 0), (0, h), (0, 255, 0), 2)  # y-axis

# Perform object detection
results = model.predict(image, conf=0.5)  # Adjust thresholds

if results[0].masks is not None:
    clss = results[0].boxes.cls.cpu().tolist()
    masks = results[0].masks.xy
    for mask, cls in zip(masks, clss):
        # Calculate the oriented bounding box
        rect = cv2.minAreaRect(np.array(mask))
        box_center = (int(rect[0][0]), int(rect[0][1]))
        
        # Calculate position relative to the bottom-left corner of the frame
        relative_position = (box_center[0], h - box_center[1])
        
        # Print the detected object class and its position
        class_name = names[int(cls)]
        print(f"Detected object class: {class_name}, Position relative to bottom-left: {relative_position}")

        # Optionally, draw the bounding box on the image
        draw_box(image, rect, cls, color=colors(int(cls), True))

# Display the image with annotations
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()