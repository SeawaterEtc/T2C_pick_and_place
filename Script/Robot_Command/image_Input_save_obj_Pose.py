import cv2
import numpy as np
import pickle
import glob
from ultralytics import YOLO

# Load the intrinsic camera matrix from cameraMatrix.pkl
with open('T2C_PickAndPlace/Data/calReWithRvecTvec/cameraMatrix.pkl', 'rb') as f:
    intrinsic_camera = pickle.load(f)

# Load the distortion coefficients from dist.pkl
with open('T2C_PickAndPlace/Data/calReWithRvecTvec/dist.pkl', 'rb') as f:
    distortion = pickle.load(f)

# Load model
model = YOLO("T2C_PickAndPlace/Data/AImodel/1-15-50-50best.pt")  # segmentation model
names = model.model.names

# Function to adjust the brightness of the image
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

def draw_dot(image, rect, cls):
    center = (int(rect[0][0]), int(rect[0][1]))
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    cv2.putText(image, names[int(cls)], (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def estimate_position(center, matrix_coefficients, distortion_coefficients):
    center = np.array([[center]], dtype=np.float32)
    center_undistorted = cv2.undistortPoints(center, matrix_coefficients, distortion_coefficients)
    real_x = center_undistorted[0][0][0] * 1000
    real_y = center_undistorted[0][0][1] * 1000
    return real_x, real_y

def main():
    image_path = 'T2C_PickAndPlace/Data/RobotArmObjectCoordinate/sample/*.png'
    save_path = 'T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt'

    # Open the file to save positions
    with open(save_path, 'w') as file:
        for img_path in glob.glob(image_path):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not load image {img_path}.")
                continue

            # Adjust brightness
            brightness_factor = 0.85 # Adjust as needed
            image = adjust_brightness(image, brightness_factor)

            # Perform object detection
            results = model.predict(image, conf=0.8)  # Adjust thresholds

            if results[0].masks is not None:
                print(f"Detected {len(results[0].masks.xy)} objects.")
                clss = results[0].boxes.cls.cpu().tolist()
                masks = results[0].masks.xy
                for mask, cls in zip(masks, clss):
                    # Calculate the oriented bounding box
                    rect = cv2.minAreaRect(np.array(mask))
                    
                    # Calculate the inner rectangle (n% smaller)
                    inner_width = rect[1][0] * 0.01
                    inner_height = rect[1][1] * 0.01
                    inner_Rec = ((rect[0][0], rect[0][1]), (inner_width, inner_height), rect[2])
                    
                    box_center = (int(inner_Rec[0][0]), int(inner_Rec[0][1]))

                    # Calculate real-world position
                    real_x, real_y = estimate_position(box_center, intrinsic_camera, distortion)
                    world_coords = (real_x, real_y, 0.00)  # Set z to 0.00

                    # Print and save the real-world position
                    class_name = names[int(cls)]
                    print(f"Detected object class: {class_name}, Real-world position: {world_coords}")
                    file.write(f"{class_name},{world_coords[0]},{world_coords[1]},{world_coords[2]}\n")

                    # Optionally, draw the bounding box on the image
                    draw_dot(image, inner_Rec, cls)

            # Display the image with annotations
            cv2.imshow("Detected Objects", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()