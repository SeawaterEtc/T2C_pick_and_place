import cv2
import os
import shutil
import platform

# go to https://markhedleyjones.com/projects/calibration-checkerboard-collection for chessboard images 
# Dynamically locate project root
def find_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = current_dir
    while root_dir and not os.path.exists(os.path.join(root_dir, 'gui_main.py')):
        parent = os.path.dirname(root_dir)
        if parent == root_dir:
            break
        root_dir = parent
    return root_dir

ROOT_DIR = find_root_dir()
save_dir = os.path.join(ROOT_DIR, 'Data', 'Image4Cal', 'F-sample_capture')
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)


def list_all_cam(max=10):
    all_camera = []
    for i in range(max):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            all_camera.append(i)
            cap.release()
    return all_camera


""" 
    cv2.CAP_DSHOW: Direct show Windows, may not work. Try alternative backends:
    cv2.CAP_MSMF: For Windows - Media Foundation.
    cv2.CAP_V4L: For Linux.
    cv2.CAP_ANY: Automatic selection.
"""


def draw_cross_lines(image, center_x, center_y):
    """
    Draw cross lines at the center of the image.
    :param image: Input image
    :param center_x: X-coordinate of the center
    :param center_y: Y-coordinate of the center
    """
    # Draw vertical and horizontal lines
    cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 1)  # y-axis
    cv2.line(image, (0, center_y), (image.shape[1], center_y), (0, 255, 0), 1)  # x-axis

    # Draw diagonal lines
    cv2.line(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 1)  # top-left to bottom-right
    cv2.line(image, (0, image.shape[0]), (image.shape[1], 0), (0, 255, 0), 1)  # bottom-left to top-right

def main():
    
    camera = list_all_cam()
    if not camera:
        print("No camera found.")
        return
    
    print("Available cameras: ", camera)

    # Prioritize CAMERA_INDEX from environment, fallback to first found camera, then to 0
    env_camera = os.getenv("CAMERA_INDEX")
    if env_camera is not None:
        camera_idx = int(env_camera)
    elif camera:
        camera_idx = camera[0]
    else:
        camera_idx = 0

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_idx)
    num = 0

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Error: Could not load image.")
            break

        # Get image dimensions and calculate the center
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Draw cross lines at the center of the image
        draw_cross_lines(img, center_x, center_y)

        k = cv2.waitKey(5)
        
        
        if k == 27:  # Press 'Esc' key or 'q' key to exit
            break


        elif k == ord('s'):  # Press 's' key to save the image
            save_path = os.path.join(save_dir, 's-image' + str(num) + '.png')
            if cv2.imwrite(save_path, img):
                print(f"Image saved at {save_path}")
                num += 1
            else:
                print(f"Failed to save image at {save_path}")

        cv2.imshow('img', img)

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()