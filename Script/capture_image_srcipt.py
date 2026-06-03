import cv2
import os
import shutil

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
save_dir = os.path.join(ROOT_DIR, 'Data', 'RobotArmObjectCoordinate', 'tmp')

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
num = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    save_path = os.path.join(save_dir, 's-image' + str(num) + '.png')
    if cv2.imwrite(save_path, img):
        print(f"Image saved at {save_path}")
        num += 1
    else:
        print(f"Failed to save image at {save_path}")
    cv2.imshow('img', img)
    break
# Release and destroy all windows before termination
cap.release()
cv2.waitKey(1000)
cv2.destroyAllWindows()