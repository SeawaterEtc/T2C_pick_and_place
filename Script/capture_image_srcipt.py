import cv2
import os
import shutil

# Ensure the directory exists
save_dir = 'C:/Users/USER/Learn_Coding/T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_image'

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
cv2.waitKey()
cv2.destroyAllWindows()