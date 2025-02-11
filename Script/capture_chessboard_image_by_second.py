import cv2, os, time, shutil
# go to https://markhedleyjones.com/projects/calibration-checkerboard-collection for chessboard images 
# Ensure the directory exists
save_dir = 'T2C_PickAndPlace/Data/Image4Cal/Freshly_Captured'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
num = 0

n = 0.25 # time of second interval

start_time = time.time()
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= n:
        save_path = os.path.join(save_dir, 'image' + str(num) + '.png')
        if cv2.imwrite(save_path, img):
            print(f"Image saved at {save_path}")
            num += 1
        else: 
            print(f"Failed to save image at {save_path}")    
        start_time = current_time # to reset the timer
    k = cv2.waitKey(5)
    if k == 27:  # Press 'Esc' key to exit
        break
    cv2.imshow('img', img)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()