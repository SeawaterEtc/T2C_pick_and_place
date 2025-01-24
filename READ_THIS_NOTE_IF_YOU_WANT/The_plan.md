What are the stuff that the user can do? 

This system, user can sign up and log in, after log in user can do a few things
1. Demo the object detection
2. Do camera calibration (With the current system, it is the best way to find objects position and orientation using a single camera) -- there are different methods in finding the position. I need to experiment which one is the best. 
3. Command GOFA CRB 15000 (robot arm) to pick up objects

The command part requires me to understand the socket communication, rapid programming language of robot arm. The logic is sending the detected object position to the robot controller and move it to the position and perform pick and place. 
    