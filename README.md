# T2C_PickAndPlace_Robot_Arm_ABB_CRB_15000

This repository contain the code to command ABB_CRB_15000 robot arm to pick up objects by using AI model. The current AI model can only detect PET-plastic-bottles, Aluminuim-cans and Snack-packets (It will be able to detect more objects in the future). 
## Showcase

| Auto Pick and Place | Manual Pick and Place |
|---------------------|-----------------------|
| ![Auto Pick and Place](Data/gif/AutoPickAndPlace.gif) | ![Manual Pick](path/to/manual_pick.gif) |


## Set up and run

### Step 1: Prerequisites

By cloning/downloading this repository and having python 3.11 in your system. You can cd to the directory, for example,  (C:\Users\USER\Learn_Coding\T2C_Pick_And_Place> ) install all the requrments using: 
```bash
pip install -r requirements.txt
```

### Step 2: Please notice that the code run using python script on top of script. If you are not on top of the directory, you will need to adjust the path accordingly.

Here is a few example of where you are. 

if 
```bash
C:\Users\USER\Learn_Coding\T2C_PickAndPlace>
```
Then you are inside the directory, code can't be run, because it won't be able to find other scripts. 

if 
```bash
C:\Users\USER>Learn_Coding>
```
Then you are not inside the directory, you can run the code.

### Step 3: run gui_main.py and Robot Studio Simulation

You need a Robot Studio to run the simulation. You can download the Robot Studio from the ABB website, and use the pack and go file to extract the station.

You can find the the Pack and Go File of Robot Studio from the data folder that contain .rspag file. It is a robot studio station for testing the code.

Then you need to start the simulation in Robot Studio.

And finally, you can run the RUNME_GUI.py to start the GUI. 

#### How to test the code

* ensure that you are on top of the directory

=> run the gui_main.py 
=> start robot studio simulation 
=> click on command the robot arm 
=> click on connect with simulation 
=> click on stationary object mode 
=> click on either manual pick or auto pick and place (it will detect the object in the input image, you need to close the detection widget to continue the process)

| Example How to run the code |
|---------------------|
| ![Auto Pick and Place](Data/gif/how2run.gif) |
