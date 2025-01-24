# T2C_PickAndPlace_Robot_Arm_ABB_CRB_15000

This repository contain the code to command ABB_CRB_15000 robot arm to pick up objects by using AI model. The current AI model can only detect PET-plastic-bottles, Aluminuim-cans and Snack-packets (It will be able to detect more objects in the future). 
## Showcase

| Auto Pick and Place | Manual Pick and Place |
|---------------------|-----------------------|
| ![Auto Pick and Place](T2C_PickAndPlace/Data/gif/AutoPickAndPlace.gif) | ![Manual Pick](path/to/manual_pick.gif) |


## Set up and run
By cloning this repository and having python 3.11 in your system. You can cd to the directory install all the requrments using: 
```bash
pip install -r requirements.txt
```

### RUNME_GUI

You can extract the Pack and Go File of Robot Studio from the data folder that contain .rspag file. It is a robot studio station for testing the code.

Before running the simulation or real station, please ensure that the port are set up correctly. The RUNME_GUI.py line 43 need to adjust accordingly: 

```python
    def connect_to_robot(self):
        # connect to simulated controller (Robot Studio simulation)
        HOST = '127.0.0.1'
        PORT = 55000  
        
        # # connect to real controller (real robot, change this address to the robot controller IP address)
        # HOST = '192.168.125.1'
        # PORT = 1025 
```
