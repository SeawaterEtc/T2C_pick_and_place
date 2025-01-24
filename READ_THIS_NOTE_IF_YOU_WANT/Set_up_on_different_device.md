Please install all requirement before running the system. 

# OS: Windows 

### Using pipreqs for Automation of generating requirements

1. **Install pipreqs**:
```bash   
    pip install pipreqs
```    
2. **Generate requirements.txt**:
    go into the path of your project cd /path/to/your/project, after inside the project: 
```bash
     pipreqs
```    














## Setup 1: 
If you prefer to use your global Python environment, you don't need to create a virtual environment. However, you will need to ensure that all the necessary dependencies are included in your project and can be installed offline. Here’s how you can do it:

### Step-by-Step Guide Without Virtual Environment

1. Download Dependencies:
    - Use pip download to download all the dependencies into a directory.
```bash   
    mkdir dependencies
    pip download -r requirements.txt -d dependencies
```    
2. Package the Project:
    - Include the requirements.txt file and the dependencies directory in your project directory.
```bash   
    xcopy dependencies myproject\dependencies /E /H /C /I
```    
3. Zip the Project:
    - Zip the entire project directory, including the dependencies.
```bash   
    powershell Compress-Archive -Path myproject -DestinationPath myproject.zip
```    
### Running the Project on Another PC

1. Unzip the Project:
    - Unzip the project on the target PC.
```bash   
    powershell Expand-Archive -Path myproject.zip -DestinationPath myproject
```    
2. Install Python:
    - Ensure Python is installed on the target PC. You can download it from the [official Python website](https://www.python.org/downloads/).

3. Install Dependencies Offline:
    - Install the dependencies from the local directory.
```bash   
    pip install --no-index --find-links myproject\dependencies -r myproject\requirements.txt
```    
4. Run the Project:
    - Run your project.
```bash   
    python myproject\main.py
```    


## Set up 1: 

You can package your Python project along with its dependencies for offline use. Here’s how you can do it:

### Step-by-Step Guide (if you don't want to use your global env)

1. Set Up Your Project and Virtual Environment:
    - Create a virtual environment and install your dependencies.
```bash
    python -m venv myenv
    myenv\Scripts\activate
    pip install -r requirements.txt
```
2. Download Dependencies:
    - Use pip download to download all the dependencies into a directory.
```bash   
    mkdir dependencies
    pip download -r requirements.txt -d dependencies
```
3. Package the Project:
    - Copy the virtual environment and the dependencies directory into your project directory.
```bash
    xcopy myenv myproject\myenv /E /H /C /I
    xcopy dependencies myproject\dependencies /E /H /C /I
```
4. Zip the Project:
    - Zip the entire project directory, including the virtual environment and dependencies.
```bash   
    powershell Compress-Archive -Path myproject -DestinationPath myproject.zip
```
### Running the Project on Another PC

1. Unzip the Project:
    - Unzip the project on the target PC.
```bash   
    powershell Expand-Archive -Path myproject.zip -DestinationPath myproject
```    
2. Install Python:
    - Ensure Python is installed on the target PC. You can download it from the [official Python website](https://www.python.org/downloads/).

3. Activate the Virtual Environment:
    - Activate the virtual environment.
```bash   
    myproject\myenv\Scripts\activate
```
4. Install Dependencies Offline:
    - Install the dependencies from the local directory.
```bash   
    pip install --no-index --find-links myproject\dependencies -r myproject\requirements.txt
```    
5. Run the Project:
    - Run your project.
```bash   
    python myproject\main.py
```

# For calibration go to https://markhedleyjones.com/projects/calibration-checkerboard-collection for chessboard images