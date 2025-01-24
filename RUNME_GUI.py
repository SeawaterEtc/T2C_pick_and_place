import tkinter as tk
from tkinter import messagebox, ttk
import os
import threading
import socket
import time

class RUNME_GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Robotics System GUI")

        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(pady=100, padx=100)

        self.robot_socket = None  # Initialize the robot socket
        self.main_page()

    def main_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Choose an Action", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Demo Object Detection", command=self.demo_object_detection_diff_terminals, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Command Robot Arm", command=self.command_robot_arm_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Camera Calibration", command=self.calibration_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Exit", command=self.window.quit, width=30).pack(pady=5)

    def demo_object_detection_diff_terminals(self):
        os.system("start cmd /k python object_detection_position_estimation.py")

    def command_robot_arm_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Command The Robot", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Send Coordinate to Robot", command=self.send_coordinate_to_robot_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Robot Follow the Coordinates", command=self.robot_follow_coordinate_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Back", command=self.main_page, width=30).pack(pady=5)

        # Establish socket connection if not already connected
        if self.robot_socket is None:
            if not self.connect_to_robot():
                messagebox.showerror("Connection Error", "Failed to connect to the robot controller. Please ensure the robot is turned on.")
                self.main_page()

    def connect_to_robot(self):
        # # connect to simulated controller (Robot Studio simulation)
        # HOST = '127.0.0.1'
        # PORT = 55000  
        
        # connect to real controller (real robot, change this address to the robot controller IP address)
        HOST = '192.168.125.1'
        PORT = 1025 
        
        
        try:
            self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.robot_socket.connect((HOST, PORT))
            print("Connected to the robot controller")
            return True
        except (socket.error, ConnectionRefusedError) as e:
            print("Socket error: ", e)
            self.robot_socket = None
            return False

    def send_coordinate_to_robot_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Pick and Place with Input Coordinate", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Manual Pick and Place", command=self.manual_pick_and_place, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Auto Pick and Place", command=self.auto_pick_and_place, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Back", command=self.command_robot_arm_page, width=30).pack(pady=5)

    def robot_follow_coordinate_page(self):
        messagebox.showinfo("Robot Follow the Coordinates", "This feature is not implemented yet.")

    def manual_pick_and_place(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Loading...", font=("Arial", 16)).pack(pady=10)
        self.window.update()

        # Run the capture and detection scripts in a separate thread
        threading.Thread(target=self.run_capture_and_detection).start()

    def run_capture_and_detection(self):
        try:
            result1 = os.system("python T2C_PickAndPlace/Script/capture_image_srcipt.py")
            result2 = os.system("python T2C_PickAndPlace/Script/Robot_Command/image_Input_save_obj_Pose.py")
            # result2 = os.system("start cmd /k python T2C_PickAndPlace/Script/Robot_Command/image_Input_save_obj_Pose.py")
            
            if result1 == 0 and result2 == 0:
                self.load_manual_pick_and_place_page()
            else:
                messagebox.showerror("Error", "Failed to run capture and detection scripts.")
                self.command_robot_arm_page()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.command_robot_arm_page()

    def load_manual_pick_and_place_page(self):
        self.clear_frame()

        tk.Label(self.main_frame, text="Manual Pick and Place", font=("Arial", 16)).pack(pady=10)

        # Load the detection data
        positions = []
        try:
            with open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'r') as file:
                for line in file:
                    positions.append(line.strip().split(','))
        except FileNotFoundError:
            messagebox.showerror("Error", "Tmp_position.txt not found.")
            self.command_robot_arm_page()
            return

        # Display the detection data
        self.position_vars = []
        for pos in positions:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.main_frame, text=f"{pos[0]}: ({pos[1]}, {pos[2]}, {pos[3]})", variable=var)
            chk.pack(anchor='w')
            self.position_vars.append((var, pos))

        # Destination buttons
        tk.Button(self.main_frame, text="Incineration", command=lambda: self.send_to_robot(1), width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Recycling", command=lambda: self.send_to_robot(2), width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Back", command=self.back_to_command_robot_arm_page, width=30).pack(pady=5)

    def send_to_robot(self, target):
        selected_positions = [pos for var, pos in self.position_vars if var.get()]
        if not selected_positions:
            messagebox.showwarning("Warning", "No object selected.")
            return

        successful_positions = []
        for pos in selected_positions:
            x, y, z = pos[1], pos[2], pos[3]
            data = f"{x},{y},{z},{target}"
            if self.send_socket_data(data):
                successful_positions.append(pos)

        # Remove moved objects from Tmp_position.txt if the socket connection was successful
        with open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'r') as file:
            lines = file.readlines()

        with open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'w') as file:
            for line in lines:
                if not any(pos[1] == line.strip().split(',')[1] for pos in successful_positions):
                    file.write(line)

        self.load_manual_pick_and_place_page()

    def send_socket_data(self, data):
        if self.robot_socket is None:
            print("No connection to the robot controller")
            return False

        def send_message(s, message):
            """ Send a message to the robot controller. """
            try:
                s.sendall(message.encode('utf-8'))
                print("Sent: ", message)
            except socket.error as e:
                print("Send error: ", e)

        def receive_message(s, buffer_size=1024):
            """ Receive a message from the robot controller. """
            try:
                data = s.recv(buffer_size)
                print("Received: ", data.decode('utf-8'))
                return data.decode('utf-8')
            except socket.error as e:
                print("Receive error: ", e)
                return None

        try:
            send_message(self.robot_socket, data)
            time.sleep(2)
            print(data)

            response = receive_message(self.robot_socket)
            if response is None:
                print("No response from robot controller")
                return False
        except socket.error as e:
            print("Socket error: ", e)
            return False
        return True

    def back_to_command_robot_arm_page(self):
        # Delete all objects in Tmp_position.txt
        open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'w').close()
        self.command_robot_arm_page()

    def back_to_main_page(self):
        self.main_page()

    def auto_pick_and_place(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Auto Pick and Place", font=("Arial", 16)).pack(pady=10)
        self.window.update()

        # Run the capture and detection scripts in a separate thread
        threading.Thread(target=self.run_auto_capture_and_detection).start()

    def run_auto_capture_and_detection(self):
        try:
            result1 = os.system("python T2C_PickAndPlace/Script/capture_image_srcipt.py")
            result2 = os.system("python T2C_PickAndPlace/Script/Robot_Command/image_Input_save_obj_Pose.py")
            if result1 == 0 and result2 == 0:
                self.load_auto_pick_and_place_page()
            else:
                messagebox.showerror("Error", "Failed to run capture and detection scripts.")
                self.command_robot_arm_page()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.command_robot_arm_page()

    def load_auto_pick_and_place_page(self):
        self.clear_frame()

        tk.Label(self.main_frame, text="Auto Pick and Place", font=("Arial", 16)).pack(pady=10)

        # Load the detection data
        positions = []
        try:
            with open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'r') as file:
                for line in file:
                    positions.append(line.strip().split(','))
        except FileNotFoundError:
            messagebox.showerror("Error", "Tmp_position.txt not found.")
            self.command_robot_arm_page()
            return

        # Automatically send objects to their respective destinations
        for pos in positions:
            class_name = pos[0]
            x, y, z = pos[1], pos[2], pos[3]
            if class_name == "Plastic-bottle":
                target = 2
            elif class_name == "Snack-packet":
                target = 1
            elif class_name == "Aluminum-cans":
                target = 2
            else:
                continue  # Skip unknown classes

            data = f"{x},{y},{z},{target}"
            self.send_socket_data(data)

            time.sleep(15)
            
        # # Clear the Tmp_position.txt file after processing
        # open('T2C_PickAndPlace/Data/RobotArmObjectCoordinate/Tmp_position.txt', 'w').close()

        self.command_robot_arm_page()

    def calibration_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="If re-calibrate, please copy path of the checkerboard image to a browser", wraplength=400, justify="left").pack(pady=15)
        tk.Button(self.main_frame, text="Capture Checkerboard Image", command=self.capture_image).pack(pady=10)
        tk.Button(self.main_frame, text="Calibrate Camera - put 49mm A4 checker board on a screen - take as many image as you can from every angle and surfaces", command=self.camera_calibration).pack(pady=10)
        tk.Button(self.main_frame, text="Test Camera", command=self.test_camera).pack(pady=10)
        tk.Button(self.main_frame, text="Back", command=self.main_page).pack(pady=10)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def camera_calibration(self):
        os.system("python T2C_PickAndPlace/Script/camera_calibration.py")

    def capture_image(self):
        os.system("python T2C_PickAndPlace/Script/capture_chessboard_image_by_second.py")

    def test_camera(self):
        os.system("python T2C_PickAndPlace/Script/test_camera_copy.py")

if __name__ == "__main__":
    app = RUNME_GUI()
    app.window.mainloop()