import tkinter as tk
from tkinter import messagebox
import os
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
from robot_socket_handler import RobotSocketHandler


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
SCRIPT_DIR = os.getenv("SCRIPT_DIR", "T2C_PickAndPlace/Script")
DATA_DIR = os.getenv("DATA_DIR", "T2C_PickAndPlace/Data/RobotArmObjectCoordinate")
TMP_POSITION_FILE = os.path.join(DATA_DIR, "Tmp_position.txt")
REAL_ROBOT_HOST = '192.168.125.1'
REAL_ROBOT_PORT = 1025
SIMULATION_HOST = '127.0.0.1'
SIMULATION_PORT = 55000

class RUNME_GUI:
    """Main GUI application for the Robotics System."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Robotics System GUI")
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(pady=20, padx=20)
        
        # Connection related variables
        self.connection_var = tk.StringVar(value="simulation")
        self.robot_socket_handler = None
        self.connection_established = False
        self.main_page()

    def main_page(self):
        """Main application page with all functionality access."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Robotics Control System", font=("Arial", 16)).pack(pady=10)
        
        
        # Main functionality buttons
        tk.Button(self.main_frame, text="Demo Object Detection", 
                command=self.demo_object_detection_diff_terminals, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Command Robot Arm", 
                command=self.connection_setup_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Camera Calibration", 
                command=self.calibration_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Exit", 
                command=self.window.quit, width=30).pack(pady=5)

    def connection_setup_page(self):
        """Page for setting up robot connection"""
        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Connection Setup", font=("Arial", 16)).pack(pady=10)
        
        connection_frame = tk.Frame(self.main_frame)
        connection_frame.pack(pady=10)
        
        # Connection type selection
        tk.Radiobutton(connection_frame, text=f"Simulation: {SIMULATION_HOST}:{SIMULATION_PORT}",
                      variable=self.connection_var, value="simulation").pack(anchor='w')
        tk.Radiobutton(connection_frame, text=f"Real Robot: {REAL_ROBOT_HOST}:{REAL_ROBOT_PORT}",
                      variable=self.connection_var, value="real").pack(anchor='w')
        
        # Connection controls
        self.connect_button = tk.Button(self.main_frame, text="Connect", command=self.establish_connection, width=20)
        self.connect_button.pack(pady=5)
        tk.Button(self.main_frame, text="Back", command=self.close_and_return_main, width=20).pack(pady=5)

    def establish_connection(self):
        """Attempt to establish connection with selected parameters"""
        self.connect_button.config(state=tk.DISABLED)
        if self.connection_var.get() == "simulation":
            host, port = SIMULATION_HOST, SIMULATION_PORT
        else:
            host, port = REAL_ROBOT_HOST, REAL_ROBOT_PORT

        # Create new handler if needed
        if not self.robot_socket_handler or \
           self.robot_socket_handler.host != host or \
           self.robot_socket_handler.port != port:
            if self.robot_socket_handler:
                self.robot_socket_handler.close()
            self.robot_socket_handler = RobotSocketHandler(host, port)

        def connection_attempt():
            try:
                connected = self.robot_socket_handler.connect()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                connected = False

            self.window.after(0, lambda: self.handle_connection_result(connected))

        threading.Thread(target=connection_attempt, daemon=True).start()

    def handle_connection_result(self, connected):
        """Handle connection attempt result"""
        self.connect_button.config(state=tk.NORMAL)
        if connected:
            self.connection_established = True

            self.command_robot_arm_page()
        else:
            messagebox.showerror("Connection Failed", 
                            "Failed to establish connection.\nPlease check:\n- Robot power\n- Network settings\n- Port availability")


    def close_and_return_main(self):
        """Close connection and return to main page"""
        if self.robot_socket_handler:
            self.robot_socket_handler.close()
        self.connection_established = False
        self.main_page()

    def command_robot_arm_page(self):
        """Robot control page (only accessible with active connection)"""
        if not self.connection_established:
            messagebox.showerror("Connection Required", "Please establish connection first")
            return

        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Control Panel", font=("Arial", 16)).pack(pady=10)
        
        # Connection status
        conn_type = "Simulation" if self.connection_var.get() == "simulation" else "Real Robot"
        tk.Label(self.main_frame, text=f"Connected to: {conn_type}", fg="green").pack(pady=5)
        
        # Control options
        tk.Button(self.main_frame, text="Stationary Objects", 
                command=self.send_coordinate_to_robot_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Moving Objects (Conveyor)", 
                command=self.robot_follow_coordinate_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Back", 
                command=self.close_and_return_main, width=30).pack(pady=5)

    def demo_object_detection_diff_terminals(self):
        """Run the object detection script in a new terminal."""
        script_path = os.path.join(SCRIPT_DIR, "object_detection_position_estimation.py")
        os.system(f"start cmd /k python {script_path}")

    def send_coordinate_to_robot_page(self):
        """Display the page for sending coordinates to the robot."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Pick and Place with Input Coordinate", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Manual Pick and Place", command=self.manual_pick_and_place, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Auto Pick and Place", command=self.auto_pick_and_place, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Back", command=self.command_robot_arm_page, width=30).pack(pady=5)

    def robot_follow_coordinate_page(self):
        """Placeholder for the follow coordinate feature."""
        messagebox.showinfo("Robot Follow the Coordinates", "This feature is not implemented yet.")

    def manual_pick_and_place(self):
        """Run the manual pick and place process."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Loading...", font=("Arial", 16)).pack(pady=10)
        self.window.update()

        # Run the capture and detection scripts in a separate thread
        threading.Thread(target=self.run_capture_and_detection).start()

    def run_capture_and_detection(self):
        """Run the capture and detection scripts."""
        try:
            capture_script = os.path.join(SCRIPT_DIR, "capture_image_srcipt.py")
            detection_script = os.path.join(SCRIPT_DIR, "Robot_Command", "image_Input_save_obj_Pose.py")

            if self.run_script(capture_script) and self.run_script(detection_script):
                self.load_manual_pick_and_place_page()
            else:
                messagebox.showerror("Error", "Failed to run capture and detection scripts.")
                self.command_robot_arm_page()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.command_robot_arm_page()

    def load_manual_pick_and_place_page(self):
        """Display the manual pick and place page."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Manual Pick and Place", font=("Arial", 16)).pack(pady=10)

        # Load the detection data
        positions = self.load_positions(TMP_POSITION_FILE)
        if not positions:
            messagebox.showerror("Error", "No positions found.")
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

    def send_to_robot(self, target: int):
        """Send selected positions to the robot."""
        selected_positions = [pos for var, pos in self.position_vars if var.get()]
        if not selected_positions:
            messagebox.showwarning("Warning", "No object selected.")
            return

        successful_positions = []
        for pos in selected_positions:
            x, y, z = pos[1], pos[2], pos[3]
            data = f"{x},{y},{z},{target}"
            
            # Attempt to send with reconnect
            if not self.robot_socket_handler.send_data(data):
                # If send failed, try to reconnect once
                if self.robot_socket_handler.connect():
                    if self.robot_socket_handler.send_data(data):
                        successful_positions.append(pos)
                else:
                    messagebox.showerror("Connection Lost", "Failed to reconnect to robot")
                    break
            else:
                successful_positions.append(pos)

        # Remove moved objects from Tmp_position.txt
        self.remove_positions(TMP_POSITION_FILE, successful_positions)
        self.load_manual_pick_and_place_page()

    def back_to_command_robot_arm_page(self):
        """Clear the position file and return to the command robot arm page."""
        open(TMP_POSITION_FILE, 'w').close()
        self.command_robot_arm_page()

    def auto_pick_and_place(self):
        """Run the auto pick and place process."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Auto Pick and Place", font=("Arial", 16)).pack(pady=10)
        self.window.update()

        # Run the capture and detection scripts in a separate thread
        threading.Thread(target=self.run_auto_capture_and_detection).start()

    def run_auto_capture_and_detection(self):
        """Run the auto capture and detection scripts."""
        try:
            capture_script = os.path.join(SCRIPT_DIR, "capture_image_srcipt.py")
            detection_script = os.path.join(SCRIPT_DIR, "Robot_Command", "image_Input_save_obj_Pose.py")

            if self.run_script(capture_script) and self.run_script(detection_script):
                self.load_auto_pick_and_place_page()
            else:
                messagebox.showerror("Error", "Failed to run capture and detection scripts.")
                self.command_robot_arm_page()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.command_robot_arm_page()

    def load_auto_pick_and_place_page(self):
        """Automatically send objects to their respective destinations."""
        positions = self.load_positions(TMP_POSITION_FILE)
        if not positions:
            messagebox.showerror("Error", "No positions found.")
            self.command_robot_arm_page()
            return

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
            self.robot_socket_handler.send_data(data)
            time.sleep(15)

        self.command_robot_arm_page()

    def calibration_page(self):
        """Display the camera calibration page."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Camera must be parallel, better checker board = better calibration", wraplength=400, justify="left").pack(pady=15)
        tk.Button(self.main_frame, text="Capture Checkerboard Image", command=self.capture_image).pack(pady=10)
        tk.Button(self.main_frame, text="Calibrate Camera", command=self.camera_calibration).pack(pady=10)
        tk.Button(self.main_frame, text="Test Camera", command=self.test_camera).pack(pady=10)
        tk.Button(self.main_frame, text="Back", command=self.main_page).pack(pady=10)

    def clear_frame(self):
        """Clear all widgets from the main frame."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def camera_calibration(self):
        """Run the camera calibration script."""
        script_path = os.path.join(SCRIPT_DIR, "camera_calibration.py")
        self.run_script(script_path)

    def capture_image(self):
        """Run the capture chessboard image script."""
        script_path = os.path.join(SCRIPT_DIR, "capture_chessboard_image_by_second.py")
        self.run_script(script_path)

    def test_camera(self):
        """Run the test camera script."""
        script_path = os.path.join(SCRIPT_DIR, "test_camera_copy.py")
        self.run_script(script_path)

    @staticmethod
    def run_script(script_path: str) -> bool:
        """Run a Python script using subprocess."""
        try:
            result = os.system(f"python {script_path}")
            return result == 0
        except Exception as e:
            logging.error(f"Error running script {script_path}: {e}")
            return False

    @staticmethod
    def load_positions(file_path: str) -> List[List[str]]:
        """Load positions from a file."""
        try:
            with open(file_path, 'r') as file:
                return [line.strip().split(',') for line in file]
        except FileNotFoundError:
            logging.error(f"{file_path} not found.")
            return []

    @staticmethod
    def remove_positions(file_path: str, positions: List[List[str]]):
        """Remove positions from a file."""
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:
                    if not any(pos[1] == line.strip().split(',')[1] for pos in positions):
                        file.write(line)
        except Exception as e:
            logging.error(f"Error removing positions: {e}")

    def on_window_close(self):
        """Handle window close event."""
        if self.robot_socket_handler:
            self.robot_socket_handler.close()
        self.window.destroy()


if __name__ == "__main__":
    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)    
    app.window.mainloop()