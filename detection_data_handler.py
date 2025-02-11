from robot_socket_handler import RobotSocketHandler

class DetectionDataHandler:
    """Handles sending detection data to the robot controller."""

    def __init__(self, host: str, port: int):
        self.socket_handler = RobotSocketHandler(host, port)

    def send_detection_data(self, data: str) -> bool:
        """Send detection data to the robot controller."""
        return self.socket_handler.send_data(data)
    
    