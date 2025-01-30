import socket
import logging

class RobotSocketHandler:
    """Handles persistent socket communication with the robot controller."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """Establish connection with the robot controller."""
        try:
            if not self.connected:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.socket.connect((self.host, self.port))
                logging.info(f"Connected to {self.host}:{self.port}")
                self.connected = True
                return True
            return True  # Already connected
        except (socket.error, ConnectionRefusedError) as e:
            logging.error(f"Connection error: {e}")
            self.connected = False
            return False

    def send_data(self, data: str) -> bool:
        """Send data while maintaining persistent connection."""
        if not self.connected and not self.connect():
            return False

        try:
            self.socket.sendall(data.encode('utf-8'))
            logging.info(f"Sent: {data}")
            response = self.socket.recv(1024).decode('utf-8')
            logging.info(f"Received: {response}")
            return True
        except (socket.error, ConnectionResetError) as e:
            logging.error(f"Send error: {e}")
            self.close()
            return False

    def close(self):
        """Properly close the connection."""
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception as e:
                logging.error(f"Close error: {e}")
            self.socket = None
            self.connected = False
            logging.info("Connection closed")