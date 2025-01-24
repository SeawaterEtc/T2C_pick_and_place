import socket
import time

# from coodinate_estimate import auruco_setup

def connect_to_robot(host, port):
    """ Connect to the robot controller and return the socket object. """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        print("Connected to the robot controller")
        return s
    except socket.error as e:
        print("Socket error: ", e)
        return None

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

def Socket_ABB_Send(corrdinate = None):
    HOST = '127.0.0.1'  # Replace with your robot controller's IP address
    PORT = 55000           # The same port as used in the RAPID code
    # Connect to the robot controller
    robot_socket = connect_to_robot(HOST, PORT)
    
    if robot_socket is None:
        print("Failed to connect to the robot controller")
        return
    try:
        while True:
            # corrdinate = auruco_setup()
            if corrdinate == None:
                # [202.01,317.202,-10]
                x = input("Enter x (or 'q' to quit): ")
                if x.lower() == 'q':
                    break
                y = input("Enter y: ")
                z = input("Enter z: ")
            else: 
                x = corrdinate[0]
                y = corrdinate[1]
                z = 0
     
            data = f"[{x}, {y}, {z}]"
            send_message(robot_socket, data)
            time.sleep(2)
            print(data)

            response = receive_message(robot_socket)
            if response is None:
                break
    finally:
        robot_socket.close()
        print("Connection closed")
        
def main(): 
    Socket_ABB_Send()

if __name__ == "__main__":
    main()