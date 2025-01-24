import socket
import time

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

def Socket_ABB_Send(corrdinate=None):
    # HOST = '127.0.0.1'  # Replace with your robot controller's IP address
    # PORT = 55000        # The same port as used in the RAPID code
    HOST = '192.168.125.1'  # Replace with your robot controller's IP address
    PORT = 1025  
    # Connect to the robot controller
    robot_socket = connect_to_robot(HOST, PORT)
    
    if robot_socket is None:
        print("Failed to connect to the robot controller")
        return
    try:
        while True:
            if corrdinate is None:
                user_input = input("Enter coordinates and target (x,y,z,bin_target) or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
                try:
                    x, y, z, target = user_input.split(',')
                except ValueError:
                    print("Invalid input format. Please enter in the format: x,y,z,bin_target")
                    continue
            else:
                x = corrdinate[0]
                y = corrdinate[1]
                z = corrdinate[2]
                target = corrdinate[3]

            data = f"{x}, {y}, {z}, {target}"
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