import os

class User:
    user_data_file = "T2C_PickAndPlace/Data/userData.txt"

    def __init__(self, user_id, name, password, role):
        self.user_id = user_id
        self.name = name
        self.password = password
        self.role = role

    @staticmethod
    def load_users():
        users = []
        if not os.path.exists(User.user_data_file):
            os.makedirs(os.path.dirname(User.user_data_file), exist_ok=True)
            open(User.user_data_file, 'w').close()  # Create file if it doesn't exist
        with open(User.user_data_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    user_id, name, password, role = line.split(',')
                    users.append(User(user_id, name, password, role))
        return users

    @staticmethod
    def save_user(user):
        with open(User.user_data_file, 'a') as file:
            file.write(f"{user.user_id},{user.name},{user.password},{user.role}\n")

    @staticmethod
    def user_exists(name):
        users = User.load_users()
        return any(user.name == name for user in users)

    @staticmethod
    def authenticate(name, password):
        users = User.load_users()
        for user in users:
            if user.name == name and user.password == password:
                return user
        return None