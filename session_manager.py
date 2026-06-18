import time
import os
import json
import requests
from config import username, password, app_key, app_version, login_url

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(SCRIPT_DIR, "session_token.json")

class SessionManager:
    def __init__(self):
        self.token_expiry_time = 600  # 10 minutes (in seconds)
        self.token_file = TOKEN_FILE

    def get_session_token(self):
        """Get the session token from the file or refresh it if expired."""
        if not self.is_token_valid():
            self.refresh_session_token()
        return self.read_token_from_file()

    def is_token_valid(self):
        """Check if the token exists and is still valid."""
        if not os.path.exists(self.token_file):
            return False
        
        try:
            with open(self.token_file, 'r') as file:
                data = json.load(file)
                last_retrieved_time = data.get("last_retrieved_time", 0)
                current_time = time.time()
                return current_time - last_retrieved_time < self.token_expiry_time
        except (json.JSONDecodeError, KeyError):
            return False  # If the file is corrupted or invalid, treat as expired.

    def read_token_from_file(self):
        """Read the session token from the file."""
        with open(self.token_file, 'r') as file:
            data = json.load(file)
            return data.get("session_token")

    def refresh_session_token(self):
        """Retrieve a new session token and save it to the file."""
        payload = {
            'Password': password,
            'UserName': username,
            'AppVersion': app_version,
            'AppComments': '',
            'AppKey': app_key
        }
        try:
            response = requests.post(login_url, json=payload, timeout=10)
            if response.status_code == 200:
                new_token = response.json().get('session')
                if new_token:
                    self.save_token_to_file(new_token)
                else:
                    raise Exception("Failed to retrieve a new session token: None returned.")
            else:
                raise Exception(f"Failed to refresh session token: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed during session refresh: {e}")
            raise

    def save_token_to_file(self, token):
        """Save the session token and the current time to the file."""
        data = {
            "session_token": token,
            "last_retrieved_time": time.time()
        }
        with open(self.token_file, 'w') as file:
            json.dump(data, file)

# Create a global session manager instance
session_manager = SessionManager()
