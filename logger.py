import time
import os

class ActivityLogger:
    def __init__(self, log_file="outputs/logs.txt"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write("\n--- New Session Started ---\n")

    def log(self, status):
        """Logs a general status message with a timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} - {status}\n")
