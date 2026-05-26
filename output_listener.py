import json
import socket
import time
from pathlib import Path
import sys

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config


def _connect_tcp(host: str, port: int, retry_delay: float = 1.0) -> socket.socket:
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return sock
        except OSError:
            time.sleep(retry_delay)


def run_l3_listener() -> None:
    config = get_config(["socket_config"])
    sock_cfg = config["socket_config"]
    host = sock_cfg["host"]
    port = sock_cfg["port"]
    protocol = sock_cfg["protocol"].lower()

    print(f"Socket listener starting on {host}:{port} ({protocol})")

    if protocol == "tcp":
        sock = _connect_tcp(host, port)
        with sock:
            buffer = ""
            while True:
                data = sock.recv(4096)
                if not data:
                    time.sleep(0.2)
                    continue
                buffer += data.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        print(f"Socket payload: {payload}\n")
                    except json.JSONDecodeError:
                        print(f"socket raw: {line}")
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        with sock:
            while True:
                data, addr = sock.recvfrom(4096)
                line = data.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    print(f"socket payload from {addr}: {payload}")
                except json.JSONDecodeError:
                    print(f"socket raw from {addr}: {line}")


if __name__ == "__main__":
    run_l3_listener()
