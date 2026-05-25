import sys
import queue
import socket
import json
import threading
import csv
import time
from pathlib import Path

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config
from L3_va_data_adaptation.smoother import DataSmoother


def parse_label(label_str: str) -> tuple[float, float]:
    """Parses 'v_a' label into valence and arousal floats."""
    v, a = label_str.split("_")
    return float(v), float(a)


def run_l3(l2_out_queue: queue.Queue):
    config = get_config(["smoothing_method", "smoothing_parameters", "socket_config"])

    method = config["smoothing_method"]
    params = config["smoothing_parameters"][method]
    smoother = DataSmoother(method, params)

    sock_cfg = config["socket_config"]
    host = sock_cfg["host"]
    port = sock_cfg["port"]
    protocol = sock_cfg["protocol"].lower()

    print(f"L3 starting with {method} smoothing. Socket on {host}:{port} ({protocol})")

    output_dir = Path(__file__).parent / "output_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"out_{int(time.time())}.csv"

    # Setup socket
    if protocol == "tcp":
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"Waiting for TCP connection on {host}:{port}...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        # Optionally, flush the queue of any backlog that built up while waiting
        while not l2_out_queue.empty():
            l2_out_queue.get_nowait()
    else:
        # UDP broadcaster
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn = server_socket
        addr = (host, port)

    try:
        print(f"L3 writing output to {output_file}")
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "raw_valence",
                    "raw_arousal",
                    "smoothed_valence",
                    "smoothed_arousal",
                    "confidence",
                    "timestamp",
                ]
            )

            while True:
                data = l2_out_queue.get()
                if data is None:
                    print("L3 queue received stop signal")
                    break

                raw_v, raw_a = parse_label(str(data["label"]))
                smooth_v, smooth_a = smoother.smooth(raw_v, raw_a)

                payload = {
                    "raw_valence": raw_v,
                    "raw_arousal": raw_a,
                    "smoothed_valence": smooth_v,
                    "smoothed_arousal": smooth_a,
                    "confidence": data.get("confidence", 0.0),
                    "timestamp": data.get("timestamp", 0.0),
                }

                writer.writerow(
                    [
                        payload["raw_valence"],
                        payload["raw_arousal"],
                        payload["smoothed_valence"],
                        payload["smoothed_arousal"],
                        payload["confidence"],
                        payload["timestamp"],
                    ]
                )
                f.flush()

                message = json.dumps(payload) + "\n"

                try:
                    if protocol == "tcp":
                        conn.sendall(message.encode("utf-8"))
                    else:
                        conn.sendto(message.encode("utf-8"), addr)
                except Exception as e:
                    print(f"Socket error: {e}")
                    if protocol == "tcp":
                        print("Waiting for new TCP connection...")
                        conn.close()
                        conn, addr = server_socket.accept()
                        print(f"Connected by {addr}")
                        while not l2_out_queue.empty():
                            l2_out_queue.get_nowait()
    finally:
        if protocol == "tcp" and "conn" in locals() and conn != server_socket:
            conn.close()
        server_socket.close()


if __name__ == "__main__":
    import time

    q = queue.Queue()
    t = threading.Thread(target=run_l3, args=(q,), daemon=True)
    t.start()

    time.sleep(1)
    q.put({"label": "1_2", "confidence": 0.9, "timestamp": time.time()})
    q.put({"label": "2_2", "confidence": 0.8, "timestamp": time.time()})

    while t.is_alive():
        time.sleep(1)
