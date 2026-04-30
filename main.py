import time
import threading
import os
import queue
from pathlib import Path
import yaml
import signal
from dotenv import load_dotenv
from L1_band_power_capture.Emotiv.Emotiv import Subcribe
from L1_band_power_capture.Simulated_pow.EmotionSimulator import EmotionSimulator

load_dotenv()


YAML_PATH = "config.yml"
OUTPUT_DIR = "L1_band_power_capture/output_data"


# Load configuration from YAML file
def load_yml_config(path: str | Path = YAML_PATH) -> dict:
    """Parse a YAML file and return the Python object it represents."""
    path = Path(path).expanduser()
    dict = {}
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dict["pow_data_source"] = yml_data["pow_data_source"]

    return dict


# Create a stop event
stop_event = threading.Event()


# Handle Ctrl+C, used to kill emotiv thead
def signal_handler(signum, frame):
    print("\nStopping...")
    stop_event.set()


# Monitor the stop event and close the Emotiv connection when set
def monitor_stop(emotiv, stop_event):
    while not stop_event.is_set():
        time.sleep(0.1)
    emotiv.c.close()


def main() -> None:
    config = load_yml_config()
    # L1
    l1_out = queue.Queue()
    if config["pow_data_source"] == "virtual":
        # Virtual data source logic
        simulator = EmotionSimulator(l1_out)
        simulator.main_loop()

        l1_out_data_path = f"{OUTPUT_DIR}/{config['pow_data_source']}"
        if os.path.exists(l1_out_data_path):
            l1_out_data_path = l1_out_data_path + "_" + str(int(time.time()))

        os.mkdir(l1_out_data_path)
        simulator.output_df.to_csv(f"{l1_out_data_path}/pow.csv")

    else:  # Emotiv data source logic
        profile_name = config["pow_data_source"]
        streams = ["mot", "dev", "eq", "pow", "met", "com", "fac", "sys"]
        emotiv_client_id = os.getenv("APP_CLIENT_ID")
        emotiv_client_secret = os.getenv("APP_CLIENT_SECRET")

        emotiv = Subcribe(
            emotiv_client_id,
            emotiv_client_secret,
            verbose=False,
            emotiv_profile=profile_name,
        )

        monitor_thread = threading.Thread(
            target=monitor_stop, args=(emotiv, stop_event), daemon=True
        )
        monitor_thread.start()

        signal.signal(signal.SIGINT, signal_handler)

        t0 = time.time()
        t = threading.Thread(target=emotiv.start, args=[profile_name, streams, l1_out])

        t.start()
        while t.is_alive():
            t.join(timeout=1)
        t1 = time.time()

        print(f"Tiempo de Aplicacion: {int(((t1 - t0) / 60) * 100) / 100} min")

        if os.path.exists(f"{OUTPUT_DIR}/{profile_name}"):
            profile_name = profile_name + "_" + str(int(time.time()))

        os.mkdir(f"{OUTPUT_DIR}/{profile_name}")
        for stream in emotiv.data:
            emotiv.data[stream].to_csv(f"{OUTPUT_DIR}/{profile_name}/{stream}.csv")


if __name__ == "__main__":
    main()
