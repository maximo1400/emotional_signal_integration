"""
Layer 1 runner.

Modes:
- virtual: run the simulated band power stream and persist pow.csv
- emotiv: connect to Emotiv and persist per-stream CSVs
"""

import os
import queue
import signal
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from L1_band_power_capture.Emotiv.Emotiv import Subcribe
from L1_band_power_capture.EmotionSimulator import EmotionSimulator

# Add parent directory to path to import config_loader
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config

load_dotenv()


def _signal_handler(stop_event: threading.Event, *_):
    print("\nStopping...")
    stop_event.set()


def _monitor_stop(emotiv, stop_event: threading.Event, l1_out_queue: queue.Queue):
    while not stop_event.is_set():
        time.sleep(0.1)
    # print("Closing Emotiv connection...")
    l1_out_queue.put(None)
    emotiv.c.close()


def _build_l1_output_path(base_folder: str, suffix: str) -> str:
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    output_path = f"{base_folder}/{suffix}"
    if os.path.exists(output_path):
        output_path = output_path + "_" + str(int(time.time()))

    os.mkdir(output_path)
    return output_path


def run_l1(l1_out_queue: queue.Queue):
    config = get_config(
        [
            "pow_data_source",
            "L1_output_folder",
            "emotiv_streams",
            "profile_name",
            "verbose",
            "save_output_files",
        ]
    )

    if config["pow_data_source"] == "virtual":
        simulator = EmotionSimulator(l1_out_queue)
        simulator.main_loop()

        if config["save_output_files"]:
            output_path = _build_l1_output_path(
                config["L1_output_folder"],
                config["pow_data_source"],
            )
            if simulator.output_df is None:
                raise ValueError("output_df not generated")
            simulator.output_df.to_csv(f"{output_path}/pow.csv")
        return

    # Emotiv data source logic
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *args: _signal_handler(stop_event, *args))

    profile_name = config["profile_name"]
    streams = config["emotiv_streams"]
    emotiv_client_id = os.getenv("APP_CLIENT_ID")
    emotiv_client_secret = os.getenv("APP_CLIENT_SECRET")

    emotiv = Subcribe(
        emotiv_client_id,
        emotiv_client_secret,
        verbose=config["verbose"],
        emotiv_profile=profile_name,
    )

    monitor_thread = threading.Thread(
        target=_monitor_stop,
        args=(emotiv, stop_event, l1_out_queue),
        daemon=True,
    )
    monitor_thread.start()

    t0 = time.time()
    t = threading.Thread(
        target=emotiv.start, args=[profile_name, streams, l1_out_queue]
    )

    t.start()
    while t.is_alive():
        t.join(timeout=1)
    t1 = time.time()

    print(f"Tiempo de Aplicacion: {int(((t1 - t0) / 60) * 100) / 100} min")

    if config["save_output_files"]:
        output_path = _build_l1_output_path(config["L1_output_folder"], profile_name)
        for stream in emotiv.data:
            emotiv.data[stream].to_csv(f"{output_path}/{stream}.csv")
