"""
Layer 1 runner.

Modes:
- virtual: run the simulated band power stream and persist pow.csv
- emotiv: connect to Emotiv and persist per-stream CSVs
"""

import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from L1_band_power_capture.EmotionSimulator import EmotionSimulator
from L1_band_power_capture.Emotiv.Emotiv import Subcribe
from L1_band_power_capture.utils import L1OutputWriter

# Add parent directory to path to import config_loader
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


def run_l1(l1_out_queue: queue.Queue, starting_timestamp: float):
    config = get_config([
        "pow_data_source",
        "L1_output_folder",
        "emotiv_streams",
        "profile_name",
        "verbose",
        "save_output_files",
        "POW_COLUMNS",
    ])

    ds = config["pow_data_source"]
    st_ts = int(starting_timestamp)
    output_file = Path(config["L1_output_folder"]) / f"out_{st_ts}.csv"
    writer = L1OutputWriter(output_file, config["save_output_files"])
    writer.write_headers(config["POW_COLUMNS"])

    if ds == "virtual":
        simulator = EmotionSimulator(
            l1_out_queue,
            starting_timestamp,
            writer=writer,
            data_origin=ds,
        )
        simulator.main_loop()
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
        starting_timestamp=starting_timestamp,
        writer=writer,
        data_origin=ds,
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
        output_dir = Path(config["L1_output_folder"]) / f"out_{st_ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        for stream in emotiv.data:
            if stream == "pow":
                continue  # pow is written via L1OutputWriter
            raw_stream_data = emotiv.data[stream]
            if isinstance(raw_stream_data, list):
                cols = getattr(emotiv, "data_labels", {}).get(stream, None)
                data_to_save = pd.DataFrame(raw_stream_data, columns=cols)
            else:
                data_to_save = raw_stream_data
            data_to_save["data_origin"] = ds
            data_to_save.to_csv(output_dir / f"{stream}.csv", index=False)

        writer.close()
