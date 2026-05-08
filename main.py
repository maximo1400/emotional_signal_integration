import time
import threading
import os
import queue
import signal
from dotenv import load_dotenv
from L1_band_power_capture.Emotiv.Emotiv import Subcribe
from L1_band_power_capture.Simulated_pow.EmotionSimulator import EmotionSimulator
from config_loader import get_config

load_dotenv()


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
    config = get_config(
        ["pow_data_source", "L1_output_folder", "emotiv_streams", "profile_name"]
    )

    # L1
    l1_out = queue.Queue()
    if config["pow_data_source"] == "virtual":
        # Virtual data source logic
        simulator = EmotionSimulator(l1_out)
        simulator.main_loop()

        l1_out_data_path = f"{config['L1_output_folder']}/{config['pow_data_source']}"
        if os.path.exists(l1_out_data_path):
            l1_out_data_path = l1_out_data_path + "_" + str(int(time.time()))

        os.mkdir(l1_out_data_path)
        simulator.output_df.to_csv(f"{l1_out_data_path}/pow.csv")

    else:  # Emotiv data source logic
        profile_name = config["profile_name"]
        streams = config["emotiv_streams"]
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

        l1_out_data_path = f"{config['L1_output_folder']}/{profile_name}"
        if os.path.exists(l1_out_data_path):
            l1_out_data_path = l1_out_data_path + "_" + str(int(time.time()))

        os.mkdir(l1_out_data_path)
        for stream in emotiv.data:
            emotiv.data[stream].to_csv(f"{l1_out_data_path}/{stream}.csv")


if __name__ == "__main__":
    main()
