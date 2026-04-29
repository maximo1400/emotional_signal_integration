import time
import threading
import os
import queue
from pathlib import Path
import yaml
from dotenv import load_dotenv
from L1_band_power_capture.Emotiv.Emotiv import Train
from L1_band_power_capture.Simulated_pow.EmotionSimulator import EmotionSimulator

load_dotenv()


YAML_PATH = "config.yml"


def load_yml_config(path: str | Path = YAML_PATH) -> dict:
    """Parse a YAML file and return the Python object it represents."""
    path = Path(path).expanduser()
    dict = {}
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dict["pow_data_source"] = yml_data["pow_data_source"]

    return dict


def main() -> None:
    config = load_yml_config()
    # L1
    l1_out = queue.Queue()
    if config["pow_data_source"] == "virtual":
        # Virtual data source logic
        simulator = EmotionSimulator(l1_out)
        # simulator.plot_emotion_distribution()
        simulator.main_loop()
    else:  # Emotiv data source logic
        # Init Train
        profile_name = "Virtual"
        # list data streams
        streams = ["mot", "dev", "eq", "pow", "met", "com", "fac", "sys"]

        emotiv_client_id = os.getenv("APP_CLIENT_ID")
        emotiv_client_secret = os.getenv("APP_CLIENT_SECRET")

        emotiv = Train(
            emotiv_client_id,
            emotiv_client_secret,
            verbose=False,
            emotiv_profile=profile_name,
        )

        t0 = time.time()
        t = threading.Thread(target=emotiv.start, args=[profile_name, streams, l1_out])
        # t = threading.Thread(target=emotiv.start, args=(streams,))
        t.start()

        t.join()
        t1 = time.time()
        print(f"Tiempo de Aplicacion: {int(((t1 - t0) / 60) * 100) / 100} min")

        if os.path.exists(f"sub_data/{profile_name}"):
            profile_name = profile_name + "_" + str(int(time.time()))

        os.mkdir(f"sub_data/{profile_name}")
        for stream in emotiv.data:
            emotiv.data[stream].to_csv(f"sub_data/{profile_name}/{stream}.csv")


if __name__ == "__main__":
    main()
