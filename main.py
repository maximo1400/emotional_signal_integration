import os
import threading
import queue
from L1_band_power_capture.run import run_l1
from L2_emot_state_estimation.run import run_l2
from L3_va_data_adaptation.run import run_l3
from output_listener import run_l3_listener
from config_loader import get_config
from Dreamer.DB_adaptation.run_adaptation import run_data_adaptation


def main() -> None:
    l1_out = queue.Queue()
    l2_out = queue.Queue()

    config = get_config(
        [
            "classifier_mode",
            "start_socket_listener",
            "feather_file_path",
            "pow_data_source",
        ]
    )
    classif_mode = config["classifier_mode"]
    listen_out = config["start_socket_listener"]
    pow_file_path = config["feather_file_path"]
    pow_source = config["pow_data_source"]

    if not os.path.exists(pow_file_path) and (
        classif_mode == "train" or pow_source == "virtual"
    ):
        print(
            f"Band power file not found at {pow_file_path}. Running data adaptation..."
        )
        run_data_adaptation()
    if classif_mode == "train":
        run_l2(l1_out, l2_out)

    if classif_mode in ["predict_from_queue", "predict_from_file"]:
        if listen_out:
            l3_listener_thread = threading.Thread(target=run_l3_listener, daemon=True)
            l3_listener_thread.start()

        l3_thread = threading.Thread(target=run_l3, args=(l2_out,), daemon=True)
        l3_thread.start()

        l2_thread = threading.Thread(target=run_l2, args=(l1_out, l2_out), daemon=True)
        l2_thread.start()

    if classif_mode == "predict_from_queue":
        run_l1(l1_out)


if __name__ == "__main__":
    main()
