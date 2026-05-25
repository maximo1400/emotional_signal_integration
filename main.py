import threading
import queue
from L1_band_power_capture.run import run_l1
from L2_emot_state_estimation.run import run_l2
from L3_va_data_adaptation.run import run_l3
from output_listener import run_l3_listener
from config_loader import get_config


def main() -> None:
    l1_out = queue.Queue()
    l2_out = queue.Queue()

    config = get_config(["classifier_mode"])
    classifier_mode = config["classifier_mode"]

    if classifier_mode == "train":
        run_l2(l1_out, l2_out)

    if classifier_mode in ["predict_from_queue", "predict_from_file"]:
        l3_listener_thread = threading.Thread(target=run_l3_listener, daemon=True)
        l3_listener_thread.start()

        l3_thread = threading.Thread(target=run_l3, args=(l2_out,), daemon=True)
        l3_thread.start()

        l2_thread = threading.Thread(target=run_l2, args=(l1_out, l2_out), daemon=True)
        l2_thread.start()

    if classifier_mode == "predict_from_queue":
        run_l1(l1_out)


if __name__ == "__main__":
    main()
