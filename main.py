import threading
import queue
from L1_band_power_capture.run import run_l1
from L2_emot_state_estimation.run import predict_from_queue
from L3_va_data_adaptation.run import run_l3
from config_loader import get_config


def main() -> None:
    classifier_mode = get_config(["classifier_mode"])
    l1_out = queue.Queue()
    l2_out = queue.Queue()

    if classifier_mode == "predict_from_queue":
        # Start L3 thread
        l3_thread = threading.Thread(target=run_l3, args=(l2_out,), daemon=True)
        l3_thread.start()

        # Start L2 thread
        l2_thread = threading.Thread(
            target=predict_from_queue, args=(l1_out, l2_out), daemon=True
        )
        l2_thread.start()
    run_l1(l1_out)


if __name__ == "__main__":
    main()
