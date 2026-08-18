import builtins
import os
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import _load_config  # noqa: E402
from main import main  # noqa: E402


def clear_config_cache() -> None:
    _load_config.cache_clear()


def test_l1_virtual_predict_from_queue() -> Path:
    print("\n==================================================")
    print("Test 1: L1 Virtual + L2 Predict from Queue")
    print("==================================================")

    test_args = [
        "main.py",
        "--pow_data_source",
        "virtual",
        "--classifier_mode",
        "predict_from_queue",
        "--save_output_files",
        "true",
        "--start_socket_listener",
        "true",
    ]

    t_start = time.time()
    input_responses = iter(["5", "q"])

    with (
        patch.object(sys, "argv", test_args),
        patch.object(builtins, "input", lambda _: next(input_responses)),
    ):
        clear_config_cache()
        main()

    l1_dir = ROOT_DIR / "output_data" / "L1_band_power_capture"
    l2_dir = ROOT_DIR / "output_data" / "L2_emot_state_estimation"
    l3_dir = ROOT_DIR / "output_data" / "L3_va_data_adaptation"

    l1_files = [f for f in l1_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]
    l2_files = [f for f in l2_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]
    l3_files = [f for f in l3_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]

    assert len(l1_files) > 0, "L1 Virtual output file was not created"
    assert len(l2_files) > 0, "L2 output file was not created"
    assert len(l3_files) > 0, "L3 output file was not created"

    time.sleep(0.2)  # Allow thread file handles to complete flushing to disk
    latest_l1 = max(l1_files, key=lambda p: p.stat().st_mtime)
    latest_l2 = max(l2_files, key=lambda p: p.stat().st_mtime)
    latest_l3 = max(l3_files, key=lambda p: p.stat().st_mtime)

    df_l1 = pd.read_csv(latest_l1)
    df_l2 = pd.read_csv(latest_l2)
    df_l3 = pd.read_csv(latest_l3)

    assert len(df_l1) > 0, "L1 CSV is empty"
    assert len(df_l2) > 0, "L2 CSV is empty"
    assert len(df_l3) > 0, "L3 CSV is empty"
    assert df_l1["data_origin"].iloc[0] == "virtual", (
        "Expected data_origin to be 'virtual'"
    )

    print(f"Verified L1 Virtual CSV ({len(df_l1)} rows): {latest_l1.name}")
    print(f"Verified L2 Prediction CSV ({len(df_l2)} rows): {latest_l2.name}")
    print(f"Verified L3 Output CSV ({len(df_l3)} rows): {latest_l3.name}")

    return latest_l1


def test_l1_emotiv_predict_from_queue() -> Path:
    print("\n==================================================")
    print(
        "Test 2: L1 Real Emotiv Mode + L2 Predict from Queue (Runs real Emotiv capture, 10s wait, Ctrl+C SIGINT)"
    )
    print("==================================================")

    test_args = [
        "main.py",
        "--pow_data_source",
        "emotiv",
        "--classifier_mode",
        "predict_from_queue",
        "--save_output_files",
        "true",
        "--start_socket_listener",
        "true",
    ]

    t_start = time.time()

    def send_sigint_after_delay(delay_sec: float = 20.0):
        print(
            f"[Timer] Waiting {delay_sec}s for Emotiv data capture, then sending Ctrl+C (SIGINT)..."
        )
        time.sleep(delay_sec)
        print(
            "\n[Timer] 10 seconds reached. Sending Ctrl+C (SIGINT) signal to stop Emotiv capture..."
        )
        if hasattr(signal, "raise_signal"):
            signal.raise_signal(signal.SIGINT)
        else:
            os.kill(os.getpid(), signal.SIGINT)

    timer_thread = threading.Thread(
        target=send_sigint_after_delay, args=(10.0,), daemon=True
    )
    timer_thread.start()

    with patch.object(sys, "argv", test_args):
        clear_config_cache()
        main()

    l1_dir = ROOT_DIR / "output_data" / "L1_band_power_capture"
    l2_dir = ROOT_DIR / "output_data" / "L2_emot_state_estimation"
    l3_dir = ROOT_DIR / "output_data" / "L3_va_data_adaptation"

    l1_files = [f for f in l1_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]
    l2_files = [f for f in l2_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]
    l3_files = [f for f in l3_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]

    assert len(l1_files) > 0, "L1 Emotiv output file missing"

    latest_l1 = max(l1_files, key=lambda p: p.stat().st_mtime)
    df_l1 = pd.read_csv(latest_l1)

    assert df_l1["data_origin"].iloc[0] == "emotiv", (
        "Expected data_origin to be 'emotiv'"
    )
    print(f"Verified L1 Emotiv POW CSV ({len(df_l1)} rows): {latest_l1.name}")

    st_ts = latest_l1.stem.replace("out_", "")
    stream_folder = l1_dir / f"out_{st_ts}"
    if stream_folder.exists():
        stream_csvs = list(stream_folder.glob("*.csv"))
        print(
            f"Verified Emotiv per-stream folder: {stream_folder.name} ({len(stream_csvs)} stream files)"
        )

    if len(l2_files) > 0:
        latest_l2 = max(l2_files, key=lambda p: p.stat().st_mtime)
        df_l2 = pd.read_csv(latest_l2)
        print(f"Verified L2 Prediction CSV ({len(df_l2)} rows): {latest_l2.name}")

    if len(l3_files) > 0:
        latest_l3 = max(l3_files, key=lambda p: p.stat().st_mtime)
        df_l3 = pd.read_csv(latest_l3)
        print(f"Verified L3 Output CSV ({len(df_l3)} rows): {latest_l3.name}")

    return latest_l1


def test_l2_predict_from_file(l1_pow_csv: Path) -> None:
    print("\n==================================================")
    print(f"Test 3: L2 Predict from File using {l1_pow_csv.name}")
    print("==================================================")

    test_args = [
        "main.py",
        "--classifier_mode",
        "predict_from_file",
        "--predict_from_file_pow_csv_path",
        str(l1_pow_csv),
        "--save_output_files",
        "true",
        "--start_socket_listener",
        "true",
    ]

    t_start = time.time()
    with patch.object(sys, "argv", test_args):
        clear_config_cache()
        main()

    l2_dir = ROOT_DIR / "output_data" / "L2_emot_state_estimation"
    l3_dir = ROOT_DIR / "output_data" / "L3_va_data_adaptation"

    l2_files = [f for f in l2_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]
    l3_files = [f for f in l3_dir.glob("out_*.csv") if f.stat().st_mtime >= t_start - 5]

    assert len(l2_files) > 0, "L2 predict_from_file output file missing"
    assert len(l3_files) > 0, "L3 predict_from_file output file missing"

    latest_l2 = max(l2_files, key=lambda p: p.stat().st_mtime)
    latest_l3 = max(l3_files, key=lambda p: p.stat().st_mtime)

    df_l1_in = pd.read_csv(l1_pow_csv)
    df_l2 = pd.read_csv(latest_l2)
    df_l3 = pd.read_csv(latest_l3)

    assert len(df_l2) == len(df_l1_in), (
        f"Expected {len(df_l1_in)} predictions in L2, got {len(df_l2)}"
    )
    assert len(df_l3) == len(df_l1_in), (
        f"Expected {len(df_l1_in)} predictions in L3, got {len(df_l3)}"
    )

    print(f"Verified predict_from_file L2 CSV ({len(df_l2)} rows): {latest_l2.name}")
    print(f"Verified predict_from_file L3 CSV ({len(df_l3)} rows): {latest_l3.name}")


def main_test():
    print(
        "Starting integration test suite for all application modes (Real App Execution)..."
    )
    l1_virtual_csv = test_l1_virtual_predict_from_queue()
    l1_emotiv_csv = test_l1_emotiv_predict_from_queue()
    test_l2_predict_from_file(l1_virtual_csv)
    df_emotiv = pd.read_csv(l1_emotiv_csv)
    if len(df_emotiv) > 0:
        test_l2_predict_from_file(l1_emotiv_csv)

    print("\n" + "=" * 60)
    print(
        ">>> ALL TESTS (VIRTUAL, REAL EMOTIV 10S CAPTURE & SIGINT, PREDICT FROM QUEUE & FILE) PASSED! <<<"
    )
    print("=" * 60)


if __name__ == "__main__":
    main_test()
