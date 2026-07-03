import sys
from pathlib import Path

from Dreamer.DB_adaptation.eeg_to_pow import dreamer_to_bandpower
from Dreamer.DB_adaptation.mat_to_df import convert_mat_to_df

# Add parent directory to path to import config_loader
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config_loader import get_config


def run_data_adaptation():
    # Retrieve config values
    keys = ["dreamer_mat_path", "feather_file_path", "epoch_sensors"]
    config = get_config(keys)
    electrodes = config["epoch_sensors"]["electrodes"]

    # Resolve the project root assuming this script is in Dreamer/DB_adaptation
    root_dir = Path(__file__).resolve().parent.parent.parent
    mat_path = root_dir / config["dreamer_mat_path"]
    bandpower_feather_path = root_dir / config["feather_file_path"]

    # Ensure output directories exist
    bandpower_feather_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: Reading MAT file and converting to DataFrame...")
    print(f"Reading from: {mat_path}")

    df_raw = convert_mat_to_df(str(mat_path), electrodes)

    print("\nStep 2: Processing EEG data to bandpower...")
    print(f"Writing to: {bandpower_feather_path}")

    df_bp_full = dreamer_to_bandpower(
        df_raw, electrodes, aggregate=None, output_db=True
    )
    df_bp_full.to_feather(str(bandpower_feather_path))

    print(f"Finished writing bandpower data to {bandpower_feather_path}")
