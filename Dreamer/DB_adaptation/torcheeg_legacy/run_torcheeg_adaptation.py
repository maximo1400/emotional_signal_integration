import sys
from pathlib import Path

# Add root directory to sys.path to resolve imports regardless of where script is executed
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config_loader import get_config  # noqa: E402

from dreamer_to_feather import (
    convert_mat_to_df,
)  # noqa: E402

from Dreamer.DB_adaptation.eeg_to_pow import (
    dreamer_to_bandpower,
)  # noqa: E402


def main():
    # Retrieve config values
    keys = ["dreamer_mat_path", "feather_file_path"]
    config = get_config(keys)

    # Resolve the project root assuming this script is in Dreamer/DB_adaptation
    root_dir = Path(__file__).resolve().parent.parent.parent

    mat_path = root_dir / config["dreamer_mat_path"]
    bandpower_feather_path = root_dir / config["feather_file_path"]

    # Ensure output directories exist
    bandpower_feather_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: Reading MAT file and converting to DataFrame...")
    print(f"Reading from: {mat_path}")

    # Use a fixed absolute path for the torcheeg cache to prevent creating random new folders
    io_cache_path = root_dir / ".torcheeg" / "datasets_cache"
    df_raw = convert_mat_to_df(str(mat_path), str(io_cache_path))

    print("\nStep 2: Processing EEG data to bandpower...")
    print(f"Writing to: {bandpower_feather_path}")

    # Option B: full spectrogram (one row per 0.125 s frame)
    df_bp_full = dreamer_to_bandpower(df_raw, aggregate=None, output_db=True)
    df_bp_full.to_feather(str(bandpower_feather_path))

    print(f"Finished writing bandpower data to {bandpower_feather_path}")


if __name__ == "__main__":
    main()
