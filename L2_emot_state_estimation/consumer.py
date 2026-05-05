"""
L1 Consumer: reads from L1 output queue or L1 CSV

Provides interfaces to consume L1 output in two modes:
1. Queue mode: real-time connection to L1 output queue
2. CSV mode: batch processing of finalized L1 output
"""

from pathlib import Path
import queue
import pandas as pd
from typing import List, Dict, Optional, Generator
import time
import pyarrow.feather as feather
import yaml

YAML_PATH = "config.yml"


def load_yml_config(path: str | Path = YAML_PATH) -> dict:
    """Parse a YAML file and return the Python object it represents."""
    path = Path(path).expanduser()
    dict = {}
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dict["feather_path"] = yml_data["feather_file_path"]
    dict["POW_COLUMNS"] = yml_data["POW_COLUMNS"]

    return dict


class FeartherConsumer:
    """Read from Feather file (DREAMER)."""

    def __init__(self, feather_path: str):
        """
        Initialize Feather consumer.

        Args:
            feather_path: Path to L1 output Feather file
        """
        self.feather_path = feather_path
        self.df = None
        self.load()

    def load(self):
        """Load Feather file into DataFrame."""
        self.df = feather.read_feather(self.feather_path)
        print(f"Loaded {len(self.df)} rows from {self.feather_path}")
        print(f"Columns: {self.df.columns.tolist()}")

    def get_rows_with_metadata(self) -> List[Dict]:
        """
        Get rows with power vectors and all metadata.

        Returns:
            List of dicts with keys: pow_vector, valence, arousal, emot_state, timestamp
        """
        rows = []
        for idx, row in self.df.iterrows():
            rows.append(
                {
                    "pow_vector": row[
                        "pow_vector"
                    ],  # Adjust based on actual column name
                    "valence": row["valence"],
                    "arousal": row["arousal"],
                    "emot_state": row["emot_state"],
                    "timestamp": row["timestamp"],
                }
            )
        return rows

    def stream(self) -> Generator[List[float], None, None]:
        """
        Generator: yield power vectors one by one.

        Yields:
            Power vector lists
        """
        for idx, row in self.df.iterrows():
            yield row["pow_vector"]  # Adjust based on actual column name


class QueueConsumer:
    """Read from L1 output queue (real-time mode)."""

    def __init__(self, l1_queue: queue.Queue, timeout: float = 1.0):
        """
        Initialize queue consumer.

        Args:
            l1_queue: The queue.Queue from L1 (EmotionSimulator)
            timeout: Seconds to wait for data before returning None
        """
        self.l1_queue = l1_queue
        self.timeout = timeout

    def get_next(self) -> Optional[List[float]]:
        """
        Get next power vector from queue.

        Returns:
            List of power floats, or None if timeout
        """
        try:
            pow_vector = self.l1_queue.get(timeout=self.timeout)
            return pow_vector
        except queue.Empty:
            return None

    def stream(self) -> Generator[List[float], None, None]:
        """
        Generator: yield power vectors indefinitely from queue.

        Yields:
            Power vector lists until queue is empty or interrupted
        """
        while True:
            pow_vec = self.get_next()
            if pow_vec is not None:
                yield pow_vec
            else:
                # Queue empty, wait a bit and retry
                time.sleep(0.1)


class CSVConsumer:
    """Read from L1 output CSV (batch mode)."""

    def __init__(self, csv_path: str):
        """
        Initialize CSV consumer.

        Args:
            csv_path: Path to L1 output CSV (output_df saved as CSV)
        """
        self.csv_path = csv_path
        self.df = None
        self.load()
        config = load_yml_config()
        self.pow_columns = config["POW_COLUMNS"]

    def load(self):
        """Load CSV into DataFrame."""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} rows from {self.csv_path}")
        print(f"Columns: {self.df.columns.tolist()}")

    def get_power_vectors(self) -> List[List[float]]:
        """
        Get all power vectors from CSV.

        Returns:
            List of power vectors (rows × 70 features)
        """
        return self.df[self.pow_columns].values.tolist()

    def get_rows_with_metadata(self) -> List[Dict]:
        """
        Get rows with power vectors and all metadata.

        Returns:
            List of dicts with keys: pow_vector, valence, arousal, emot_state, smoothed, timestamp
        """
        rows = []
        for idx, row in self.df.iterrows():
            rows.append(
                {
                    "pow_vector": row[self.pow_columns].tolist(),
                    "valence": row["valence"],
                    "arousal": row["arousal"],
                    "emot_state": row["emot_state"],
                    "smoothed": row["smoothed"],
                    "timestamp": row["timestamp"],
                }
            )
        return rows

    def stream(self) -> Generator[List[float], None, None]:
        """
        Generator: yield power vectors one by one.

        Yields:
            Power vector lists
        """
        for vec in self.get_power_vectors():
            yield vec
