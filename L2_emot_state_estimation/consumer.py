"""
L1 Consumer: reads from L1 output queue or L1 CSV

Provides interfaces to consume L1 output in two modes:
1. Queue mode: real-time connection to L1 output queue
2. CSV mode: batch processing of finalized L1 output
"""

import queue
import pandas as pd
from typing import List, Dict, Optional, Generator
import time


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
        self.pow_columns = [
            "AF3/theta", "AF3/alpha", "AF3/betaL", "AF3/betaH", "AF3/gamma",
            "F7/theta",  "F7/alpha",  "F7/betaL",  "F7/betaH",  "F7/gamma",
            "F3/theta",  "F3/alpha",  "F3/betaL",  "F3/betaH",  "F3/gamma",
            "FC5/theta", "FC5/alpha", "FC5/betaL", "FC5/betaH", "FC5/gamma",
            "T7/theta",  "T7/alpha",  "T7/betaL",  "T7/betaH",  "T7/gamma",
            "P7/theta",  "P7/alpha",  "P7/betaL",  "P7/betaH",  "P7/gamma",
            "O1/theta",  "O1/alpha",  "O1/betaL",  "O1/betaH",  "O1/gamma",
            "O2/theta",  "O2/alpha",  "O2/betaL",  "O2/betaH",  "O2/gamma",
            "P8/theta",  "P8/alpha",  "P8/betaL",  "P8/betaH",  "P8/gamma",
            "T8/theta",  "T8/alpha",  "T8/betaL",  "T8/betaH",  "T8/gamma",
            "FC6/theta", "FC6/alpha", "FC6/betaL", "FC6/betaH", "FC6/gamma",
            "F4/theta",  "F4/alpha",  "F4/betaL",  "F4/betaH",  "F4/gamma",
            "F8/theta",  "F8/alpha",  "F8/betaL",  "F8/betaH",  "F8/gamma",
            "AF4/theta", "AF4/alpha", "AF4/betaL", "AF4/betaH", "AF4/gamma",
        ]
        self.load()
    
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
            rows.append({
                "pow_vector": row[self.pow_columns].tolist(),
                "valence": row["valence"],
                "arousal": row["arousal"],
                "emot_state": row["emot_state"],
                "smoothed": row["smoothed"],
                "timestamp": row["timestamp"],
            })
        return rows
    
    def stream(self) -> Generator[List[float], None, None]:
        """
        Generator: yield power vectors one by one.
        
        Yields:
            Power vector lists
        """
        for vec in self.get_power_vectors():
            yield vec
