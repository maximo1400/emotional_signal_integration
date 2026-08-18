import csv
from pathlib import Path


class L3OutputWriter:
    """Writer for Layer 3 (VA Data Adaptation) output CSV files."""

    def __init__(self, output_file: Path, save_files: bool):
        self.output_file = output_file
        self.save_files = save_files
        self.f = None
        self.writer = None

        if self.save_files:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.f = self.output_file.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.f)

    def write_headers(self):
        if not self.save_files or self.writer is None:
            return
        headers = [
            "raw_valence",
            "raw_arousal",
            "smoothed_valence",
            "smoothed_arousal",
            "confidence",
            "timestamp",
            "previous_layer_timestamp",
            "smoothing_method",
        ]
        self.writer.writerow(headers)

    def write_row(
        self,
        raw_v: float,
        raw_a: float,
        smooth_v: float,
        smooth_a: float,
        confidence: float,
        timestamp: float,
        prev_layer_timestamp: float,
        smoothing_method: str,
    ):
        if not self.save_files or self.writer is None or self.f is None:
            return
        row = [
            raw_v,
            raw_a,
            smooth_v,
            smooth_a,
            confidence,
            timestamp,
            prev_layer_timestamp,
            smoothing_method,
        ]
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            self.writer = None
