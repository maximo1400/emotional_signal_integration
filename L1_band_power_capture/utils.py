import csv
from pathlib import Path


class L1OutputWriter:
    """Writer for Layer 1 (Band Power Capture) output CSV files."""

    def __init__(self, output_file: Path, save_files: bool):
        self.output_file = output_file
        self.save_files = save_files
        self.f = None
        self.writer = None

        if self.save_files:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.f = self.output_file.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.f)

    def write_headers(self, feature_names: list[str]):
        if not self.save_files or self.writer is None:
            return
        headers = feature_names + [
            "valence",
            "arousal",
            "emot_state",
            "smoothed",
            "timestamp",
            "data_origin",
        ]
        self.writer.writerow(headers)

    def write_row(
        self,
        pow_vector: list[float],
        valence: float,
        arousal: float,
        emot_state: str,
        smoothed: bool,
        timestamp: float,
        data_origin: str,
    ):
        if not self.save_files or self.writer is None or self.f is None:
            return
        row = [
            *pow_vector,
            valence,
            arousal,
            emot_state,
            smoothed,
            timestamp,
            data_origin,
        ]
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            self.writer = None
