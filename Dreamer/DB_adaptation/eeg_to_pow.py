import numpy as np
import pandas as pd
from scipy.signal import lfilter
from typing import Optional

# fmt: off
# Emotiv EPOC+ channel order
EEG_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

BANDS = {
    # "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "betaL": (12, 16),
    "betaH": (16, 25),
    "gamma": (25, 45),
}

# Emotiv 0.5 Hz high-pass IIR coefficients (2nd order Butterworth)
HP_B = np.array([0.96588528974407006000, -1.93177057948814010000, 0.96588528974407006000])
HP_A = np.array([1.00000000000000000000, -1.93060642721966810000, 0.93293473175661223000])
# fmt: on


def slew_clip(data: np.ndarray, max_delta: float = 30.0) -> np.ndarray:
    """Clip sample-to-sample jumps to ±max_delta µV."""
    diff = np.diff(data, axis=0, prepend=data[:1])
    diff = np.clip(diff, -max_delta, max_delta)
    return np.cumsum(diff, axis=0)


def hann_window_emotiv(n: int, scale: float = 1.0) -> np.ndarray:
    """
    Exact replica of Emotiv's loop-based Hann window.

    Formula: w[i] = 0.5 * (1 - cos(2π * (i+1) / (n+1)))   i = 0 … n-1

    Parameters
    ----------
    n : int
        Window length.
    scale : float
        Amplitude multiplier (Emotiv uses 2.0 in some pipelines).

    Returns
    -------
    w : ndarray, shape (n,), dtype float32
    """
    idx = np.arange(1, n + 1, dtype=np.float32)  # 1 … n
    w = 0.5 * (1 - np.cos(2.0 * np.pi * idx / (n + 1)))
    return (scale * w).astype(np.float32)


def emotiv_bandpower(
    data: np.ndarray,
    fs: int = 128,
    win_size: int = 256,
    hop: int = 16,
    bands: dict = BANDS,
    output_db: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Emotiv EPOC+ band power replication.

    Parameters
    ----------
    data : np.ndarray, shape (samples, channels)
        Raw EEG in µV, 128 Hz.
    fs : int
        Sampling frequency.
    win_size : int
        FFT window length in samples (256 = 2 s).
    hop : int
        Hop length in samples (16 = 0.125 s).
    bands : dict
        {name: (f_low, f_high)} frequency bands.
    output_db : bool
        If True, return 10*log10(power). Otherwise raw power.

    Returns
    -------
    bp : np.ndarray, shape (n_frames, n_channels, n_bands)
    t : np.ndarray, shape (n_frames,)
        Center time of each frame in seconds.
    band_names : list[str]
    """
    n_samples, n_ch = data.shape

    # 1) Slew-rate clip
    data = slew_clip(data, max_delta=30.0)

    # 2) High-pass filter 0.5 Hz
    data = lfilter(HP_B, HP_A, data, axis=0)

    # 3) Epoch into overlapping frames
    n_frames = (n_samples - win_size) // hop + 1
    if n_frames < 1:
        raise ValueError(f"Signal too short: {n_samples} samples < win_size {win_size}")

    frames = np.empty((n_frames, win_size, n_ch), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frames[i] = data[start : start + win_size]

    # 4) DC removal per epoch
    q25, q75 = np.percentile(frames, [25, 75], axis=1, keepdims=True)
    mask = (frames >= q25) & (frames <= q75)  # (n_frames, win_size, n_ch)
    sums = (frames * mask).sum(
        axis=1, keepdims=True
    )  # sum inside IQR per frame/channel
    counts = mask.sum(axis=1, keepdims=True).astype(np.float32)
    fallback = frames.mean(axis=1, keepdims=True)
    iqm_ref = np.where(counts == 0, fallback, sums / counts)
    frames -= iqm_ref
    # Alternative: mean removal
    # frames -= frames.mean(axis=1, keepdims=True)

    # 5) Hann window (×2 scaling as per Emotiv)
    # hann = (2 * np.hanning(win_size)).astype(np.float32) # NumPy version
    hann = hann_window_emotiv(win_size, scale=2.0)  # Emotiv exact version

    frames *= hann[None, :, None]

    # 6) FFT → power spectrum
    fft = np.fft.rfft(frames, n=win_size, axis=1) / win_size
    power = np.abs(fft) ** 2  # (n_frames, n_bins, n_ch)

    # 7) Band aggregation
    freq = np.fft.rfftfreq(win_size, 1 / fs)
    band_names = list(bands.keys())
    bp = np.empty((n_frames, n_ch, len(bands)), dtype=np.float32)

    for idx, (name, (f_lo, f_hi)) in enumerate(bands.items()):
        mask = (freq >= f_lo) & (freq < f_hi)
        bp[:, :, idx] = power[:, mask, :].sum(axis=1)

    if output_db:
        bp = 10 * np.log10(bp + 1e-20)

    # Time axis (center of each window)
    t = (np.arange(n_frames) * hop + win_size // 2) / fs

    return bp, t, band_names


def dreamer_to_bandpower(
    df: pd.DataFrame,
    fs: int = 128,
    win_size: int = 256,
    hop: int = 16,
    output_db: bool = True,
    aggregate: Optional[str] = "mean",
) -> pd.DataFrame:
    """
    Process DREAMER feather DataFrame → band power table.

    Parameters
    ----------
    df : pd.DataFrame
        DREAMER dataset with EEG + metadata columns.
    fs : int
        Sampling rate (128 Hz for DREAMER/Emotiv).
    win_size : int
        Window size in samples.
    hop : int
        Hop in samples.
    output_db : bool
        Return dB or raw power.
    aggregate : str or None
        'mean' → one row per trial (mean across frames).
        'median' → one row per trial (median across frames).
        None → one row per frame (exploded).

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, trial_id, clip_id, valence, arousal, dominance,
                 baseline_id, {channel}/{band} for all 14×5 combinations,
                 (plus frame_time if aggregate is None).
    """
    # Validate channels
    missing = set(EEG_CHANNELS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing EEG channels: {missing}")

    # Group by unique trial
    group_cols = ["subject_id", "trial_id"]
    meta_cols = ["clip_id", "valence", "arousal", "dominance", "baseline_id"]

    records = []

    for (subj, trial), grp in df.groupby(group_cols, sort=False):
        # Extract EEG matrix (samples × 14 channels)
        eeg = grp[EEG_CHANNELS].values.astype(np.float32)

        # Skip if too short
        if eeg.shape[0] < win_size:
            print(f"Skipping subject {subj} trial {trial}: too short")
            continue

        # Compute band power
        bp, t, band_names = emotiv_bandpower(
            eeg, fs=fs, win_size=win_size, hop=hop, output_db=output_db
        )
        # bp shape: (n_frames, 14 channels, 5 bands)

        # Grab metadata from first row of group
        meta = grp[meta_cols].iloc[0].to_dict()

        if aggregate is None:
            # One row per frame
            for f_idx in range(bp.shape[0]):
                row = {"subject_id": subj, "trial_id": trial, **meta}
                row["frame_time"] = t[f_idx]
                for ch_idx, ch in enumerate(EEG_CHANNELS):
                    for b_idx, band in enumerate(band_names):
                        row[f"{ch}/{band}"] = bp[f_idx, ch_idx, b_idx]
                records.append(row)
        else:
            # Aggregate across frames
            if aggregate == "mean":
                bp_agg = bp.mean(axis=0)
            elif aggregate == "median":
                bp_agg = np.median(bp, axis=0)
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")

            row = {"subject_id": subj, "trial_id": trial, **meta}
            for ch_idx, ch in enumerate(EEG_CHANNELS):
                for b_idx, band in enumerate(band_names):
                    row[f"{ch}/{band}"] = bp_agg[ch_idx, b_idx]
            records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load DREAMER feather
    df_raw = pd.read_feather("emotion_data/Dreamer/dreamer_eeg.feather")

    # # Option A: one row per trial (mean band power)
    # df_bp = dreamer_to_bandpower(df_raw, aggregate="mean", output_db=True)
    # print(df_bp.head())
    # df_bp.to_feather("emotion_data/Dreamer/dreamer_bandpower_trial.feather")

    # Option B: full spectrogram (one row per 0.125 s frame)
    df_bp_full = dreamer_to_bandpower(df_raw, aggregate=None, output_db=True)
    df_bp_full.to_feather("emotion_data/Dreamer/dreamer_bandpower_frames.feather")
