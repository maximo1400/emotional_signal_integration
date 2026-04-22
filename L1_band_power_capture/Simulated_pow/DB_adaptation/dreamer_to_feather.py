from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DREAMER_CHANNEL_LIST
import numpy as np
import pandas as pd
import pyarrow.feather as feather

dataset = DREAMERDataset(
    mat_path="emotion_data/Dreamer/DREAMER.mat",
    online_transform=transforms.To2d(),
    label_transform=None,
    io_path=".torcheeg/datasets_1768410175800_ZxOwu",
    num_worker=4,
)
# print(dataset[1])
df_columns = DREAMER_CHANNEL_LIST.copy()
df_columns += [
    "start_at",
    "end_at",
    "clip_id",
    "subject_id",
    "trial_id",
    "valence",
    "arousal",
    "dominance",
    "baseline_id",
    "_record_id",
]

# df = pd.DataFrame(columns=df_columns)

rows = []
for idx in range(len(dataset)):
    eeg, dic = dataset[
        idx
    ]  # adjust if your print(dataset[0]) shows a different structure
    # eeg: (n_channels, n_points)
    eeg = np.array(eeg)
    eeg = eeg[0]
    eeg = eeg.T
    # flat = eeg.flatten()
    star_at = dic["start_at"]
    end_at = dic["end_at"]
    clip_idx = dic["clip_id"]
    subject_id = dic["subject_id"]
    trial_id = dic["trial_id"]
    valence = dic["valence"]
    arousal = dic["arousal"]
    dominance = dic["dominance"]
    baseline_id = dic["baseline_id"]
    _record_id = dic["_record_id"]
    # df_row = eeg.tolist()
    for row in eeg:
        df_row = row.tolist()
        df_row += [
            star_at,
            end_at,
            clip_idx,
            subject_id,
            trial_id,
            valence,
            arousal,
            dominance,
            baseline_id,
            _record_id,
        ]
        rows.append(df_row)

df = pd.DataFrame(rows, columns=df_columns)
feather.write_feather(df, "emotion_data/Dreamer/dreamer_eeg.feather")


print("Finished writing DREAMER EEG data to dreamer_eeg.feather")
