import numpy as np
import pandas as pd
from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DREAMER_CHANNEL_LIST


def convert_mat_to_df(mat_path: str, io_path: str) -> pd.DataFrame:
    dataset = DREAMERDataset(
        mat_path=mat_path,
        online_transform=transforms.To2d(),
        label_transform=None,
        io_path=io_path,
        num_worker=4,
    )

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

    rows = []
    for idx in range(len(dataset)):
        eeg, dic = dataset[idx]
        eeg = np.array(eeg)
        eeg = eeg[0]
        eeg = eeg.T

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
    return df


# if __name__ == "__main__":
#     df = convert_mat_to_df("emotion_data/Dreamer/DREAMER.mat")
#     print(df.head())
