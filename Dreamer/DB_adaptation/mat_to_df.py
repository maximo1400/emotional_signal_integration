import scipy.io as scio
import pandas as pd
import numpy as np

# fmt: off
DREAMER_CHANNEL_LIST = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# fmt: on


def convert_mat_to_df(mat_path: str) -> pd.DataFrame:
    """
    Reads DREAMER.mat and converts it to a DataFrame.
    """
    mat_data = scio.loadmat(mat_path, verify_compressed_data_integrity=False)

    subject_len = len(mat_data["DREAMER"][0, 0]["Data"][0])
    trial_len = len(
        mat_data["DREAMER"][0, 0]["Data"][0, 0]["EEG"][0, 0]["stimuli"][0, 0]
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

    for subject in range(subject_len):
        for trial_id in range(trial_len):
            valence = mat_data["DREAMER"][0, 0]["Data"][0, subject]["ScoreValence"][
                0, 0
            ][trial_id, 0]
            arousal = mat_data["DREAMER"][0, 0]["Data"][0, subject]["ScoreArousal"][
                0, 0
            ][trial_id, 0]
            dominance = mat_data["DREAMER"][0, 0]["Data"][0, subject]["ScoreDominance"][
                0, 0
            ][trial_id, 0]

            # EEG data shape is (time_steps, 14)
            trial_samples = mat_data["DREAMER"][0, 0]["Data"][0, subject]["EEG"][0, 0][
                "stimuli"
            ][0, 0][trial_id, 0]
            trial_samples = trial_samples[:, :14]

            time_steps = trial_samples.shape[0]

            # Create an array for the metadata for this trial
            meta_cols = np.empty((time_steps, 10), dtype=object)
            meta_cols[:, 0] = 0  # start_at
            meta_cols[:, 1] = time_steps  # end_at
            meta_cols[:, 2] = f"{subject}_{trial_id}"  # clip_id
            meta_cols[:, 3] = subject  # subject_id
            meta_cols[:, 4] = trial_id  # trial_id
            meta_cols[:, 5] = valence
            meta_cols[:, 6] = arousal
            meta_cols[:, 7] = dominance
            meta_cols[:, 8] = f"{subject}_{trial_id}_base"  # baseline_id
            meta_cols[:, 9] = f"{subject}_{trial_id}_rec"  # _record_id

            # Append to rows
            trial_df_data = np.hstack((trial_samples, meta_cols))
            rows.append(trial_df_data)

    # Concatenate all rows
    all_data = np.vstack(rows)
    df = pd.DataFrame(all_data, columns=df_columns)

    # Ensure proper types
    for col in DREAMER_CHANNEL_LIST:
        df[col] = df[col].astype(float)

    df["start_at"] = df["start_at"].astype(int)
    df["end_at"] = df["end_at"].astype(int)
    df["subject_id"] = df["subject_id"].astype(int)
    df["trial_id"] = df["trial_id"].astype(int)
    df["valence"] = df["valence"].astype(float)
    df["arousal"] = df["arousal"].astype(float)
    df["dominance"] = df["dominance"].astype(float)

    return df
