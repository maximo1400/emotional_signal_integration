import numpy as np
import pandas as pd
import scipy.io as scio


def convert_mat_to_df(mat_path: str, electrodes: list[str]) -> pd.DataFrame:
    """
    Reads DREAMER.mat and converts it to a DataFrame.
    """
    mat_data = scio.loadmat(mat_path, verify_compressed_data_integrity=False)
    epoch_len = len(electrodes)

    # The MATLAB file structure is heavily nested. We extract the core 'Data' array here.
    # mat_data["DREAMER"][0, 0]["Data"][0] contains an array of structs, one for each subject.
    dreamer_data = mat_data["DREAMER"][0, 0]["Data"][0]
    subject_len = len(dreamer_data)

    df_columns = electrodes.copy()
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
        # Extract the specific subject's data struct
        subject_data = dreamer_data[subject]
        trial_len = len(subject_data["EEG"][0, 0]["stimuli"][0, 0])

        for trial_id in range(trial_len):
            # Extract scores for the current trial
            valence = subject_data["ScoreValence"][0, 0][trial_id, 0]
            arousal = subject_data["ScoreArousal"][0, 0][trial_id, 0]
            dominance = subject_data["ScoreDominance"][0, 0][trial_id, 0]

            # EEG data shape is (time_steps, 14)
            trial_samples = subject_data["EEG"][0, 0]["stimuli"][0, 0][trial_id, 0]
            trial_samples = trial_samples[:, :epoch_len]

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
    for col in electrodes:
        df[col] = df[col].astype(float)

    df["start_at"] = df["start_at"].astype(int)
    df["end_at"] = df["end_at"].astype(int)
    df["subject_id"] = df["subject_id"].astype(int)
    df["trial_id"] = df["trial_id"].astype(int)
    df["valence"] = df["valence"].astype(float)
    df["arousal"] = df["arousal"].astype(float)
    df["dominance"] = df["dominance"].astype(float)

    return df
