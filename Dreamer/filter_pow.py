import pandas as pd
from pathlib import Path


def main():
    "script to filter pow.feather for specific subject ids and save as pow_filtered.feather"
    peolple = [1, 2, 3, 4, 5]
    pow_file = Path("Dreamer/pow.feather")
    output_file = Path("Dreamer/pow_filtered.feather")

    df = pd.read_feather(pow_file)

    # Convert subject_id to a temporary numeric column for robust filtering
    temp_col = df["subject_id"].astype(int)

    print(f"Filtering for subject(s): {peolple}")
    filtered_df = df[temp_col.isin(peolple)].copy()

    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("Warning: No rows match the specified subject IDs.")

    print(f"Saving to {output_file}...")
    filtered_df.reset_index(drop=True).to_feather(output_file)
    print("Done!")


if __name__ == "__main__":
    main()
