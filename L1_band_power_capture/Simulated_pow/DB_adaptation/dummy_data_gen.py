import pandas as pd
import numpy as np
import os


# Read the CSV file
input = "emotion_data/virtual/pow.csv"
output = "emotion_data/virtual/dummy_pow.feather"


df = pd.read_csv(input)

# Delete img and round columns
df = df.drop(columns=["img", "round"])

# Drop the first column
df = df.iloc[:, 1:]

# Add valence and arousal columns with random float values from -1 to 1
df["valence"] = np.random.uniform(-1, 1, len(df))
df["arousal"] = np.random.uniform(-1, 1, len(df))

# print(df.head())

# Save as pyarrow format
if os.path.exists(output):
    os.remove(output)
df.reset_index(drop=True).to_feather(output)
