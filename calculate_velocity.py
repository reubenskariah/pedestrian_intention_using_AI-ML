import pandas as pd
import numpy as np

# Load trajectory data
df = pd.read_csv("trajectory.csv")

# Sort properly
df = df.sort_values(["id", "frame"])

# -------------------------
# VELOCITY
# -------------------------

df["vx"] = df.groupby("id")["nose_x"].diff()
df["vy"] = df.groupby("id")["nose_y"].diff()

df["speed"] = np.sqrt(
    df["vx"]**2 +
    df["vy"]**2
)

# -------------------------
# ACCELERATION
# -------------------------

df["ax"] = df.groupby("id")["vx"].diff()
df["ay"] = df.groupby("id")["vy"].diff()

df["acceleration"] = np.sqrt(
    df["ax"]**2 +
    df["ay"]**2
)

# -------------------------
# DIRECTION
# -------------------------

df["direction_deg"] = np.degrees(
    np.arctan2(df["vy"], df["vx"])
)

# -------------------------
# SAVE
# -------------------------

df.to_csv(
    "trajectory_motion_features.csv",
    index=False
)

print("\nMotion features saved successfully.\n")

print(
    df[
        [
            "frame",
            "id",
            "nose_x",
            "nose_y",
            "vx",
            "vy",
            "speed",
            "ax",
            "ay",
            "acceleration",
            "direction_deg"
        ]
    ].head(20)
)