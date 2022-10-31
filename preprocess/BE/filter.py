import glob
import json
import pandas as pd
from typing import List, Dict

file = "data/BE/fpt-log.csv.annotation"
ng_fields = [
    "authorsData",
    "caption",
    "listItem",
    "profile",
    "titleBlock",
]
data: List[Dict] = json.load(open(file))
df = pd.DataFrame(data)
idx_invalid_field = df["metadata"].isin(ng_fields)
times = df["time"].values
mean_time = times[times > 0].mean()
std_time = times[times > 0].std()
idx_long_time = df["time"] > mean_time + 3 * std_time
idx_short_time = df["time"] < mean_time - 3 * std_time
df.loc[idx_long_time, "time"] = 0
df.loc[idx_long_time, "logtime"] = "-Infinity"
df.loc[idx_long_time, "invtime"] = "Infinity"
df.loc[idx_short_time, "time"] = 0
df.loc[idx_short_time, "logtime"] = "-Infinity"
df.loc[idx_short_time, "invtime"] = "Infinity"
df.loc[idx_invalid_field, "time"] = 0
df.loc[idx_invalid_field, "logtime"] = "-Infinity"
df.loc[idx_invalid_field, "invtime"] = "Infinity"
df.to_csv(file + ".filtered.csv", sep="\t", quoting=2, escapechar="\\")
