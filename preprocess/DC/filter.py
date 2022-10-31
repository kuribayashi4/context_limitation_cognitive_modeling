import json
import pandas as pd
from typing import List, Dict

file = "data/DC/all.txt.annotation"
data: List[Dict] = json.load(open(file))
df = pd.DataFrame(data)
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
df.to_csv(file + ".filtered.csv", sep="\t", quoting=2, escapechar="\\")
