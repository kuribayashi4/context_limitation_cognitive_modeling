import pandas as pd

df = pd.read_table("data/DC/all.txt.annotation.filtered.csv")
df = df[df["time"] > 0]
df = df[df["has_num"] == False]
df = df[df["has_num_prev_1"] == False]
df = df[df["has_punct"] == False]
df = df[df["has_punct_prev_1"] == False]
df = df[df["is_first"] == False]
df = df[df["is_last"] == False]
df = df[df["pos"] != "CD"]
assert len(df) == 212649
df.to_csv(
    "data/DC/all.txt.annotation.filtered.csv.data4modeling",
    sep="\t",
    quoting=2,
    escapechar="\\",
)
