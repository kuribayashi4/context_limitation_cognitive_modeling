import pandas as pd

df = pd.read_table("data/BE/fpt-log.csv.annotation.filtered.csv")
df = df[df["time"] > 0]
df = df[df["has_num"] == False]
df = df[df["is_first"] == False]
assert len(df) == 9217
df.to_csv(
    "data/BE/fpt-log.csv.annotation.filtered.csv.data4modeling",
    sep="\t",
    quoting=2,
    escapechar="\\",
)
