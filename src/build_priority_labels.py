# ── src/build_priority_labels.py ─────────────────────────────
"""
Creates data/priority_train.csv with 400 rows / priority class
(high / medium / low).  Every cell – header included – is quoted.
"""

from pathlib import Path
import re, csv, pandas as pd

root = Path(__file__).resolve().parents[1]
src  = root / "data" / "enron_labeled.csv"
out  = root / "data" / "priority_train.csv"

df = pd.read_csv(src)

# ── heuristic tags ───────────────────────────────────────────
df["priority_label"] = "medium"
df.loc[df["subject"].str.contains(r"\b(?:urgent|asap|deadline)\b",
                                  case=False, na=False, regex=True),
       "priority_label"] = "high"
df.loc[df["sender"].str.contains(r"(?:friend|mom|dad|@yahoo\.|@gmail\.)",
                                  case=False, na=False, regex=True),
       "priority_label"] = "low"

# ── balance: exactly 400 per class ───────────────────────────
balanced = (df.groupby("priority_label", group_keys=False)
              .apply(lambda g: g.sample(400, replace=True, random_state=1))
              .reset_index(drop=True))

# ── write ─────────────────────────────
balanced.to_csv(
    out,
    index=False,
    quoting=csv.QUOTE_ALL,
    lineterminator="\n"
)
print("✅  Saved balanced priority set →", out)
