import matplotlib; matplotlib.use("Agg")
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import numpy as np

root  = Path(__file__).resolve().parents[1]
plots = root / "plots"; plots.mkdir(exist_ok=True)
df    = pd.read_csv(root / "data" / "enron_balanced.csv")

# ── category counts ───────────────────────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x="category_label", data=df, palette="pastel", legend=False)
plt.title("Email category counts (balanced sample)")
plt.ylabel("Count"); plt.xlabel("")
plt.tight_layout()
plt.savefig(plots / "cat_counts.png", dpi=140); plt.close()

# ── body length hist – filter, not clip ──────────────────────
df["body_len"] = df["body"].str.len()
short = df[df["body_len"] < 5_000]["body_len"]
plt.figure(figsize=(6,4))
sns.histplot(short, bins=60, color="#69b3a2")
plt.title("Body length distribution (< 5 000 chars)")
plt.xlabel("Characters"); plt.ylabel("Emails")
plt.tight_layout()
plt.savefig(plots / "body_len_hist.png", dpi=140); plt.close()

print("Charts saved to", plots)
