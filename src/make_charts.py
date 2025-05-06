# ── src/make_charts.py ────────────────────────────────────────────
"""
Rebuilds all static evaluation charts into the plots/ folder.

Creates:
  plots/cat_counts.png
  plots/body_len_hist.png
  plots/latency_hist.png
  plots/throughput_bar.png

Run head‑less; no UI.
"""
from pathlib import Path
import json
import matplotlib; matplotlib.use("Agg")           # non‑GUI backend
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd

root   = Path(__file__).resolve().parents[1]
data   = pd.read_csv(root / "data" / "enron_balanced.csv")
plots  = root / "plots";  plots.mkdir(exist_ok=True)

# ───────────────── Category counts bar chart ─────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(
    x="category_label",
    data=data,
    order=sorted(data["category_label"].unique()),
    color="#a8dadc",
    legend=False,
)
plt.title("Email category counts")
plt.ylabel("Count"); plt.xlabel("")
plt.tight_layout()
plt.savefig(plots / "cat_counts.png", dpi=140)
plt.close()

# ───────────────── Body‑length histogram ─────────────────────────
data["body_len"] = data["body"].str.len()
maxlength = 5_000
lengths   = data["body_len"].where(data["body_len"] <= maxlength).dropna()

plt.figure(figsize=(6, 4))
sns.histplot(lengths, bins=60, color="#69b3a2")
plt.title(f"Body length distribution (≤ {maxlength:,} chars)")
plt.xlabel("Characters"); plt.ylabel("Emails")
plt.tight_layout()
plt.savefig(plots / "body_len_hist.png", dpi=140)
plt.close()

latency_json = plots / "latency_ms.json"
if latency_json.exists():
    lat_ms = json.loads(latency_json.read_text())

    warm_ms = lat_ms[1:]  # skip first element


    plt.figure(figsize=(6, 4))
    plt.hist(warm_ms, bins=50)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Messages")
    plt.title(f"Latency distribution (warm‑state, n = {len(warm_ms)})")
    plt.tight_layout()
    plt.savefig(plots / "latency_hist.png", dpi=140)
    plt.close()


stress_json = plots / "stress_summary.json"
if stress_json.exists():
    summary = json.loads(stress_json.read_text())
    thr = summary["throughput"]

    plt.figure(figsize=(6, 2.5))
    plt.barh([""], [thr], color="#1f77b4")
    plt.xlabel("Messages per second")
    plt.title("Sustained throughput (batch 256)")
    plt.xlim(0, thr * 1.1)
    plt.text(thr * 0.98, 0, f"{thr:.0f}", va="center", ha="right", color="white",
             fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots / "throughput_bar.png", dpi=140)
    plt.close()
else:
    print("stress_summary.json not found – skipping throughput chart")

print("Charts rebuilt →", plots)
