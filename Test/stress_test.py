"""
Run:
    python -m src.stress_test --messages 10000 --batch 256
"""
from __future__ import annotations
import argparse, time, concurrent.futures, psutil, os, json
from pathlib import Path
import pandas as pd, joblib
from src.preprocessing import preprocess_text

# ───── paths ────────────────────────────────────────────────────────────────
root        = Path(__file__).resolve().parents[1]
models_dir  = root / "models"
data_dir    = root / "data"
plots_dir   = root / "plots"; plots_dir.mkdir(exist_ok=True)

# ───── load models ───────────────────────────
cat_art               = joblib.load(models_dir / "category_lr.pkl")
cat_vec, clf_cat      = cat_art["vec"], cat_art["clf"]

prio_art              = joblib.load(models_dir / "priority_lr.pkl")
prio_vec, clf_prio    = prio_art["vec"], prio_art["clf"]

# ───── preload & preprocess data once to avoid I/O in timing window ─────────
df = pd.read_csv(data_dir / "enron_balanced.csv")

# Build lists so index‑based slicing is cheap
subj_body = (
    df["subject"].str.lower().fillna("") + " " +
    df["body"].apply(preprocess_text)
).tolist()

senders = df["sender"].fillna("").tolist()

TOTAL_AVAILABLE = len(subj_body)

# ───── batch‑inference helper ───────────────────────────────────────────────
def batch_predict(indices: list[int]) -> None:
    """Run both classifiers on a slice of message indices."""
    sb_batch   = [subj_body[i] for i in indices]
    prio_batch = [f"{senders[i]} {subj_body[i]}" for i in indices]

    _ = clf_cat.predict(cat_vec.transform(sb_batch))
    _ = clf_prio.predict(prio_vec.transform(prio_batch))

# ───── driver ───────────────────────────────────────────────────────────────
def main(total_msgs: int, batch: int):
    if total_msgs > TOTAL_AVAILABLE:
        raise ValueError(f"--messages {total_msgs} exceeds corpus size "
                         f"({TOTAL_AVAILABLE}).  Reduce the flag or "
                         f"augment the dataset.")

    indices = list(range(total_msgs))
    chunks  = [indices[i:i + batch] for i in range(0, total_msgs, batch)]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        _ = list(ex.map(batch_predict, chunks))
    duration = time.perf_counter() - t0

    throughput = total_msgs / duration
    rss_mb     = psutil.Process(os.getpid()).memory_info().rss / 1_048_576

    print(f"Processed {total_msgs:,} msgs in {duration:.2f} s "
          f"→ {throughput:.0f} msg s⁻¹  (peak RSS {rss_mb:.0f} MB)")

    json.dump(
        {"messages": total_msgs,
         "seconds":  duration,
         "throughput": throughput,
         "rss_mb":     rss_mb},
        open(plots_dir / "stress_summary.json", "w"),
        indent=2,
    )

# ───── CLI ‑‑----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", type=int, default=9000,
                    help="total messages to benchmark (≤ corpus size)")
    ap.add_argument("--batch",    type=int, default=256,
                    help="messages per thread batch")
    args = ap.parse_args()
    main(args.messages, args.batch)
