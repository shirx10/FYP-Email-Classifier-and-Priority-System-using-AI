"""
Run:
    python -m src.latency_test --n 1000
"""
import argparse, time, statistics, json, random, os, psutil, joblib, pandas as pd
from pathlib import Path
from src.preprocessing import preprocess_text

# ───── paths ─────────────────────────────────────────────────────────────────
root        = Path(__file__).resolve().parents[1]
models_dir  = root / "models"
data_dir    = root / "data"
plots_dir   = root / "plots"; plots_dir.mkdir(exist_ok=True)

# ───── load models ─────────────────────────────
cat_art   = joblib.load(models_dir / "category_lr.pkl")   # dict: {"vec", "clf"}
cat_vec, clf_cat = cat_art["vec"], cat_art["clf"]

prio_art  = joblib.load(models_dir / "priority_lr.pkl")   # dict: {"vec", "clf"}
prio_vec, clf_prio = prio_art["vec"], prio_art["clf"]

# ───── sample set ────────────────────────────────────────────────────────────
df      = pd.read_csv(data_dir / "enron_balanced.csv").sample(frac=1.0, random_state=1)
emails  = df[["subject", "sender", "body"]].head(10_000).to_dict("records")

# ───── classify helper ───────────────────────────────────────────────────────
def classify(rec: dict) -> tuple[str, str]:
    subj, body = rec["subject"], rec["body"]
    clean      = preprocess_text(body)
    text       = f"{subj} {clean}"

    cat = clf_cat.predict(cat_vec.transform([text]))[0]

    prio_text = f"{rec['sender']} {subj} {clean}"
    prio = clf_prio.predict(prio_vec.transform([prio_text]))[0]

    return cat, prio

# ───── main loop ─────────────────────────────────────────────────────────────
def main(n: int):
    lat  = []
    proc = psutil.Process(os.getpid())

    for _ in range(n):
        rec  = random.choice(emails)
        t0   = time.perf_counter()
        _    = classify(rec)
        lat.append((time.perf_counter() - t0) * 1_000)   # ms

    p95 = statistics.quantiles(lat, n=20)[18]
    print(f"\nLatency: mean {statistics.mean(lat):.1f} ms, "
          f"p95 {p95:.1f} ms, max {max(lat):.1f} ms")
    print(f"Peak RSS: {proc.memory_info().rss/1_048_576:.0f} MB")

    json.dump(lat, open(plots_dir / "latency_ms.json", "w"), indent=0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000,
                    help="number of random e‑mails to time")
    args = ap.parse_args()
    main(args.n)
