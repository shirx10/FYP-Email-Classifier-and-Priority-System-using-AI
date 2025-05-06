# ── src/train_priority_model.py ─────────────────────────────
from pathlib import Path
import subprocess, sys, pandas as pd, joblib
from preprocessing  import preprocess_text
from classification import vectorize, train_lr

root       = Path(__file__).resolve().parents[1]
data_dir   = root / "data"
models_dir = root / "models"; models_dir.mkdir(exist_ok=True)
prio_csv   = data_dir / "priority_train.csv"
builder_py = root / "src" / "build_priority_labels.py"

def ensure_csv() -> None:
    """Create priority_train.csv if absent."""
    if not prio_csv.exists():
        print("priority_train.csv missing – building …")
        if subprocess.call([sys.executable, str(builder_py)]) != 0:
            sys.exit("build_priority_labels.py failed")

ensure_csv()

# ── load & clean header ─────────────────────────────────────
df = pd.read_csv(prio_csv, engine="python")
df.columns = (df.columns
                .str.strip(' "\'\r\n')   # remove quotes / spaces / CR
                .str.lower())

if "priority_label" not in df.columns:
    # one automatic rebuild attempt
    print("priority_label not found – rebuilding file once …")
    prio_csv.unlink(missing_ok=True)
    ensure_csv()
    df = pd.read_csv(prio_csv, engine="python")
    df.columns = df.columns.str.strip(' "\'\r\n').str.lower()
    if "priority_label" not in df.columns:
        sys.exit("priority_label column still missing! "
                 "Open priority_train.csv and inspect the header.")

# ── normal training  ───────────────────────────────────────
df["clean_body"] = df["body"].apply(preprocess_text)
df["combined"]   = df["subject"].str.lower().fillna("") + " " + df["clean_body"]

vec, X  = vectorize(df, "combined", 20_000)
clf     = train_lr(X, df["priority_label"], label_name="Priority")
joblib.dump({"vec": vec, "clf": clf}, models_dir / "priority_lr.pkl")
print("ML‑priority model saved →", models_dir / "priority_lr.pkl")
