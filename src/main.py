# ──────────────────────────────────────────────────────────────
# src/main.py - Model Training Pipeline
#
# This script handles:
# 1. Training the primary TF-IDF + Logistic Regression model for email categorization
# 2. Optional comparative evaluation using MiniLM sentence embeddings
#
# Key Functionality:
# - Data loading and preprocessing
# - Feature vectorization (TF-IDF)
# - Logistic Regression model training
# - Model serialization for deployment
# - Optional transformer-based baseline comparison
# ──────────────────────────────────────────────────────────────

from pathlib import Path
import argparse, sys, joblib, pandas as pd
from preprocessing import preprocess_text
from classification import vectorize, train_lr
from priority import priority_rule

# Configure project paths
root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
models_dir = root / "models"
models_dir.mkdir(exist_ok=True)  # Ensure models directory exists


def run_pipeline(run_minilm: bool = False) -> None:
    """
    Execute the complete model training pipeline.

    Args:
        run_minilm: If True, runs additional MiniLM baseline comparison

    Workflow:
        1. Load and preprocess Enron dataset
        2. Train TF-IDF + Logistic Regression classifier
        3. Save trained model artifacts
        4. Optionally run MiniLM baseline experiment
    """
    # Configuration flags
    USE_ENRON = True  # Always use Enron dataset in current implementation
    BALANCED = True  # Use balanced dataset version

    # Load dataset
    fname = "enron_balanced.csv" if BALANCED else "enron_labeled.csv"
    df = pd.read_csv(data_dir / fname)
    print(f"Loaded {len(df):,} rows from {fname}")

    # ── Data Preprocessing ────────────────────────────────────
    # Clean email bodies and combine with subject lines
    df["clean_body"] = df["body"].apply(preprocess_text)
    df["combined"] = df["subject"].str.lower().fillna("") + " " + df["clean_body"]

    # ── Primary Model Training (TF-IDF + Logistic Regression) ──
    # Vectorize text and train classifier
    vec, X = vectorize(df, "combined", max_feats=30_000)
    clf = train_lr(X, df["category_label"])

    # Save model artifacts
    joblib.dump({"vec": vec, "clf": clf}, models_dir / "category_lr.pkl")
    print("Category model saved →", models_dir / "category_lr.pkl")

    # Generate rule-based priority labels (for debugging/analysis)
    df["priority_rule"] = df.apply(
        lambda r: priority_rule(r["subject"], r["sender"]), axis=1
    )

    # ── Optional MiniLM Baseline Experiment ────────────────────
    if run_minilm:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score

            print("\n🟢 Running MiniLM baseline comparison on 5,000 samples...")

            # Create reproducible subsample
            samp = df.sample(n=5_000, random_state=1).reset_index(drop=True)
            sentences = samp["combined"].tolist()  # Convert to plain Python list

            # Initialize and run MiniLM encoder
            model_st = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model_st.encode(sentences, show_progress_bar=True)

            # Train/test split and model training
            y_st = samp["category_label"]
            Xtr, Xte, ytr, yte = train_test_split(
                emb, y_st, test_size=0.2, random_state=42, stratify=y_st
            )

            clf_st = LogisticRegression(max_iter=1_000).fit(Xtr, ytr)
            preds = clf_st.predict(Xte)

            # Output performance metrics
            acc = round(accuracy_score(yte, preds), 3)
            print("MiniLM Accuracy:", acc)
            print(classification_report(yte, preds))

        except ModuleNotFoundError:
            print("Warning: sentence-transformers not installed - skipping MiniLM experiment")


if __name__ == "__main__":
    # Configure command-line interface
    ap = argparse.ArgumentParser(
        description="Train email classification models with optional MiniLM baseline comparison"
    )
    ap.add_argument(
        "--minilm",
        action="store_true",
        help="Run additional 5,000-row MiniLM baseline experiment"
    )
    args = ap.parse_args()

    # Execute training pipeline
    run_pipeline(run_minilm=args.minilm)