"""
Priority Model Training Script

Trains and serializes a Logistic Regression model for email priority classification
(high/medium/low) based on manually labeled examples in the Enron dataset.

Key Functionality:
- Loads and preprocesses labeled email data
- Extracts text features using TF-IDF vectorization
- Trains and evaluates Logistic Regression classifier
- Saves model artifacts for deployment

Requirements:
- Pre-labeled 'priority_label' column in the dataset
- Balanced samples (recommended: 100 high, 100 low priority examples)
"""

from pathlib import Path
import pandas as pd
import joblib
from classification import vectorize, train_lr
from preprocessing import preprocess_text

# Configure project paths
root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
models_dir = root / "models"
models_dir.mkdir(exist_ok=True)  # Ensure output directory exists

# Load and validate dataset
df = pd.read_csv(data_dir / "enron_balanced.csv")
assert "priority_label" in df.columns, (
    "Dataset must contain manually labeled 'priority_label' column "
    "(expected values: high/medium/low)"
)

# Text preprocessing pipeline
df["clean_body"] = df["body"].apply(preprocess_text)
df["combined"] = (
    df["subject"].str.lower().fillna("") + " " + df["clean_body"]
)

# Feature engineering and model training
vec, X = vectorize(df, "combined", max_feats=30_000)
clf = train_lr(X, df["priority_label"], label_name="Priority")

# Serialize trained model artifacts
model_path = models_dir / "priority_lr.pkl"
joblib.dump({"vec": vec, "clf": clf}, model_path)
print(f"Successfully saved priority classification model → {model_path}")