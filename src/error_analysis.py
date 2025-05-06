from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from preprocessing  import preprocess_text
from classification import vectorize
import numpy as np

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
models_dir = root / "models"

df  = pd.read_csv(data_dir / "enron_balanced.csv")
df["clean_body"] = df["body"].apply(preprocess_text)
df["combined"]   = df["subject"].str.lower().fillna("") + " " + df["clean_body"]

vec, X = vectorize(df, "combined", 30_000)
clf    = joblib.load(models_dir / "category_lr.pkl")["clf"]

scores = cross_val_score(clf, X, df["category_label"], cv=5, scoring="f1_macro", n_jobs=-1)
print("5‑fold Macro F1:", scores.round(3), "→ mean", scores.mean().round(3))

preds = clf.predict(X)
df["preds"] = preds

wrong = df[df["preds"] != df["category_label"]].head(10)

cols = ["sender", "subject", "category_label", "preds"]
wrong[cols].to_csv(data_dir / "misclassified_sample.csv", index=False)
print("Saved 10 mis‑classified examples →",
      data_dir / "misclassified_sample.csv")
