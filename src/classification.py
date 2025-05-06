# classification.py - Model Training Utilities
#
# Core functionality:
# - Text vectorization using TF-IDF
# - Logistic Regression model training and evaluation
# - Performance visualization (confusion matrices)
# - Model serialization for deployment

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib  # For model serialization


def vectorize(df, text_col: str, max_feats: int = 30_000):
    """
    Convert text data into TF-IDF feature vectors.

    Args:
        df: Pandas DataFrame containing text data
        text_col: Column name containing text to vectorize
        max_feats: Maximum number of vocabulary features to retain

    Returns:
        tuple: (TfidfVectorizer instance, Sparse feature matrix)
    """
    vec = TfidfVectorizer(max_features=max_feats)
    X = vec.fit_transform(df[text_col])
    return vec, X


def train_lr(X, y, label_name="Category"):
    """
    Train and evaluate a Logistic Regression classifier with comprehensive diagnostics.

    Args:
        X: Feature matrix (sparse or dense)
        y: Target labels
        label_name: Descriptive name for the classification task (used in outputs)

    Returns:
        LogisticRegression: Trained classifier instance

    Workflow:
        1. Performs stratified 80/20 train-test split
        2. Trains Logistic Regression classifier
        3. Generates performance metrics (accuracy, classification report)
        4. Creates and saves confusion matrix visualization
        5. Serializes model artifacts
    """
    # Create stratified split for reliable evaluation
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Preserve class distribution
    )

    # Initialize and train classifier
    clf = LogisticRegression(max_iter=600).fit(Xtr, ytr)

    # Generate predictions and performance metrics
    preds = clf.predict(Xte)
    print(f"===== {label_name} Classification Performance =====")
    print(f"Accuracy: {accuracy_score(yte, preds):.3f}")
    print(classification_report(yte, preds))

    # Create and save confusion matrix visualization
    disp = ConfusionMatrixDisplay.from_predictions(
        yte, preds,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )
    plt.title(f"{label_name} - Confusion Matrix")

    # Ensure models directory exists
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)

    # Save visualization
    fig_path = models_dir / f"{label_name.lower()}_cm.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {fig_path}")

    # Serialize model artifacts
    model_path = models_dir / f"{label_name.lower()}_lr.pkl"
    joblib.dump({"vec": None, "clf": clf}, model_path)

    return clf