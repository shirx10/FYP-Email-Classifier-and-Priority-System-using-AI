# ──────────────────────────────────────────────────────────────
# Main Application: AI Email Classifier Interface
#
# This Streamlit application provides a dual-mode interface for:
# 1. Single email classification (manual input)
# 2. Bulk processing of email datasets (CSV/Enron corpus)
#
# Core functionality includes:
# - Text preprocessing and feature extraction
# - Category classification (work/personal/spam)
# - Dual priority scoring (rule-based and ML-driven)
# - Interactive filtering and visualization
# ──────────────────────────────────────────────────────────────

import streamlit as st
st.set_page_config(page_title="AI Email Classifier", layout="wide")  # Must be first Streamlit call

import pandas as pd, joblib
from pathlib import Path
from preprocessing import preprocess_text
from priority import priority_rule

# ── Model Loading ─────────────────────────────────────────────
# Load pre-trained machine learning models for classification
# Models are stored as joblib files containing:
# - vec: TF-IDF vectorizer
# - clf: Logistic Regression classifier
root = Path(__file__).resolve().parents[1]
models_dir = root / "models"

# Load category classification model (work/personal/spam)
cat_model = joblib.load(models_dir / "category_lr.pkl")
vec_cat, clf_cat = cat_model["vec"], cat_model["clf"]

# Load priority prediction model (if available)
prio_file = models_dir / "priority_lr.pkl"
if prio_file.exists():
    prio_model = joblib.load(prio_file)
    vec_prio, clf_prio = prio_model["vec"], prio_model["clf"]
else:
    vec_prio = clf_prio = None  # Fallback to rule-based only

# ── User Interface Setup ──────────────────────────────────────
st.title("AI-Powered Email Classifier & Priority Viewer")

# Sidebar for input mode selection
with st.sidebar:
    st.header("Input options")
    mode = st.radio("Choose input mode",
                   ["Single e-mail", "Upload / Enron CSV"],
                   index=0)

# ═════════════════════════════════════════════════════════════
# SINGLE EMAIL CLASSIFICATION MODE
# ═════════════════════════════════════════════════════════════
if mode == "Single e-mail":
    # Input fields for manual email entry
    sender = st.text_input("Sender", "boss@enron.com")
    subject = st.text_input("Subject", "URGENT: Need budget numbers ASAP")
    body_raw = st.text_area("Body", height=180,
                          value="Hi team, we need those Q3 numbers by EOD. Thanks!")

    if st.button("Classify"):
        # Preprocess and combine text features
        clean = preprocess_text(body_raw)
        combined = f"{subject.lower()} {clean}"

        # Run predictions
        cat = clf_cat.predict(vec_cat.transform([combined]))[0]
        pr_r = priority_rule(subject, sender)
        pr_m = (clf_prio.predict(vec_prio.transform([combined]))[0]
                if clf_prio else "–no-ML-model–")

        # Display results
        st.subheader("Result")
        st.write({
            "Sender": sender,
            "Subject": subject,
            "Category": cat,
            "Priority (Rule)": pr_r,
            "Priority (ML)": pr_m
        })

# ═════════════════════════════════════════════════════════════
# BULK PROCESSING MODE
# ═════════════════════════════════════════════════════════════
else:
    st.subheader("Bulk classify data")

    # Data source selection (Enron dataset or user upload)
    enron_path = root / "data" / "enron_balanced.csv"
    use_enron = st.checkbox(f"Use bundled Enron CSV ({enron_path.name})",
                           value=True)

    if use_enron:
        df_raw = pd.read_csv(enron_path)
        st.caption(f"Loaded {len(df_raw):,} rows from {enron_path.name}")
    else:
        upload = st.file_uploader("Upload your own CSV", type=["csv"])
        if upload is None:
            st.stop()
        df_raw = pd.read_csv(upload)
        st.caption(f"Loaded {len(df_raw):,} rows")

    # Configure sample size for display
    sample_n = st.slider("Number of random emails to display",
                        10, 200, 30, step=10)
    df = df_raw.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # Validate required columns
    need = {"sender", "subject", "body"}
    if not need.issubset(df.columns):
        st.error(f"CSV must contain: {', '.join(need)}")
        st.stop()

    # Preprocess and predict
    df["clean_body"] = df["body"].apply(preprocess_text)
    df["combined"] = df["subject"].str.lower().fillna("") + " " + df["clean_body"]

    # Category prediction
    Xc = vec_cat.transform(df["combined"])
    df["Category"] = clf_cat.predict(Xc)

    # Priority predictions
    df["PriorityRule"] = df.apply(lambda r: priority_rule(r["subject"],
                                                       r["sender"]),
                                                       axis=1)
    df["PriorityML"] = (clf_prio.predict(vec_prio.transform(df["combined"]))
                       if clf_prio else "–no-ML-model–")

    # ── Interactive Filters ───────────────────────────────────
    # Priority filters (rule-based and ML)
    prio_rule_sel = st.multiselect(
        "Filter by Rule priority",
        ["high", "medium", "low"],
        default=["high", "medium", "low"]
    )

    # Dynamic ML priority options based on actual predictions
    prio_ml_vals = sorted(df["PriorityML"].unique())
    prio_ml_sel = st.multiselect(
        "Filter by ML priority",
        prio_ml_vals,
        default=prio_ml_vals
    )

    # Category filter
    cat_sel = st.multiselect(
        "Filter by category",
        sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique())
    )

    # Apply filters to dataframe view
    df_view = df[
        df["PriorityRule"].isin(prio_rule_sel)
        & df["PriorityML"].isin(prio_ml_sel)
        & df["Category"].isin(cat_sel)
    ]

    # Color coding for priority levels
    PRIORITY_COLORS = {"high": "#ff6b6b", "medium": "#ffd93d", "low": "#51cf66"}

    def apply_priority_border(row):
        """Add left border styling based on rule priority"""
        styles = [""] * len(row)
        if row.PriorityRule in PRIORITY_COLORS:
            styles[0] = f"border-left:6px solid {PRIORITY_COLORS[row.PriorityRule]}"
        return styles

    # Display filtered results
    st.dataframe(
        df_view[["sender", "subject", "Category", "PriorityRule", "PriorityML"]]
        .style.apply(apply_priority_border, axis=1),
        use_container_width=True,
        height=min(600, 35 * len(df_view) + 35)  # Dynamic height based on row count
    )

    # Email body preview (for smaller selections)
    if len(df_view) <= 200:
        st.markdown("### Email bodies")
        for _, row in df_view.iterrows():
            title =(row.subject if pd.notna(row.subject) else "")[:80]
            with st.expander(title):
                st.write(row.body)
    else:
        st.info("Body expanders disabled for large selections (>200 rows)")

    # Data export
    st.download_button(
        "Download current view as CSV",
        df_view.to_csv(index=False).encode(),
        "classified_emails.csv",
        "text/csv"
    )