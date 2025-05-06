# AI‑Powered Email Classifier & Priority System

This system is **not web-hosted**. It runs locally using **Python + Streamlit** on Windows, macOS, or Linux.

## Windows Setup Instructions

### 1. Clone and install dependencies

```bat
git clone <your-repo-url>
cd your-repo-name
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Re‑train everything (models, charts, error sample)
Run retrain_all.py

# 3. Launch the GUI
streamlit run src/app.py in terminal
