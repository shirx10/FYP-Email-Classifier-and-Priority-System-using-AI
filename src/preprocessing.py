# preprocessing.py
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

_stop = set(stopwords.words('english'))
_lemma = WordNetLemmatizer()

# ──────────────────────────────────────────────────────────────────────────
def parse_enron_email(raw: str):
    """Return (subject, sender, body) from a raw Enron message string."""
    subject = sender = None
    body_lines, in_hdrs = [], True
    for line in raw.splitlines():
        if in_hdrs and not line.strip():
            in_hdrs = False; continue
        if in_hdrs:
            if line.lower().startswith("subject:"):
                subject = line.split(":",1)[1].strip()
            elif line.lower().startswith("from:"):
                sender  = line.split(":",1)[1].strip()
        else:
            body_lines.append(line)
    return subject or "", sender or "", "\n".join(body_lines)

# ──────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Lower-case, strip URLs, HTML, non-alpha; stop-word remove, lemmatise."""
    text  = text.lower()
    text  = re.sub(r'http\S+', ' ', text)
    text  = re.sub(r'<[^>]+>', ' ', text)
    text  = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [ _lemma.lemmatize(t) for t in text.split() if t not in _stop ]
    return " ".join(tokens)
