import pandas as pd

KEY_HIGH_SUB   = ["urgent", "asap", "deadline", "budget", "important"]
KEY_LOW_SENDER = ["friend", "mom", "dad", "@yahoo.", "@gmail."]

def _safe(x: str) -> str:
    """Return lower‑case string; convert NaN/None to empty."""
    return "" if pd.isna(x) else str(x).lower()

def priority_rule(subject, sender) -> str:
    ssub  = _safe(subject)
    sfrom = _safe(sender)

    if any(k in ssub for k in KEY_HIGH_SUB) or any(tag in sfrom for tag in ["vp@", "ceo@"]):
        return "high"
    if any(k in sfrom for k in KEY_LOW_SENDER):
        return "low"
    return "medium"
