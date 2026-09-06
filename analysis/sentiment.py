"""
InsightAI - Sentiment analysis (STEP 12).

Detects candidate text columns and scores sentiment. Prefers a local, free
model: VADER (via NLTK). If NLTK's VADER lexicon is unavailable it falls back
to a built-in lightweight lexicon so the application never fails. Also extracts
frequently used positive/negative words and simple themes.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ingestion.schema_detector import TEXT

# ------------------------------------------------------------------------ Lexicons

_POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "loved", "awesome", "perfect",
    "happy", "satisfied", "best", "helpful", "thanks", "thank", "recommend",
    "recommended", "fast", "easy", "useful", "superb", "wonderful", "friendly",
    "positive", "quality", "brilliant", "delighted", "impressive", "fantastic",
    "like", "liked", "comfortable", "reliable", "smooth", "beautiful", "nice",
    "pleased", "great", "enjoyed", "value", "clear", "quick", "responsive", "nice",
}
_NEGATIVE = {
    "bad", "terrible", "worst", "awful", "hate", "hated", "poor", "slow", "worse",
    "disappointed", "disappointing", "fail", "failed", "broken", "problem",
    "problems", "issue", "issues", "bad", "unhappy", "angry", "frustrated",
    "frustrating", "negative", "useless", "difficult", "unhelpful", "waste",
    "horrible", "rude", "late", "delayed", "error", "error", "bugs", "buggy",
    "complaint", "cancelled", "refund", "unclear", "confusing", "annoying",
    "scam", "overpriced", "unreliable", "garbage", "crappy", "disgusting", "drop",
    "never", "not", "no", "wrong", "cheap", "damaged", "missing", "ghost",
}
_STRONG_NEG = {"terrible", "awful", "worst", "horrible", "hate", "hated", "fail",
               "failed", "broken", "frustrated", "refund", "scam", "disgusting"}

# Remove overlapping/ambiguous tokens that would skew counts.
_NEGATIVE = _NEGATIVE - {"no", "not", "bad"}


def _valence(text: str) -> float:
    """Lexicon-based valence in [-1, 1]. Uses a small handling of negation."""
    if not text:
        return 0.0
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    if not tokens:
        return 0.0
    score = 0.0
    weight_total = 0
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok in _POSITIVE:
            val = 1.0
        elif tok in _NEGATIVE:
            val = -1.0
            if tok in _STRONG_NEG:
                val *= 1.8
        else:
            continue
        # Simple negation flip within 2 tokens before.
        window = tokens[max(0, i - 2): i]
        if any(w in {"not", "no", "never", "n't", "hardly", "barely"} for w in window):
            val *= 0.7
        # Intensifiers.
        if i > 0 and tokens[i - 1] in {"very", "really", "so", "extremely", "totally", "absolutely"}:
            val *= 1.5
        score += val
        weight_total += 1
    if weight_total == 0:
        return 0.0
    return max(-1.0, min(1.0, score / weight_total))


def _vader_scores(texts: List[str]) -> Optional[List[float]]:
    """Try NLTK VADER; return None on failure."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return [sia.polarity_scores(t)["compound"] for t in texts]
    except Exception:
        return None


def sentiment_of_text(text: str, sia=None, use_vader: bool = True) -> Dict:
    """Return {'compound', 'label'} for a single text string."""
    if use_vader and sia is not None:
        compound = sia.polarity_scores(text)["compound"]
    else:
        compound = _valence(text)
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {"compound": round(float(compound), 4), "label": label}


def analyze_sentiment(df: pd.DataFrame, text_cols: List[str],
                      use_vader: bool = True) -> Dict:
    """Score sentiment for each text column; build aggregate report."""
    if not text_cols:
        return {"available": False, "columns": [], "summary": {}}

    sia = None
    if use_vader:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
        except Exception:
            sia = None

    col_results: Dict[str, Dict] = {}
    for col in text_cols:
        series = df[col].astype(str).fillna("")
        compounds: List[float] = []
        labels: List[str] = []
        for text in series:
            text = text.strip()
            if not text:
                continue
            c = _valence(text) if sia is None else sia.polarity_scores(text)["compound"]
            compounds.append(float(c))
            labels.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
        if not compounds:
            continue
        counts = Counter(labels)
        total = len(compounds)
        pos_text = " ".join(series[i] for i, lbl in enumerate(labels) if lbl == "positive")
        neg_text = " ".join(series[i] for i, lbl in enumerate(labels) if lbl == "negative")
        col_results[col] = {
            "n": total,
            "positive_pct": round(100.0 * counts["positive"] / total, 2),
            "neutral_pct": round(100.0 * counts["neutral"] / total, 2),
            "negative_pct": round(100.0 * counts["negative"] / total, 2),
            "avg_compound": round(float(np.mean(compounds)), 4),
            "counts": dict(counts),
            "positive_words": _frequent_words(pos_text),
            "negative_words": _frequent_words(neg_text),
        }

    # Themes across all text.
    all_text = " ".join(str(x) for text_col in text_cols for x in df[text_col].astype(str).fillna(""))
    themes = _themes(all_text)

    return {
        "available": True,
        "columns": text_cols,
        "summary": col_results,
        "themes": themes,
        "use_vader": use_vader,
    }


def _frequent_words(text: str, k: int = 12) -> List[str]:
    tokens = re.findall(r"[a-z']+", text.lower())
    stop = _STOPWORDS
    filtered = [t for t in tokens if t not in stop and len(t) > 2]
    return [t for t, _ in Counter(filtered).most_common(k)]


def _themes(text: str, k: int = 8) -> List[str]:
    tokens = re.findall(r"[a-z']+", text.lower())
    stop = _STOPWORDS
    filtered = [t for t in tokens if t not in stop and len(t) > 2]
    return [t for t, _ in Counter(filtered).most_common(k)]


_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
    "has", "have", "had", "will", "would", "can", "could", "should", "their",
    "they", "them", "then", "than", "when", "where", "what", "which", "while",
    "who", "whose", "your", "yours", "you", "our", "ours", "not", "but", "all",
    "any", "because", "been", "being", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "too", "very", "just",
    "about", "above", "after", "again", "against", "before", "between", "through",
    "into", "onto", "been", "over", "under", "again", "there", "here", "also",
    "it", "its", "i", "we", "they", "be", "is", "do", "does", "got", "get", "one",
    "two", "on", "in", "at", "by", "of", "to", "as", "an",
}
