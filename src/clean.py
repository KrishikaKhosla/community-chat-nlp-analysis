"""
Cleaning and preprocessing for raw chat/message data.

Two separate outputs are produced per message, because sentiment analysis
and word-frequency analysis want different things:
  - `text_for_sentiment`: lightly cleaned (keep punctuation/case/emoji — VADER
    actually uses capitalization, exclamation points, and emoji as signal).
  - `tokens`: heavily cleaned (lowercased, stopwords removed, punctuation
    stripped) — used for word frequency / word clouds, where stopwords and
    punctuation are just noise.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords"]


def ensure_nltk_data():
    """Downloads required NLTK corpora (no-op if already present locally)."""
    for resource in _NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"/u/\w+|/r/\w+|u/\w+|r/\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")

REMOVED_MARKERS = {"[deleted]", "[removed]", "", "none", "nan"}


def basic_clean_text(text: str) -> str:
    """Light cleaning: strip URLs/subreddit-mentions and collapse whitespace.
    Keeps case and punctuation intact since VADER relies on them."""
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize_for_frequency(text: str, stop_words: set) -> list:
    """Heavy cleaning for word-frequency analysis: lowercase, alphabetic
    tokens only, stopwords removed."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]


def clean_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw ingested messages and returns a cleaned DataFrame ready for
    sentiment analysis and aggregation.

    Steps:
      1. Drop deleted/removed/bot messages
      2. Drop duplicates and empty text
      3. Add a lightly-cleaned `text_for_sentiment` column
      4. Add a `tokens` column (list of cleaned words) for frequency analysis
      5. Convert `created_utc` into a proper datetime column
    """
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    cleaned = df.copy()

    # Drop rows with no usable text or from deleted/bot accounts.
    cleaned["text"] = cleaned["text"].astype(str)
    cleaned = cleaned[~cleaned["text"].str.lower().isin(REMOVED_MARKERS)]
    cleaned = cleaned[~cleaned["author"].isin(["None", "AutoModerator"])]

    cleaned = cleaned.drop_duplicates(subset="message_id")
    cleaned = cleaned.dropna(subset=["text"])

    cleaned["text_for_sentiment"] = cleaned["text"].apply(basic_clean_text)
    cleaned = cleaned[cleaned["text_for_sentiment"].str.len() > 0]

    cleaned["tokens"] = cleaned["text_for_sentiment"].apply(
        lambda t: tokenize_for_frequency(t, stop_words)
    )

    cleaned["created_dt"] = pd.to_datetime(cleaned["created_utc"], unit="s")
    cleaned["date"] = cleaned["created_dt"].dt.date
    cleaned["hour"] = cleaned["created_dt"].dt.hour

    return cleaned.reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw ingested message data.")
    parser.add_argument("raw_csv", help="Path to raw CSV from ingest.py")
    parser.add_argument("--out", default=None, help="Output path for cleaned CSV")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw_csv)
    cleaned_df = clean_messages(raw_df)

    out_path = args.out or args.raw_csv.replace("raw", "processed").replace(
        "_raw", "_clean"
    )
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned {len(cleaned_df)}/{len(raw_df)} messages -> {out_path}")
