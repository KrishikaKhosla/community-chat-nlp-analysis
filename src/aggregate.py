"""
Aggregation: turns per-message rows into summary tables used for visualization.
"""

from collections import Counter

import pandas as pd


def sentiment_trend_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment compound score per conversation segment, plus
    message volume. `segment` is a sequential bucket standing in for a time
    axis (see src/clean.py) since this dataset has no real timestamps."""
    grouped = (
        df.groupby("segment")
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            message_count=("message_id", "count"),
        )
        .reset_index()
        .sort_values("segment")
    )
    return grouped


def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Message count per hour of day (0-23). Only meaningful when real
    timestamps are available (e.g. from the live Discord source) — the
    static dataset source has no real clock times, so callers should check
    for an `hour` column before using this."""
    grouped = (
        df.groupby("hour")
        .agg(message_count=("message_id", "count"))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    return grouped


def message_length_distribution(df: pd.DataFrame) -> pd.Series:
    """Word count per message — shows typical message length in the community."""
    return df["text_for_sentiment"].str.split().str.len()


def top_authors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most active authors by message count, with their average sentiment."""
    grouped = (
        df[df["author"].notna()]
        .groupby("author")
        .agg(
            message_count=("message_id", "count"),
            avg_sentiment=("sentiment_compound", "mean"),
        )
        .reset_index()
        .sort_values("message_count", ascending=False)
        .head(n)
    )
    return grouped


def word_frequency(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Most common words across all cleaned token lists."""
    all_tokens = []
    for tokens in df["tokens"]:
        if isinstance(tokens, str):
            # tokens may come back from CSV as a stringified list
            tokens = eval(tokens) if tokens.startswith("[") else tokens.split()
        all_tokens.extend(tokens)

    counts = Counter(all_tokens)
    top = counts.most_common(n)
    return pd.DataFrame(top, columns=["word", "count"])


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Overall counts of positive/neutral/negative messages."""
    return df["sentiment_label"].value_counts().reset_index(
        name="count"
    ).rename(columns={"index": "sentiment_label"})
