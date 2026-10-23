"""
Aggregation: turns per-message rows into summary tables used for visualization.
"""

from collections import Counter

import pandas as pd


def daily_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment compound score per day, plus message volume."""
    grouped = (
        df.groupby("date")
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            message_count=("message_id", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    return grouped


def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Message count per hour of day (0-23) — shows when the community is most active."""
    grouped = (
        df.groupby("hour")
        .agg(message_count=("message_id", "count"))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    return grouped


def top_authors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most active authors by message count, with their average sentiment."""
    grouped = (
        df[df["author"] != "nan"]
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
