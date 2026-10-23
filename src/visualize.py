"""
Visualization: turns aggregated tables into saved chart images.

All charts are saved as PNGs into output/figures/ so they can be dropped
straight into the README or a portfolio/demo.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src import aggregate

FIGURES_DIR = "output/figures"


def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_sentiment_trend(df: pd.DataFrame):
    trend = aggregate.daily_sentiment_trend(df)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(trend["date"], trend["avg_sentiment"], color="tab:blue", marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Avg. sentiment (compound)", color="tab:blue")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(trend["date"], trend["message_count"], alpha=0.2, color="tab:gray")
    ax2.set_ylabel("Message volume", color="tab:gray")

    fig.suptitle("Community Sentiment Over Time")
    fig.tight_layout()
    _save(fig, "sentiment_trend.png")


def plot_hourly_activity(df: pd.DataFrame):
    hourly = aggregate.hourly_activity(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly["hour"], hourly["message_count"], color="tab:purple")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Message count")
    ax.set_title("Community Activity by Hour of Day")
    ax.set_xticks(range(0, 24, 2))

    fig.tight_layout()
    _save(fig, "hourly_activity.png")


def plot_top_authors(df: pd.DataFrame, n: int = 10):
    authors = aggregate.top_authors(df, n)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:green" if s >= 0 else "tab:red" for s in authors["avg_sentiment"]]
    ax.barh(authors["author"], authors["message_count"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Message count")
    ax.set_title(f"Top {n} Most Active Authors (color = avg. sentiment)")

    fig.tight_layout()
    _save(fig, "top_authors.png")


def plot_sentiment_distribution(df: pd.DataFrame):
    dist = df["sentiment_label"].value_counts()
    colors = {"positive": "tab:green", "neutral": "tab:gray", "negative": "tab:red"}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        dist.values,
        labels=dist.index,
        autopct="%1.1f%%",
        colors=[colors[label] for label in dist.index],
        startangle=90,
    )
    ax.set_title("Overall Sentiment Distribution")

    fig.tight_layout()
    _save(fig, "sentiment_distribution.png")


def plot_word_frequency(df: pd.DataFrame, n: int = 20):
    freq = aggregate.word_frequency(df, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(freq["word"][::-1], freq["count"][::-1], color="tab:orange")
    ax.set_xlabel("Count")
    ax.set_title(f"Top {n} Most Frequent Words")

    fig.tight_layout()
    _save(fig, "word_frequency.png")


def plot_wordcloud(df: pd.DataFrame):
    all_tokens = []
    for tokens in df["tokens"]:
        if isinstance(tokens, str):
            tokens = eval(tokens) if tokens.startswith("[") else tokens.split()
        all_tokens.extend(tokens)

    text = " ".join(all_tokens)
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Community Word Cloud")

    fig.tight_layout()
    _save(fig, "wordcloud.png")


def generate_all_figures(df: pd.DataFrame):
    plot_sentiment_trend(df)
    plot_hourly_activity(df)
    plot_top_authors(df)
    plot_sentiment_distribution(df)
    plot_word_frequency(df)
    plot_wordcloud(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate all figures from scored message data.")
    parser.add_argument("scored_csv", help="Path to scored CSV from sentiment.py")
    args = parser.parse_args()

    data = pd.read_csv(args.scored_csv)
    generate_all_figures(data)
