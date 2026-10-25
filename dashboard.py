"""
Interactive Streamlit dashboard for exploring scored chat data.

Reuses the same src/aggregate.py tables as the static Matplotlib charts in
src/visualize.py, but renders them with Plotly so they're filterable and
hoverable instead of fixed PNGs.

Usage:
    streamlit run dashboard.py
"""

import glob
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src import aggregate

SENTIMENT_COLORS = {"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"}

st.set_page_config(page_title="Community Chat NLP Dashboard", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(
        lambda t: eval(t) if isinstance(t, str) and t.startswith("[") else []
    )
    return df


def pick_source_file() -> Optional[str]:
    candidates = sorted(glob.glob("data/processed/*_scored.csv"))
    if not candidates:
        return None
    labels = [os.path.basename(c) for c in candidates]
    choice = st.sidebar.selectbox("Scored dataset", labels)
    return candidates[labels.index(choice)]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    authors = sorted(df["author"].dropna().unique())
    selected_authors = st.sidebar.multiselect("Author", authors)
    if selected_authors:
        df = df[df["author"].isin(selected_authors)]

    labels = ["positive", "neutral", "negative"]
    selected_labels = st.sidebar.multiselect("Sentiment", labels, default=labels)
    if selected_labels:
        df = df[df["sentiment_label"].isin(selected_labels)]

    lengths = df["text_for_sentiment"].str.split().str.len().fillna(0)
    min_len, max_len = int(lengths.min()), int(lengths.max() or 1)
    if min_len < max_len:
        lo, hi = st.sidebar.slider("Message length (words)", min_len, max_len, (min_len, max_len))
        df = df[lengths.between(lo, hi)]

    return df


def render_kpis(df: pd.DataFrame):
    total = len(df)
    avg_sentiment = df["sentiment_compound"].mean() if total else 0
    pct_positive = (df["sentiment_label"] == "positive").mean() * 100 if total else 0
    n_authors = df["author"].nunique()

    cols = st.columns(4)
    cols[0].metric("Messages", f"{total:,}")
    cols[1].metric("Avg. sentiment", f"{avg_sentiment:+.3f}")
    cols[2].metric("% positive", f"{pct_positive:.1f}%")
    cols[3].metric("Authors", f"{n_authors:,}")


def render_sentiment_trend(df: pd.DataFrame):
    trend = aggregate.sentiment_trend_by_segment(df)
    fig = go.Figure()
    fig.add_bar(
        x=trend["segment"], y=trend["message_count"], name="Message volume",
        yaxis="y2", opacity=0.25, marker_color="gray",
    )
    fig.add_scatter(
        x=trend["segment"], y=trend["avg_sentiment"], name="Avg. sentiment",
        mode="lines+markers", line_color="#1f77b4",
    )
    fig.update_layout(
        title="Sentiment Across the Conversation Timeline",
        xaxis_title="Conversation segment (earliest → latest)",
        yaxis=dict(title="Avg. sentiment (compound)"),
        yaxis2=dict(title="Message volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hourly_activity(df: pd.DataFrame):
    hourly = aggregate.hourly_activity(df)
    fig = px.bar(
        hourly, x="hour", y="message_count",
        title="Community Activity by Hour of Day (UTC)",
        color_discrete_sequence=["#9467bd"],
    )
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Message count")
    st.plotly_chart(fig, use_container_width=True)


def render_top_authors(df: pd.DataFrame, n: int):
    authors = aggregate.top_authors(df, n)
    fig = px.bar(
        authors.sort_values("message_count"), x="message_count", y="author",
        orientation="h", color="avg_sentiment", color_continuous_scale="RdYlGn",
        title=f"Top {n} Most Active Authors (color = avg. sentiment)",
    )
    fig.update_layout(xaxis_title="Message count", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_distribution(df: pd.DataFrame):
    dist = aggregate.sentiment_distribution(df)
    fig = px.pie(
        dist, names="sentiment_label", values="count",
        color="sentiment_label", color_discrete_map=SENTIMENT_COLORS,
        title="Overall Sentiment Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_message_length(df: pd.DataFrame):
    lengths = aggregate.message_length_distribution(df)
    lengths = lengths[lengths <= lengths.quantile(0.98)] if len(lengths) else lengths
    fig = px.histogram(
        lengths, nbins=30, title="Message Length Distribution",
        color_discrete_sequence=["#9467bd"],
    )
    fig.update_layout(xaxis_title="Message length (words)", yaxis_title="Message count", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_word_frequency(df: pd.DataFrame, n: int):
    freq = aggregate.word_frequency(df, n)
    fig = px.bar(
        freq.sort_values("count"), x="count", y="word", orientation="h",
        title=f"Top {n} Most Frequent Words", color_discrete_sequence=["#ff7f0e"],
    )
    fig.update_layout(xaxis_title="Count", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_wordcloud(df: pd.DataFrame):
    all_tokens = [t for tokens in df["tokens"] for t in tokens]
    if not all_tokens:
        st.info("Not enough tokens to build a word cloud for the current filters.")
        return
    wc = WordCloud(width=1200, height=500, background_color="white").generate(" ".join(all_tokens))
    st.image(wc.to_array(), caption="Community Word Cloud", use_column_width=True)


def main():
    st.title("Community Chat NLP Dashboard")
    st.caption(
        "Interactive exploration of cleaned, sentiment-scored chat data — "
        "filter by author/sentiment/length in the sidebar."
    )

    source_path = pick_source_file()
    if source_path is None:
        st.error(
            "No scored dataset found in data/processed/. Run `python main.py` first "
            "to generate one."
        )
        return

    df = load_data(source_path)
    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("No messages match the current filters.")
        return

    render_kpis(filtered)
    st.divider()

    render_sentiment_trend(filtered)
    if "hour" in filtered.columns and filtered["hour"].notna().any():
        render_hourly_activity(filtered)

    col1, col2 = st.columns(2)
    with col1:
        render_top_authors(filtered, st.sidebar.slider("Top N authors", 5, 25, 10))
    with col2:
        render_sentiment_distribution(filtered)

    col3, col4 = st.columns(2)
    with col3:
        render_message_length(filtered)
    with col4:
        render_word_frequency(filtered, st.sidebar.slider("Top N words", 10, 40, 20))

    render_wordcloud(filtered)

    with st.expander("Raw filtered data"):
        st.dataframe(filtered.drop(columns=["tokens"]), use_container_width=True)


if __name__ == "__main__":
    main()
