# Community Chat NLP Analysis

A Python pipeline that pulls real community discussion data (from Reddit),
cleans and preprocesses it with NLTK, scores sentiment with VADER, and
produces visual insights with Matplotlib — sentiment trends over time,
activity patterns, most active contributors, and word frequency.

## Why this project

Online communities generate huge amounts of unstructured conversation.
This project answers questions like:
- Is the community's mood trending more positive or negative over time?
- When is the community most active?
- Who are the most engaged contributors, and what's their typical tone?
- What topics/words dominate the conversation?

## Pipeline

```
Reddit API (PRAW)  →  Clean (NLTK)  →  Sentiment (VADER)  →  Aggregate  →  Visualize (Matplotlib)
    ingest.py            clean.py        sentiment.py       aggregate.py    visualize.py
```

1. **Ingest** (`src/ingest.py`) — pulls hot posts and top-level comments from
   a given subreddit via PRAW, normalizing posts and comments into a single
   "message" schema.
2. **Clean** (`src/clean.py`) — removes deleted/bot messages and duplicates,
   strips URLs and subreddit mentions, and tokenizes text with NLTK
   (stopword removal for word-frequency analysis, while preserving
   case/punctuation separately for sentiment analysis — VADER relies on
   those signals).
3. **Sentiment** (`src/sentiment.py`) — scores every message with VADER,
   a lexicon-based sentiment model built for short, informal, social-media
   text (handles negation, punctuation emphasis, capitalization, emoji).
4. **Aggregate** (`src/aggregate.py`) — rolls per-message scores up into
   daily trends, hourly activity, top authors, and word frequency tables.
5. **Visualize** (`src/visualize.py`) — renders all of the above as saved
   PNG charts in `output/figures/`.

## Setup

```bash
git clone <your-repo-url>
cd community-chat-nlp-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get Reddit API credentials (free, ~2 minutes)

1. Go to https://www.reddit.com/prefs/apps
2. Click **"create another app..."** at the bottom
3. Choose type **"script"**
4. Name it anything (e.g. `community-chat-nlp-analysis`), redirect URI can be
   `http://localhost:8080`
5. Copy the client ID (under the app name) and client secret
6. Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

## Usage

Run the full pipeline on any subreddit:

```bash
python main.py learnpython --posts 100 --comments 50
```

This will:
1. Save raw pulled data to `data/raw/learnpython_raw.csv`
2. Save cleaned + sentiment-scored data to `data/processed/learnpython_scored.csv`
3. Save all charts to `output/figures/`

You can also run each stage independently:

```bash
python -m src.ingest learnpython --posts 100 --comments 50
python -m src.clean data/raw/learnpython_raw.csv
python -m src.sentiment data/processed/learnpython_clean.csv
python -m src.visualize data/processed/learnpython_scored.csv
```

## Sample output

_Run the pipeline and drop screenshots from `output/figures/` here, e.g.:_

- `sentiment_trend.png` — average sentiment per day, with message volume overlay
- `hourly_activity.png` — when the community is most active
- `top_authors.png` — most active contributors, colored by average sentiment
- `sentiment_distribution.png` — overall positive/neutral/negative split
- `word_frequency.png` / `wordcloud.png` — most common topics/words

## Tests

```bash
pytest
```

Covers text cleaning (URL/mention stripping, stopword removal, dedup logic)
and sentiment scoring (positive/negative/neutral classification correctness).

## Design decisions

- **Reddit over private chat exports**: public, legally safe to collect, and
  naturally timestamped/threaded like a chat community.
- **VADER over a transformer model**: VADER is purpose-built for short,
  informal text and runs instantly on thousands of messages with no GPU —
  a better fit than a heavier model for this scale and style of text.
- **Two cleaning paths**: sentiment analysis needs case/punctuation/emoji
  preserved (VADER uses them as signal), while word-frequency analysis needs
  them stripped — so cleaning produces both a lightly-cleaned and a
  heavily-cleaned version of each message.

## Possible extensions

- Topic modeling (LDA) to auto-label what each spike in conversation is about
- A small Streamlit dashboard for live, interactive exploration
- Compare sentiment across multiple subreddits
- Swap in a transformer-based sentiment model and compare results to VADER
