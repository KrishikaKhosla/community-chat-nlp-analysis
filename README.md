# Community Chat NLP Analysis

A Python pipeline that analyzes real community chat data — the public
**EleutherAI Discord server logs** (the open-source AI research community
behind GPT-Neo and GPT-J) — cleaning and preprocessing with NLTK, scoring
sentiment with VADER, and producing visual insights with Matplotlib:
sentiment trends, most active contributors, message patterns, and word
frequency.

## Why this project

Online communities generate huge amounts of unstructured conversation.
This project answers questions like:
- Is the community's mood trending more positive or negative over time?
- Who are the most engaged contributors, and what's their typical tone?
- What topics/words dominate the conversation?
- What does typical engagement (message length, sentiment split) look like?

## Pipeline

```
HF Dataset (parquet)  →  Clean (NLTK)  →  Sentiment (VADER)  →  Aggregate  →  Visualize (Matplotlib)
     ingest.py            clean.py         sentiment.py        aggregate.py    visualize.py
```

1. **Ingest** (`src/ingest.py`) — downloads real Discord chat logs from a
   public Hugging Face dataset (`breadlicker45/discord-chat`) and parses
   the raw `"Username#1234: message"` text blocks into individual messages.
2. **Clean** (`src/clean.py`) — drops deleted/empty messages and Discord
   system notices (joins, pins), strips URLs and Discord mention IDs, and
   tokenizes text with NLTK (stopword removal for word-frequency analysis,
   while preserving case/punctuation separately for sentiment analysis —
   VADER relies on those signals).
3. **Sentiment** (`src/sentiment.py`) — scores every message with VADER,
   a lexicon-based sentiment model built for short, informal, social-media
   text (handles negation, punctuation emphasis, capitalization, emoji).
4. **Aggregate** (`src/aggregate.py`) — rolls per-message scores up into
   a conversation-timeline trend, top authors, message-length distribution,
   and word frequency tables.
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

No API keys or account setup needed — data is downloaded directly from
Hugging Face's public dataset hub.

## Usage

Run the full pipeline:

```bash
python main.py --blocks 400
```

`--blocks` controls how much of the dataset to parse (each block is roughly
15-20 messages; the full dataset has ~11,000 blocks / ~200k messages total).
400 blocks (~7,000 messages) runs in well under a minute.

This will:
1. Save raw parsed data to `data/raw/discord_chat_raw.csv`
2. Save cleaned + sentiment-scored data to `data/processed/discord_chat_scored.csv`
3. Save all charts to `output/figures/`

You can also run each stage independently:

```bash
python -m src.ingest --blocks 400
python -m src.clean data/raw/discord_chat_raw.csv
python -m src.sentiment data/processed/discord_chat_clean.csv
python -m src.visualize data/processed/discord_chat_scored.csv
```

## Sample output

On a run of 400 blocks (~6,500 cleaned messages), sentiment skewed clearly
positive (41% positive vs. 12% negative), consistent with a collaborative
research community. See `output/figures/` after running the pipeline for:

- `sentiment_trend.png` — average sentiment across the conversation
  timeline, with message volume overlaid
- `top_authors.png` — most active contributors, colored by average sentiment
- `sentiment_distribution.png` — overall positive/neutral/negative split
- `message_length_distribution.png` — typical message length in the community
- `word_frequency.png` / `wordcloud.png` — most common topics/words
  (dominated by terms like "data", "model", "channel", "welcome" — a
  believable fingerprint of an early ML research Discord)

## Tests

```bash
pytest
```

12 tests covering: raw chat-block parsing (including multi-line message
continuations), text cleaning (URL/mention stripping, stopword removal,
system-message and dedup filtering), and sentiment scoring correctness.

## Design decisions

- **A public, pre-hosted dataset over live scraping**: the pipeline
  originally scraped Reddit live, but public scraping endpoints are
  unreliable to reach from many cloud/sandboxed environments (aggressive
  bot-blocking on datacenter IPs) and the official API requires registering
  a developer app. Hugging Face's dataset hub serves real, publicly-shared
  chat data over plain HTTPS with no auth and no rate-limit dance — far more
  reproducible for anyone (e.g. an interviewer) who wants to actually run it.
- **Sequential "conversation segments" instead of real timestamps**: this
  dataset doesn't include message timestamps, so trend analysis buckets
  messages by their position in the (chronologically-ordered) chat history
  rather than pretending to have real dates — an honest substitute time axis.
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
- Swap in a transformer-based sentiment model and compare results to VADER
- Apply the same pipeline to a different public chat dataset for comparison
