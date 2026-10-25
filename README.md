# Community Chat NLP Analysis

A Python pipeline that analyzes real community chat data, cleaning and
preprocessing with NLTK, scoring sentiment with VADER, and producing visual
insights with Matplotlib: sentiment trends, most active contributors,
message patterns, and word frequency.

Two pluggable data sources are supported:
- **`dataset`** (default, zero setup) — a public dataset of real
  **EleutherAI Discord server logs** (the open-source AI research community
  behind GPT-Neo/GPT-J), downloaded directly from Hugging Face.
- **`live`** — pulls real, current message history from any Discord channel
  you have access to, via a bot token you create yourself.

Both sources feed the exact same clean → sentiment → aggregate → visualize
pipeline, normalized to a common message schema — a small "pluggable data
source" design that keeps the analysis code source-agnostic.

## Why this project

Online communities generate huge amounts of unstructured conversation.
This project answers questions like:
- Is the community's mood trending more positive or negative over time?
- Who are the most engaged contributors, and what's their typical tone?
- What topics/words dominate the conversation?
- What does typical engagement (message length, sentiment split) look like?

## Pipeline

```
                 ┌─ ingest.py        (static dataset, no setup) ─┐
                 ├─ discord_live.py  (your own live channel)     ┴→  Clean (NLTK)  →  Sentiment (VADER)  →  Aggregate  →  Visualize (Matplotlib)
                 └────────────────────────────────────────────────    clean.py         sentiment.py        aggregate.py    visualize.py
```

1. **Ingest** — two interchangeable sources, normalized to the same schema
   (`message_id`, `author`, `text`, plus timing info):
   - `src/ingest.py` downloads a public Hugging Face dataset of real
     Discord chat logs (`breadlicker45/discord-chat`) and parses the raw
     `"Username#1234: message"` text blocks into individual messages.
   - `src/discord_live.py` pulls live message history from a real Discord
     channel via Discord's REST API, using a bot token you provide.
2. **Clean** (`src/clean.py`) — drops deleted/empty messages and Discord
   system notices (joins, pins), strips URLs and Discord mention IDs, and
   tokenizes text with NLTK (stopword removal for word-frequency analysis,
   while preserving case/punctuation separately for sentiment analysis —
   VADER relies on those signals). Also buckets messages into sequential
   "segments" for trend analysis, and — only when real timestamps are
   available (the `live` source) — adds real date/hour columns too.
3. **Sentiment** (`src/sentiment.py`) — scores every message with VADER,
   a lexicon-based sentiment model built for short, informal, social-media
   text (handles negation, punctuation emphasis, capitalization, emoji).
4. **Aggregate** (`src/aggregate.py`) — rolls per-message scores up into
   a conversation-timeline trend, top authors, message-length distribution,
   and word frequency tables (plus hour-of-day activity, when real
   timestamps are available).
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

## Usage

### Option A: static dataset (default, zero setup)

```bash
python main.py --source dataset --blocks 400
```

No API keys needed — data is downloaded directly from Hugging Face's public
dataset hub. `--blocks` controls how much of the dataset to parse (each
block is roughly 15-20 messages; the full dataset has ~11,000 blocks /
~200k messages total). 400 blocks (~7,000 messages) runs in under a minute.

### Option B: your own live Discord channel

```bash
cp .env.example .env   # fill in DISCORD_BOT_TOKEN — see .env.example for setup steps
python main.py --source live --channel-id 123456789012345678 --limit 500
```

Setup (one-time, ~10 minutes):
1. Create a bot at https://discord.com/developers/applications
2. Under the bot's settings, enable **"Message Content Intent"**
3. Under **OAuth2 > URL Generator**, check scope `bot` and permissions
   `View Channels` + `Read Message History`, then open the generated URL to
   invite the bot to a server you have access to
4. Copy the bot token into `.env` as `DISCORD_BOT_TOKEN`
5. In Discord, enable **Developer Mode** (User Settings > Advanced), then
   right-click the channel you want to analyze > **Copy Channel ID**

Both options write to the same places: raw data in `data/raw/`, cleaned +
scored data in `data/processed/`, and charts in `output/figures/`.

You can also run each stage independently:

```bash
python -m src.ingest --blocks 400                       # or: python -m src.discord_live <channel_id>
python -m src.clean data/raw/discord_chat_raw.csv
python -m src.sentiment data/processed/discord_chat_clean.csv
python -m src.visualize data/processed/discord_chat_scored.csv
```

## Interactive dashboard

For live, filterable exploration instead of static PNGs, run the Streamlit
dashboard against any scored CSV in `data/processed/`:

```bash
streamlit run dashboard.py
```

It reuses the exact same `src/aggregate.py` tables as the static charts
above, re-rendered with Plotly, plus:
- sidebar filters for author, sentiment label, and message length
- KPI tiles (message count, avg. sentiment, % positive, author count)
- a dataset picker if `data/processed/` has more than one scored CSV

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

Running with `--source live` against a channel with real timestamps also
produces `hourly_activity.png` — when the community is most active.

## Tests

```bash
pytest
```

14 tests covering: raw chat-block parsing (including multi-line message
continuations), live-API message fetching and pagination (mocked HTTP),
text cleaning (URL/mention stripping, stopword removal, system-message and
dedup filtering), and sentiment scoring correctness.

## Design decisions

- **Pluggable ingestion over a single hardcoded source**: `src/ingest.py`
  and `src/discord_live.py` are two independent adapters that both produce
  the same message schema, so the rest of the pipeline (clean, sentiment,
  aggregate, visualize) doesn't know or care which one supplied the data.
  This also made it easy to add the live source later without touching
  anything downstream.
- **A public, pre-hosted dataset as the zero-setup default**: an earlier
  version of this pipeline only scraped Reddit live, but public scraping
  endpoints turned out to be unreliable to reach from many cloud/sandboxed
  environments (aggressive bot-blocking on datacenter IPs), and anyone
  wanting to run the project would first need to register a Reddit
  developer app. Defaulting to a dataset served over plain HTTPS with no
  auth keeps the project runnable by anyone (e.g. an interviewer) with zero
  setup, while `--source live` is there for real, personalized analysis.
- **Discord's REST message-history endpoint over a persistent gateway
  bot**: `discord.py`'s typical pattern runs a bot as a long-lived process
  listening for new messages in real time — built for chatbots, not a
  one-shot analysis script. Paging backwards through history via the REST
  API gets the same real, live data but lets the script exit cleanly when done.
- **Sequential "conversation segments" as a universal time axis**: the
  static dataset has no real timestamps, so trend analysis buckets messages
  by their position in the (chronologically-ordered) history rather than
  pretending to have real dates. When real timestamps are available (the
  live source), the pipeline additionally unlocks real date/hour-of-day
  analysis — see `clean.py`'s conditional column logic.
- **VADER over a transformer model**: VADER is purpose-built for short,
  informal text and runs instantly on thousands of messages with no GPU —
  a better fit than a heavier model for this scale and style of text.
- **Two cleaning paths**: sentiment analysis needs case/punctuation/emoji
  preserved (VADER uses them as signal), while word-frequency analysis needs
  them stripped — so cleaning produces both a lightly-cleaned and a
  heavily-cleaned version of each message.

## Possible extensions

- Topic modeling (LDA) to auto-label what each spike in conversation is about
- Swap in a transformer-based sentiment model and compare results to VADER
- A Reddit (PRAW) ingestion adapter alongside the two Discord sources, for
  cross-community comparison
