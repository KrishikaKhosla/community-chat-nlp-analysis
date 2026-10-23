"""
End-to-end pipeline: ingest -> clean -> score sentiment -> aggregate -> visualize.

Two data sources are supported (see README for setup details):
  --source dataset   (default) a static public Discord-chat dataset, zero setup
  --source live       a real Discord channel you have access to, via a bot token

Usage:
    python main.py --source dataset --blocks 400
    python main.py --source live --channel-id 123456789012345678 --limit 500
"""

import argparse

from src import clean, discord_live, ingest, sentiment, visualize


def run_pipeline(source: str, blocks: int, channel_id: str, limit: int):
    print(f"\n=== Step 1/4: Loading data (source: {source}) ===")
    if source == "dataset":
        raw_df = ingest.fetch_discord_messages(blocks)
        ingest.save_raw_data(raw_df)
        scored_name = "discord_chat"
    else:
        if not channel_id:
            raise ValueError("--channel-id is required when --source live")
        raw_df = discord_live.fetch_channel_messages(channel_id, limit)
        discord_live.save_raw_data(raw_df, channel_id)
        scored_name = f"discord_live_{channel_id}"
    print(f"Loaded {len(raw_df)} raw messages")

    print("\n=== Step 2/4: Cleaning data ===")
    clean_df = clean.clean_messages(raw_df)
    print(f"Kept {len(clean_df)}/{len(raw_df)} messages after cleaning")

    print("\n=== Step 3/4: Scoring sentiment (VADER) ===")
    scored_df = sentiment.score_sentiment(clean_df)
    scored_path = f"data/processed/{scored_name}_scored.csv"
    scored_df.to_csv(scored_path, index=False)
    print(f"Saved scored data -> {scored_path}")
    print(scored_df["sentiment_label"].value_counts())

    print("\n=== Step 4/4: Generating figures ===")
    visualize.generate_all_figures(scored_df)

    print("\nDone! Check output/figures/ for charts and data/processed/ for the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Community Chat NLP Analysis pipeline."
    )
    parser.add_argument(
        "--source",
        choices=["dataset", "live"],
        default="dataset",
        help="'dataset' = static public dataset (no setup). 'live' = a real Discord "
        "channel you have access to (requires a bot token, see README).",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=400,
        help="[dataset source] number of raw chat blocks to parse (~15-20 messages each)",
    )
    parser.add_argument(
        "--channel-id", default=None, help="[live source] Discord channel ID to pull from"
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="[live source] max messages to pull"
    )
    args = parser.parse_args()

    run_pipeline(args.source, args.blocks, args.channel_id, args.limit)
