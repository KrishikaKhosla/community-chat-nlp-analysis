"""
End-to-end pipeline: ingest -> clean -> score sentiment -> aggregate -> visualize.

Usage:
    python main.py --blocks 400
"""

import argparse

from src import ingest, clean, sentiment, visualize


def run_pipeline(num_blocks: int):
    print(f"\n=== Step 1/4: Loading {num_blocks} chat blocks from the dataset ===")
    raw_df = ingest.fetch_discord_messages(num_blocks)
    ingest.save_raw_data(raw_df)
    print(f"Parsed {len(raw_df)} raw messages")

    print("\n=== Step 2/4: Cleaning data ===")
    clean_df = clean.clean_messages(raw_df)
    print(f"Kept {len(clean_df)}/{len(raw_df)} messages after cleaning")

    print("\n=== Step 3/4: Scoring sentiment (VADER) ===")
    scored_df = sentiment.score_sentiment(clean_df)
    scored_path = "data/processed/discord_chat_scored.csv"
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
        "--blocks",
        type=int,
        default=400,
        help="Number of raw chat blocks to parse (each block ~15-20 messages)",
    )
    args = parser.parse_args()

    run_pipeline(args.blocks)
