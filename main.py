"""
End-to-end pipeline: ingest -> clean -> score sentiment -> aggregate -> visualize.

Usage:
    python main.py learnpython --posts 100 --comments 50
"""

import argparse

from src import ingest, clean, sentiment, visualize


def run_pipeline(subreddit_name: str, post_limit: int, comments_per_post: int):
    print(f"\n=== Step 1/4: Ingesting data from r/{subreddit_name} ===")
    raw_df = ingest.fetch_subreddit_messages(subreddit_name, post_limit, comments_per_post)
    ingest.save_raw_data(raw_df, subreddit_name)
    print(f"Pulled {len(raw_df)} raw messages")

    print("\n=== Step 2/4: Cleaning data ===")
    clean_df = clean.clean_messages(raw_df)
    print(f"Kept {len(clean_df)}/{len(raw_df)} messages after cleaning")

    print("\n=== Step 3/4: Scoring sentiment (VADER) ===")
    scored_df = sentiment.score_sentiment(clean_df)
    scored_path = f"data/processed/{subreddit_name}_scored.csv"
    scored_df.to_csv(scored_path, index=False)
    print(f"Saved scored data -> {scored_path}")
    print(scored_df["sentiment_label"].value_counts())

    print("\n=== Step 4/4: Generating figures ===")
    visualize.generate_all_figures(scored_df)

    print("\nDone! Check output/figures/ for charts and data/processed/ for the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Community Chat NLP Analysis pipeline on a subreddit."
    )
    parser.add_argument("subreddit", help="Subreddit name, e.g. 'learnpython'")
    parser.add_argument("--posts", type=int, default=100, help="Number of hot posts to pull")
    parser.add_argument("--comments", type=int, default=50, help="Top-level comments per post")
    args = parser.parse_args()

    run_pipeline(args.subreddit, args.posts, args.comments)
