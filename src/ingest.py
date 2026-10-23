"""
Data ingestion: pulls real community discussion data from a subreddit.

Why Reddit instead of a private Discord/WhatsApp export?
- It's public data we're allowed to collect (no ToS/privacy issues).
- Threads are naturally a "community chat" with replies, timestamps, and authors.
- PRAW (Python Reddit API Wrapper) makes pulling it straightforward.

Each row we collect is one message (a post OR a comment), normalized into the
same shape, so the rest of the pipeline doesn't need to care which it came from.
"""

import os
import time
import praw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_reddit_client() -> praw.Reddit:
    """Builds an authenticated, read-only Reddit client from .env credentials."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        raise EnvironmentError(
            "Missing Reddit API credentials. Copy .env.example to .env and fill it in."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit


def fetch_subreddit_messages(
    subreddit_name: str,
    post_limit: int = 100,
    comments_per_post: int = 50,
) -> pd.DataFrame:
    """
    Pulls hot posts from a subreddit plus their top-level comments.

    Returns a DataFrame with one row per "message" (post or comment):
        message_id, parent_type, author, text, created_utc, score, subreddit
    """
    reddit = get_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)

    rows = []

    for submission in subreddit.hot(limit=post_limit):
        rows.append(
            {
                "message_id": submission.id,
                "parent_type": "post",
                "author": str(submission.author),
                "text": f"{submission.title} {submission.selftext or ''}".strip(),
                "created_utc": submission.created_utc,
                "score": submission.score,
                "subreddit": subreddit_name,
            }
        )

        # Expand only top-level comments (fast, and most "chat-like").
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:comments_per_post]:
            rows.append(
                {
                    "message_id": comment.id,
                    "parent_type": "comment",
                    "author": str(comment.author),
                    "text": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "subreddit": subreddit_name,
                }
            )

        # Be polite to Reddit's API — avoid hammering it.
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    return df


def save_raw_data(df: pd.DataFrame, subreddit_name: str) -> str:
    """Saves the raw pulled data to data/raw/ and returns the file path."""
    out_path = f"data/raw/{subreddit_name}_raw.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull chat-like data from a subreddit.")
    parser.add_argument("subreddit", help="Subreddit name, e.g. 'learnpython'")
    parser.add_argument("--posts", type=int, default=100, help="Number of hot posts to pull")
    parser.add_argument("--comments", type=int, default=50, help="Top-level comments per post")
    args = parser.parse_args()

    print(f"Pulling data from r/{args.subreddit} ...")
    data = fetch_subreddit_messages(args.subreddit, args.posts, args.comments)
    path = save_raw_data(data, args.subreddit)
    print(f"Saved {len(data)} messages to {path}")
