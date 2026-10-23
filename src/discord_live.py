"""
Live data source: pulls real, current message history from a Discord
channel via Discord's REST API — a second ingestion path alongside the
static dataset in src/ingest.py.

Why REST history instead of a persistent "gateway" bot?
discord.py's usual pattern runs a bot as a long-lived process listening for
new messages in real time (an event loop that never exits) — great for a
chatbot, overkill for an analysis script. Discord's REST API also exposes
`GET /channels/{id}/messages`, which lets a short-lived script page
backwards through a channel's existing history and exit when done. Same
real, live data; much simpler to run as a one-shot pipeline step.

Requirements (the user must set this up themselves — there's no way around
needing credentials tied to a real account/server, this is a platform
security requirement, not a limitation of this code):
  1. Create a bot at https://discord.com/developers/applications
  2. Enable the "Message Content Intent" under the bot's settings
  3. Invite the bot to a server you have access to (OAuth2 URL Generator:
     scope "bot", permissions "View Channels" + "Read Message History")
  4. Put the bot token in .env as DISCORD_BOT_TOKEN
  5. Enable Developer Mode in Discord (User Settings > Advanced), then
     right-click the channel you want to analyze > "Copy Channel ID"
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://discord.com/api/v10"


def get_bot_token() -> str:
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise EnvironmentError(
            "Missing DISCORD_BOT_TOKEN. Copy .env.example to .env and fill it in "
            "(see src/discord_live.py docstring for how to create a bot)."
        )
    return token


def fetch_channel_messages(channel_id: str, limit: int = 500) -> pd.DataFrame:
    """
    Pages backwards through a Discord channel's message history via the
    REST API (100 messages per request, Discord's max) until `limit`
    messages have been collected or the channel runs out of history.

    Returns a DataFrame with the same message schema used elsewhere in the
    pipeline: message_id, author, text, created_utc.
    """
    token = get_bot_token()
    headers = {"Authorization": f"Bot {token}"}

    rows = []
    before = None

    while len(rows) < limit:
        params = {"limit": min(100, limit - len(rows))}
        if before:
            params["before"] = before

        response = requests.get(
            f"{API_BASE}/channels/{channel_id}/messages", headers=headers, params=params
        )

        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 1)
            time.sleep(retry_after)
            continue

        response.raise_for_status()
        batch = response.json()

        if not batch:
            break  # reached the beginning of the channel's history

        for message in batch:
            created = datetime.fromisoformat(message["timestamp"]).astimezone(timezone.utc)
            rows.append(
                {
                    "message_id": message["id"],
                    "author": message["author"]["username"],
                    "text": message.get("content", ""),
                    "created_utc": created.timestamp(),
                }
            )

        before = batch[-1]["id"]
        time.sleep(0.5)  # stay well under Discord's rate limits

    return pd.DataFrame(rows)


def save_raw_data(df: pd.DataFrame, channel_id: str) -> str:
    out_path = f"data/raw/discord_live_{channel_id}_raw.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pull live message history from a Discord channel you have access to."
    )
    parser.add_argument("channel_id", help="Discord channel ID (enable Developer Mode to copy it)")
    parser.add_argument("--limit", type=int, default=500, help="Max messages to pull")
    args = parser.parse_args()

    print(f"Pulling up to {args.limit} messages from channel {args.channel_id} ...")
    data = fetch_channel_messages(args.channel_id, args.limit)
    path = save_raw_data(data, args.channel_id)
    print(f"Saved {len(data)} messages to {path}")
