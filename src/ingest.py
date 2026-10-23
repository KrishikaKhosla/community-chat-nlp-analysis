"""
Data ingestion: loads real community chat data from a public Hugging Face
dataset — "breadlicker45/discord-chat", a dump of the EleutherAI Discord
server (the open-source AI research community behind GPT-Neo/GPT-J).

Why this source instead of live-scraping Reddit?
Reddit's public endpoints are unreliable to reach from many cloud/CI
environments (aggressive bot blocking on datacenter IPs), and the official
API requires registering a developer app. Hugging Face's dataset hub serves
real, publicly-shared chat logs over plain HTTPS with no auth, no API key,
and no rate-limit dance — a much more reproducible source for a pipeline
other people (e.g. an interviewer) will actually want to run.

Each row in the raw dataset is a *block* of consecutive chat lines in the
form "Username#1234: message text". This module parses those blocks into
individual messages. There are no real timestamps in this dataset, so a
sequential `seq_index` (the message's position in server history, since the
dataset is chronologically ordered) stands in as a time axis — later stages
treat it as "conversation order" rather than wall-clock time.
"""

import re

import pandas as pd

DATASET_PARQUET_URL = (
    "https://huggingface.co/datasets/breadlicker45/discord-chat/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

# Matches lines like "SomeUser#1234: message text"
MESSAGE_LINE_PATTERN = re.compile(r"^(.+?#\d{2,6}): ?(.*)$")


def _parse_block(block: str, rows: list, start_seq: int) -> int:
    """
    Parses one raw text block into individual message rows, appended to
    `rows`. Lines that don't start a new "Author#tag: " message are treated
    as a continuation of the previous message (multi-line messages).
    Returns the next available seq_index.
    """
    seq = start_seq
    current = None

    for line in block.split("\n"):
        if not line.strip():
            continue
        match = MESSAGE_LINE_PATTERN.match(line)
        if match:
            if current is not None:
                rows.append(current)
            author, text = match.group(1), match.group(2)
            current = {
                "message_id": f"m{seq}",
                "author": author,
                "text": text,
                "seq_index": seq,
            }
            seq += 1
        elif current is not None:
            current["text"] += " " + line.strip()

    if current is not None:
        rows.append(current)

    return seq


def fetch_discord_messages(num_blocks: int = 400) -> pd.DataFrame:
    """
    Downloads the dataset's parquet file and parses the first `num_blocks`
    raw chat blocks into a flat DataFrame of individual messages:
        message_id, author, text, seq_index
    """
    raw = pd.read_parquet(DATASET_PARQUET_URL)
    blocks = raw.iloc[:num_blocks]["data"]

    rows = []
    seq = 0
    for block in blocks:
        seq = _parse_block(block, rows, seq)

    return pd.DataFrame(rows)


def save_raw_data(df: pd.DataFrame, name: str = "discord_chat") -> str:
    """Saves the raw parsed data to data/raw/ and returns the file path."""
    out_path = f"data/raw/{name}_raw.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and parse chat data from the EleutherAI Discord dataset."
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=400,
        help="Number of raw chat blocks to parse (each block ~15-20 messages)",
    )
    args = parser.parse_args()

    print(f"Loading {args.blocks} chat blocks from the dataset...")
    data = fetch_discord_messages(args.blocks)
    path = save_raw_data(data)
    print(f"Parsed {len(data)} messages -> {path}")
