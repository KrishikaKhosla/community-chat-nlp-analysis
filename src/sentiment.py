"""
Sentiment scoring using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Why VADER and not a transformer model (e.g. BERT)?
VADER is a lexicon + rule-based model specifically tuned for short, informal,
social-media-style text — it understands things like capitalization ("GREAT"
vs "great"), punctuation emphasis ("good!!!"), negation ("not good"), and
emoji/emoticons, without needing training data or a GPU. That's exactly the
kind of text found in chat communities, and it's fast enough to score
thousands of messages in seconds. A transformer model would likely be more
accurate on longer, more nuanced text, but is overkill (and much slower) for
this use case.
"""

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def ensure_vader_lexicon():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def score_sentiment(df: pd.DataFrame, text_column: str = "text_for_sentiment") -> pd.DataFrame:
    """
    Adds VADER sentiment scores to each row:
      - sentiment_compound: overall score from -1 (most negative) to +1 (most positive)
      - sentiment_label: 'positive' / 'neutral' / 'negative' bucket, using VADER's
        own recommended compound-score thresholds (>= 0.05 positive, <= -0.05 negative)
    """
    ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()

    scored = df.copy()
    scores = scored[text_column].apply(analyzer.polarity_scores)

    scored["sentiment_compound"] = scores.apply(lambda s: s["compound"])
    scored["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    scored["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    scored["sentiment_neg"] = scores.apply(lambda s: s["neg"])

    def label(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    scored["sentiment_label"] = scored["sentiment_compound"].apply(label)
    return scored


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score sentiment on cleaned message data.")
    parser.add_argument("clean_csv", help="Path to cleaned CSV from clean.py")
    parser.add_argument("--out", default=None, help="Output path for scored CSV")
    args = parser.parse_args()

    clean_df = pd.read_csv(args.clean_csv)
    scored_df = score_sentiment(clean_df)

    out_path = args.out or args.clean_csv.replace("_clean", "_scored")
    scored_df.to_csv(out_path, index=False)
    print(f"Scored {len(scored_df)} messages -> {out_path}")
    print(scored_df["sentiment_label"].value_counts())
