import pandas as pd

from src.sentiment import score_sentiment


def _make_df(texts):
    return pd.DataFrame(
        {
            "message_id": [str(i) for i in range(len(texts))],
            "text_for_sentiment": texts,
        }
    )


def test_positive_text_scores_positive():
    df = _make_df(["This is absolutely wonderful, I love it!"])
    result = score_sentiment(df)
    assert result.iloc[0]["sentiment_label"] == "positive"
    assert result.iloc[0]["sentiment_compound"] > 0


def test_negative_text_scores_negative():
    df = _make_df(["This is terrible, I hate it, worst thing ever."])
    result = score_sentiment(df)
    assert result.iloc[0]["sentiment_label"] == "negative"
    assert result.iloc[0]["sentiment_compound"] < 0


def test_neutral_text_scores_neutral():
    df = _make_df(["The meeting is scheduled for 3pm on Tuesday."])
    result = score_sentiment(df)
    assert result.iloc[0]["sentiment_label"] == "neutral"


def test_output_has_expected_columns():
    df = _make_df(["Some text here"])
    result = score_sentiment(df)
    for col in [
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
        "sentiment_label",
    ]:
        assert col in result.columns
