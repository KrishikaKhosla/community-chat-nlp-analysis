import pandas as pd

from src.clean import basic_clean_text, clean_messages, tokenize_for_frequency


def test_basic_clean_text_strips_urls():
    text = "Check this out https://example.com/thing it's cool"
    result = basic_clean_text(text)
    assert "https://" not in result
    assert "example.com" not in result


def test_basic_clean_text_strips_discord_mentions():
    text = "Thanks <@123456789> and welcome to <#987654321>"
    result = basic_clean_text(text)
    assert "<@123456789>" not in result
    assert "<#987654321>" not in result


def test_basic_clean_text_preserves_case_and_punctuation():
    # VADER needs case/punctuation for sentiment signal, so basic_clean_text
    # must NOT lowercase or strip punctuation.
    text = "This is GREAT!!!"
    result = basic_clean_text(text)
    assert result == "This is GREAT!!!"


def test_tokenize_for_frequency_removes_stopwords_and_lowercases():
    stop_words = {"is", "the", "a"}
    tokens = tokenize_for_frequency("This is the best day", stop_words)
    assert "is" not in tokens
    assert "the" not in tokens
    assert "best" in tokens
    assert all(t == t.lower() for t in tokens)


def test_clean_messages_drops_deleted_system_and_duplicate_rows():
    df = pd.DataFrame(
        [
            {
                "message_id": "1",
                "author": "alice#1111",
                "text": "[deleted]",
                "seq_index": 0,
            },
            {
                "message_id": "2",
                "author": "bob#2222",
                "text": "Great thread, thanks for sharing!",
                "seq_index": 1,
            },
            {
                "message_id": "2",
                "author": "bob#2222",
                "text": "Great thread, thanks for sharing!",
                "seq_index": 1,
            },
            {
                "message_id": "3",
                "author": "carol#3333",
                "text": "Joined the server.",
                "seq_index": 2,
            },
        ]
    )

    cleaned = clean_messages(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["message_id"] == "2"
    assert "text_for_sentiment" in cleaned.columns
    assert "tokens" in cleaned.columns
    assert "segment" in cleaned.columns
    assert isinstance(cleaned.iloc[0]["tokens"], list)
