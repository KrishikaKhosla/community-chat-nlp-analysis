from unittest.mock import MagicMock, patch

from src.discord_live import fetch_channel_messages


def _mock_message(msg_id, author, content, timestamp):
    return {
        "id": msg_id,
        "author": {"username": author},
        "content": content,
        "timestamp": timestamp,
    }


@patch("src.discord_live.get_bot_token", return_value="fake-token")
@patch("src.discord_live.requests.get")
def test_fetch_channel_messages_stops_at_limit(mock_get, mock_token):
    batch = [
        _mock_message(str(i), "alice", f"message {i}", "2024-01-01T12:00:00+00:00")
        for i in range(100)
    ]
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = batch
    response.raise_for_status.return_value = None
    mock_get.return_value = response

    result = fetch_channel_messages("123456", limit=100)

    assert len(result) == 100
    assert set(result.columns) >= {"message_id", "author", "text", "created_utc"}
    assert result.iloc[0]["author"] == "alice"


@patch("src.discord_live.get_bot_token", return_value="fake-token")
@patch("src.discord_live.requests.get")
def test_fetch_channel_messages_stops_when_history_runs_out(mock_get, mock_token):
    first_batch = [
        _mock_message(str(i), "bob", f"message {i}", "2024-01-01T12:00:00+00:00")
        for i in range(10)
    ]
    empty_response = MagicMock()
    empty_response.status_code = 200
    empty_response.json.return_value = []
    empty_response.raise_for_status.return_value = None

    first_response = MagicMock()
    first_response.status_code = 200
    first_response.json.return_value = first_batch
    first_response.raise_for_status.return_value = None

    mock_get.side_effect = [first_response, empty_response]

    result = fetch_channel_messages("123456", limit=500)

    assert len(result) == 10
