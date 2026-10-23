from src.ingest import _parse_block


def test_parse_block_splits_simple_messages():
    block = "alice#1111: hello there\nbob#2222: hi alice!"
    rows = []
    next_seq = _parse_block(block, rows, start_seq=0)

    assert next_seq == 2
    assert len(rows) == 2
    assert rows[0]["author"] == "alice#1111"
    assert rows[0]["text"] == "hello there"
    assert rows[1]["author"] == "bob#2222"
    assert rows[1]["text"] == "hi alice!"


def test_parse_block_appends_continuation_lines_to_previous_message():
    block = "alice#1111: here is some code\ndef foo():\n    return 1"
    rows = []
    _parse_block(block, rows, start_seq=0)

    assert len(rows) == 1
    assert rows[0]["author"] == "alice#1111"
    assert "here is some code" in rows[0]["text"]
    assert "def foo" in rows[0]["text"]


def test_parse_block_assigns_sequential_seq_index_across_calls():
    rows = []
    next_seq = _parse_block("alice#1111: first\nbob#2222: second", rows, start_seq=5)
    _parse_block("carol#3333: third", rows, start_seq=next_seq)

    assert [r["seq_index"] for r in rows] == [5, 6, 7]
