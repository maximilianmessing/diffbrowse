"""Tests for action space flattening and single-token label pool."""

from unittest.mock import Mock

from jev_ultrafast.action_candidates import LabelPool, flatten_actions


def sample_actions():
    return [
        {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
        {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
        {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]


def test_flatten_actions_produces_grounded_candidates():
    actions = sample_actions()
    candidates = flatten_actions(actions)
    operations = [c.operation for c in candidates]

    assert "CLICK" in operations
    assert "TYPE_TEXT" in operations
    assert "DONE" in operations
    assert "BLOCKED" in operations

    # e1 should map to TYPE_TEXT
    e1_cand = next(c for c in candidates if c.action_id == "e1")
    assert e1_cand.operation == "TYPE_TEXT"
    assert "textbox" in e1_cand.label

    # e3 should map to CLICK
    e3_cand = next(c for c in candidates if c.action_id == "e3")
    assert e3_cand.operation == "CLICK"
    assert "Go" in e3_cand.label

    # Terminal actions exist
    assert any(c.action_id == "DONE" for c in candidates)
    assert any(c.action_id == "BLOCKED" for c in candidates)


def test_label_pool_with_mock_tokenizer():
    vocab = {chr(i): i for i in range(ord("A"), ord("Z") + 1)}
    vocab.update({chr(i): i for i in range(ord("a"), ord("z") + 1)})
    vocab.update({chr(i): i for i in range(ord("0"), ord("9") + 1)})
    for idx, w in enumerate(["alpha", "beta", "gamma", "delta", "click", "wait", "done", "next"]):
        vocab[w] = 200 + idx
    inv_vocab = {v: k for k, v in vocab.items()}

    def encode(text, add_special_tokens=False):
        return [vocab[text]] if text in vocab else [999]

    def decode(ids):
        return "".join(inv_vocab.get(i, "") for i in ids)

    mock_tok = Mock()
    mock_tok.encode = encode
    mock_tok.decode = decode
    mock_tok.all_special_ids = [0, 1, 2]
    mock_tok.vocab_size = 300

    pool = LabelPool(mock_tok)
    labels = pool.get_pool(strategy="index", size=60)
    assert len(labels) >= 60

    # Ensure all labels roundtrip cleanly
    for label_str, token_id in labels:
        assert len(mock_tok.encode(label_str)) == 1
        assert mock_tok.decode([token_id]) == label_str


def test_map_candidates_formats_prompt():
    actions = sample_actions()
    candidates = flatten_actions(actions)

    vocab = {chr(i): i for i in range(ord("A"), ord("Z") + 1)}
    vocab.update({chr(i): i for i in range(ord("a"), ord("z") + 1)})
    vocab.update({chr(i): i for i in range(ord("0"), ord("9") + 1)})
    for i in range(300):
        vocab[f"tok{i}"] = 1000 + i
    inv_vocab = {v: k for k, v in vocab.items()}

    def encode(text, add_special_tokens=False):
        return [vocab[text]] if text in vocab else [9999]

    def decode(ids):
        return "".join(inv_vocab.get(i, "") for i in ids)

    mock_tok = Mock(encode=encode, decode=decode, all_special_ids=[0, 1], vocab_size=1500)
    pool = LabelPool(mock_tok)
    cand_ids, tok_map, cand_map, prompt_sec = pool.map_candidates(candidates, strategy="index")

    assert len(cand_ids) == len(candidates)
    assert len(tok_map) == len(candidates)
    assert "Executable Actions:" in prompt_sec
    assert "[A]" in prompt_sec
