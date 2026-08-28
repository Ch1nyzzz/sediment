from sediment.semantic import action_semantic_mask, is_semantic_token


class Tok:
    """Whitespace/punct-splitting fake tokenizer: id = index into a vocab list."""
    def __init__(self):
        self.vocab: list[str] = []

    def encode(self, text, add_special_tokens=False):
        import re
        ids = []
        for piece in re.findall(r"<\|im_start\|>|<\|im_end\|>|<tool_call>|</tool_call>|[A-Za-z0-9_\-]+|[^\sA-Za-z0-9]", text):
            if piece not in self.vocab:
                self.vocab.append(piece)
            ids.append(self.vocab.index(piece))
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def test_mask_keeps_values_drops_framing_keys_and_tool_name():
    tok = Tok()
    action = '<tool_call>\n{"name": "get_user", "arguments": {"user_id": "USR-2B"}}\n</tool_call>'
    ids = tok.encode(action)
    mask = action_semantic_mask(tok, action, ids)
    kept = [tok.decode([i]) for i, k in zip(ids, mask) if k]
    assert kept == ["get_user", "USR-2B"]  # tool name eligible, keys/framing not
    assert not is_semantic_token("<tool_call>") and not is_semantic_token('"name"')
    assert is_semantic_token("premium")


def test_trainer_binds_semantic_mask():
    import sediment.trainer as t
    assert t.action_semantic_mask is action_semantic_mask


def test_framing_mask_drops_template_tokens():
    from sediment.semantic import framing_mask
    tok = Tok()
    ids = tok.encode("<|im_start|> assistant \n <tool_call> premium USR-2B")
    keep = [tok.decode([i]) for i, k in zip(ids, framing_mask(tok, ids)) if k]
    assert keep == ["premium", "USR-2B"]


def test_tool_call_region_mask():
    from sediment.semantic import tool_call_region_mask
    tok = Tok()
    content = 'Task Completed <tool_call> {"name": "upgrade", "arguments": {"user": "Alice"}} </tool_call> done'
    ids = tok.encode(content)
    inside = [tok.decode([i]) for i, k in zip(ids, tool_call_region_mask(tok, content, ids)) if k]
    assert "Alice" in inside and "upgrade" in inside
    assert "Task" not in inside and "Completed" not in inside and "done" not in inside
