"""Pydantic guards on SubmitRequest (issue #5).

100k-char cap stops a runaway / malicious client from attaching megabytes
of text that would then ride along in every subsequent turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from models import SubmitRequest  # noqa: E402


def test_submit_request_accepts_normal_text():
    req = SubmitRequest(session_id="s1", text="hello world")
    assert req.text == "hello world"


def test_submit_request_rejects_empty_text():
    with pytest.raises(ValidationError) as exc:
        SubmitRequest(session_id="s1", text="")
    assert "at least 1 character" in str(exc.value) or "min_length" in str(exc.value)


def test_submit_request_rejects_text_over_100k_chars():
    big = "x" * 100_001
    with pytest.raises(ValidationError) as exc:
        SubmitRequest(session_id="s1", text=big)
    msg = str(exc.value)
    assert "100000" in msg or "max_length" in msg or "at most" in msg


def test_submit_request_accepts_text_exactly_at_cap():
    body = "x" * 100_000
    req = SubmitRequest(session_id="s1", text=body)
    assert len(req.text) == 100_000
