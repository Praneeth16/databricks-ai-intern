"""Tests for backend.uc_volume_uploads (issue #12).

Coverage:
  * sanitize_dataset_filename: traversal / unsafe chars stripped, extension lower-cased.
  * dataset_format_from_filename: rejects unsupported extensions with 400.
  * session_volume_dir: lays out per-session, per-upload subdirs predictably.
  * read_snippet: format-aware (csv → spark, jsonl → json line read, etc).
  * validate_dataset_upload: empty / oversized rejected with 400 / 413.
  * push_dataset_upload_to_volume: happy path writes via wc.files.upload to
    the resolved volume_path; create_directory failure does NOT block upload;
    upload failure surfaces as 502.
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

# Backend module isn't on the package import path by default — sibling-load it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from uc_volume_uploads import (  # noqa: E402
    DatasetUpload,
    MAX_DATASET_UPLOAD_BYTES,
    dataset_format_from_filename,
    push_dataset_upload_to_volume,
    read_snippet,
    sanitize_dataset_filename,
    session_volume_dir,
    validate_dataset_upload,
)


# ── sanitize_dataset_filename ──────────────────────────────────────────


def test_sanitize_strips_traversal_and_unsafe_chars():
    assert sanitize_dataset_filename("../../etc/passwd.csv").endswith("passwd.csv")
    assert sanitize_dataset_filename("data;rm -rf /.csv").endswith(".csv")
    # Spaces and special chars collapse to dashes.
    out = sanitize_dataset_filename("my file (1).csv")
    assert " " not in out
    assert "(" not in out
    assert out.endswith(".csv")


def test_sanitize_empty_falls_back_to_dataset_csv():
    assert sanitize_dataset_filename(None) == "dataset.csv"
    assert sanitize_dataset_filename("") == "dataset.csv"
    assert sanitize_dataset_filename("   ") == "dataset.csv"


def test_sanitize_lowercases_extension():
    assert sanitize_dataset_filename("DATA.CSV").endswith(".csv")
    assert sanitize_dataset_filename("File.JSONL").endswith(".jsonl")


def test_sanitize_caps_stem_length():
    out = sanitize_dataset_filename("a" * 200 + ".csv")
    assert len(out) <= 100  # 96 stem + 4 ext
    assert out.endswith(".csv")


# ── dataset_format_from_filename ────────────────────────────────────────


def test_dataset_format_accepts_supported_extensions():
    assert dataset_format_from_filename("a.csv") == "csv"
    assert dataset_format_from_filename("a.json") == "json"
    assert dataset_format_from_filename("a.JSONL") == "jsonl"
    assert dataset_format_from_filename("a.parquet") == "parquet"


def test_dataset_format_rejects_unsupported():
    with pytest.raises(HTTPException) as exc:
        dataset_format_from_filename("malware.exe")
    assert exc.value.status_code == 400


# ── session_volume_dir ──────────────────────────────────────────────────


def test_session_volume_dir_layout():
    out = session_volume_dir(
        "/Volumes/cat/schema/vol", "sess-abc-123", "upload42"
    )
    assert out == "/Volumes/cat/schema/vol/sessions/sess-abc-123/upload42"


def test_session_volume_dir_strips_unsafe_session_id():
    # Embedded path traversal in the session id must be sanitised — the
    # caller-controlled session_id should never let a write escape the
    # configured volume_base prefix.
    out = session_volume_dir(
        "/Volumes/cat/schema/vol", "../../bad/path", "u1",
    )
    assert "../" not in out
    assert out.startswith("/Volumes/cat/schema/vol/sessions/")


# ── read_snippet ────────────────────────────────────────────────────────


def test_read_snippet_csv_uses_spark():
    snip = read_snippet("/Volumes/x/y/z/file.csv", "csv")
    assert "spark.read" in snip
    assert "/Volumes/x/y/z/file.csv" in snip


def test_read_snippet_parquet_uses_spark():
    snip = read_snippet("/Volumes/x/y/z/file.parquet", "parquet")
    assert "spark.read.parquet" in snip


def test_read_snippet_jsonl_line_iterates():
    snip = read_snippet("/Volumes/x/y/z/file.jsonl", "jsonl")
    assert "json.loads(line)" in snip


# ── validate_dataset_upload ─────────────────────────────────────────────


def _make_upload(content: bytes, filename: str = "data.csv"):
    """Build a Starlette-shaped UploadFile that ``validate_dataset_upload``
    can introspect (we only need .filename + .file with seek/tell/read).
    """
    upload = MagicMock()
    upload.filename = filename
    upload.file = io.BytesIO(content)
    return upload


@pytest.mark.asyncio
async def test_validate_rejects_empty_upload():
    upload = _make_upload(b"")
    with pytest.raises(HTTPException) as exc:
        await validate_dataset_upload(upload)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_rejects_oversized_upload():
    upload = _make_upload(b"x" * (MAX_DATASET_UPLOAD_BYTES + 1))
    with pytest.raises(HTTPException) as exc:
        await validate_dataset_upload(upload)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_validate_returns_safe_filename_and_size():
    upload = _make_upload(b"hello,world\n1,2\n", filename="My Data.CSV")
    safe_name, fmt, size = await validate_dataset_upload(upload)
    assert safe_name.endswith(".csv")
    assert fmt == "csv"
    assert size == len(b"hello,world\n1,2\n")


# ── push_dataset_upload_to_volume ──────────────────────────────────────


@pytest.mark.asyncio
async def test_push_writes_to_volume_path_and_returns_metadata():
    """Happy path: upload lands under the configured volume_base + the
    metadata block carries the canonical path."""
    upload = _make_upload(b"col1,col2\nA,B\n", filename="data.csv")
    wc = MagicMock()
    wc.files.create_directory.return_value = None
    wc.files.upload.return_value = None

    result = await push_dataset_upload_to_volume(
        upload=upload,
        session_id="sess-X",
        volume_base="/Volumes/cat/schema/vol",
        wc=wc,
    )

    assert isinstance(result, DatasetUpload)
    assert result.session_id == "sess-X"
    assert result.format == "csv"
    assert result.volume_path.startswith("/Volumes/cat/schema/vol/sessions/sess-X/")
    assert result.volume_path.endswith("/data.csv")
    # spark snippet contains the canonical path.
    assert result.volume_path in result.read_snippet

    # wc.files.upload was called exactly once with the expected path.
    assert wc.files.upload.call_count == 1
    kwargs = wc.files.upload.call_args.kwargs
    assert kwargs["file_path"] == result.volume_path
    assert kwargs["overwrite"] is False


@pytest.mark.asyncio
async def test_push_continues_when_create_directory_fails():
    """create_directory often fails when the path already exists or when
    the parent isn't yet provisioned — wc.files.upload creates intermediate
    dirs anyway. A create_directory exception must NOT block the actual
    upload."""
    upload = _make_upload(b"col1\n1\n", filename="d.csv")
    wc = MagicMock()
    wc.files.create_directory.side_effect = RuntimeError("dir already exists")
    wc.files.upload.return_value = None

    result = await push_dataset_upload_to_volume(
        upload=upload, session_id="s1",
        volume_base="/Volumes/cat/schema/vol", wc=wc,
    )
    assert result.volume_path.endswith("/d.csv")
    assert wc.files.upload.called


@pytest.mark.asyncio
async def test_push_surfaces_upload_failure_as_502():
    """A real upload failure (auth, network, permission) is surfaced to
    the caller as a clean HTTP 502 rather than bubbling the raw SDK
    exception into the route handler."""
    upload = _make_upload(b"x\n1\n", filename="d.csv")
    wc = MagicMock()
    wc.files.create_directory.return_value = None
    wc.files.upload.side_effect = RuntimeError("PERMISSION_DENIED")

    with pytest.raises(HTTPException) as exc:
        await push_dataset_upload_to_volume(
            upload=upload, session_id="s1",
            volume_base="/Volumes/cat/schema/vol", wc=wc,
        )
    assert exc.value.status_code == 502
