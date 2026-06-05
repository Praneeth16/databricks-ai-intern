"""Helpers for session-scoped dataset uploads to a Unity Catalog Volume.

Databricks-native adaptation of huggingface/ml-intern#255. Instead of
pushing the uploaded file to a private HF Hub dataset repo we write it to
``/Volumes/<cat>/<schema>/<vol>/sessions/<session_id>/<upload_id>/<file>``
via ``WorkspaceClient.files.upload``. The path is plumbed back to the
agent through a context note so the LLM can read it via the
``uc_volume_read`` or ``uc_inspect_dataset`` tools without the user having
to repeat the path.

OBO required: the upload writes as the human user (the Apps proxy
forwards the access token) so the audit log names the right principal.
A missing OBO token returns a 401 — we never fall back to the App SP for
user data writes.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024
# Spark + Pandas handle these without extra deps. Parquet stays out of the
# allow-list for now because the frontend snippet writer doesn't have a
# clean "show me 5 rows" path that doesn't require pyarrow on the client.
ALLOWED_DATASET_EXTENSIONS = {"csv", "json", "jsonl", "parquet"}
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class DatasetUpload:
    """One uploaded file's metadata + payload reference."""

    session_id: str
    upload_id: str
    filename: str
    original_filename: str
    volume_path: str
    size_bytes: int
    format: str
    read_snippet: str

    def response_payload(self) -> dict[str, str | int | bool]:
        return {
            "session_id": self.session_id,
            "upload_id": self.upload_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "volume_path": self.volume_path,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "read_snippet": self.read_snippet,
        }


def sanitize_dataset_filename(filename: str | None) -> str:
    """Return a UC-Volume-safe basename while preserving the extension.

    UC Volumes accept Unix-style names; we still strip path traversal,
    non-portable characters, and overly long basenames so a hostile
    filename can't blow past the path-length budget upstream APIs cap at.
    """
    raw = os.path.basename(filename or "").strip()
    if not raw:
        raw = "dataset.csv"
    safe = _SAFE_FILENAME_RE.sub("-", raw).strip(".-_")
    if not safe:
        safe = "dataset.csv"
    stem, ext = os.path.splitext(safe)
    if not stem:
        stem = "dataset"
    if not ext:
        ext = ".csv"
    max_stem_len = 96 - len(ext)
    stem = stem[:max_stem_len].strip(".-_") or "dataset"
    return f"{stem}{ext.lower()}"


def display_filename(filename: str | None, fallback: str) -> str:
    """Best-effort preserve the user's original basename for UI display."""
    raw = os.path.basename(filename or "").strip()
    if not raw:
        return fallback
    cleaned = "".join(c for c in raw if ord(c) >= 32)
    return cleaned[:160] or fallback


def dataset_format_from_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .csv, .json, .jsonl, and .parquet dataset files are supported.",
        )
    return ext


async def upload_size_bytes(upload: UploadFile) -> int:
    """Return the byte length of the file body without consuming it."""
    await asyncio.to_thread(upload.file.seek, 0, os.SEEK_END)
    size = await asyncio.to_thread(upload.file.tell)
    await asyncio.to_thread(upload.file.seek, 0)
    return int(size)


async def validate_dataset_upload(upload: UploadFile) -> tuple[str, str, int]:
    dataset_format = dataset_format_from_filename(upload.filename or "")
    safe_filename = sanitize_dataset_filename(upload.filename)
    size = await upload_size_bytes(upload)
    if size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded dataset file is empty.")
    if size > MAX_DATASET_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset upload exceeds the 100 MB limit.",
        )
    return safe_filename, dataset_format, size


def session_volume_dir(volume_base: str, session_id: str, upload_id: str) -> str:
    """Build the per-session, per-upload directory inside the configured
    catalog/schema/volume. The two-level layout (session_id then upload_id)
    keeps a session's uploads grouped while still letting two uploads in
    the same session avoid filename collisions.
    """
    safe_session_id = re.sub(r"[^A-Za-z0-9]+", "-", session_id).strip("-")
    if not safe_session_id:
        safe_session_id = uuid.uuid4().hex[:8]
    safe_session_id = safe_session_id[:32]
    return f"{volume_base}/sessions/{safe_session_id}/{upload_id}"


def read_snippet(volume_path: str, fmt: str) -> str:
    """Return the snippet the agent or a user would run to read the file.

    Format-aware so the snippet is "do the right thing" rather than a
    generic stub. Spark is the right path inside Databricks notebooks for
    csv/parquet (UC permissions plug in); jsonl is line-by-line.
    """
    if fmt == "csv":
        return (
            f'df = spark.read.option("header", True).csv("{volume_path}")\n'
            "df.show(5)"
        )
    if fmt == "parquet":
        return (
            f'df = spark.read.parquet("{volume_path}")\n'
            "df.show(5)"
        )
    if fmt == "jsonl":
        return (
            "import json\n"
            f'with open("{volume_path}", "r") as f:\n'
            "    rows = [json.loads(line) for line in f if line.strip()]\n"
            "print(len(rows), \"rows\"); print(rows[:2])"
        )
    # csv default already returned above; this is the .json branch.
    return (
        "import json\n"
        f'with open("{volume_path}", "r") as f:\n'
        "    data = json.load(f)\n"
        "print(type(data).__name__)\n"
    )


def dataset_context_note(upload: DatasetUpload) -> str:
    """Inline system note added to the agent's context so the next turn
    sees the upload without the user having to paste the path."""
    return f"""[SYSTEM: The user uploaded a dataset file for this session.

Use this Unity Catalog Volume reference when the task needs the uploaded data.
Do not look for the uploaded file elsewhere on local disk and do not ask the
user to re-upload it unless this path is rejected by a downstream tool.

- Volume path: {upload.volume_path}
- Original filename: {upload.original_filename}
- Stored filename: {upload.filename}
- Format: {upload.format}
- Size: {upload.size_bytes} bytes

Read it with:
```python
{upload.read_snippet}
```
]"""


async def push_dataset_upload_to_volume(
    *,
    upload: UploadFile,
    session_id: str,
    volume_base: str,
    wc,
) -> DatasetUpload:
    """Push the uploaded file body to UC Volumes via ``wc.files.upload``.

    ``wc`` is the per-request workspace client built from the OBO token so
    the write is audited as the human user, not the App SP. Caller wires
    that — see ``backend/routes/agent.py``'s upload handler.

    Returns the canonical metadata block, which the route serialises into
    the response payload + adds to the agent context as a system note.
    """
    safe_filename, dataset_format, size = await validate_dataset_upload(upload)
    original_filename = display_filename(upload.filename, safe_filename)
    upload_id = uuid.uuid4().hex[:12]
    dir_path = session_volume_dir(volume_base, session_id, upload_id)
    volume_path = f"{dir_path}/{safe_filename}"
    snippet = read_snippet(volume_path, dataset_format)

    await asyncio.to_thread(upload.file.seek, 0)
    file_bytes = await asyncio.to_thread(upload.file.read)

    try:
        await asyncio.to_thread(
            wc.files.create_directory, directory_path=dir_path,
        )
    except Exception as e:
        # create_directory raises when the parent doesn't yet exist or
        # permissions are off. We try the upload anyway — wc.files.upload
        # on most Databricks runtimes creates intermediate directories
        # on demand. If that ALSO fails we surface a clean 502 below.
        logger.debug("create_directory(%s) suppressed: %s", dir_path, e)

    try:
        await asyncio.to_thread(
            wc.files.upload,
            file_path=volume_path,
            contents=io.BytesIO(file_bytes),
            overwrite=False,
        )
    except Exception as e:
        logger.warning("UC volume upload failed for %s: %s", volume_path, e)
        raise HTTPException(
            status_code=502,
            detail=f"UC Volume upload failed: {e}",
        ) from e

    return DatasetUpload(
        session_id=session_id,
        upload_id=upload_id,
        filename=safe_filename,
        original_filename=original_filename,
        volume_path=volume_path,
        size_bytes=size,
        format=dataset_format,
        read_snippet=snippet,
    )
