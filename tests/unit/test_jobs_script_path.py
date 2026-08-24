"""`script_path` submission, and the placeholder guard.

Both come from one live failure. Told to author a long training script, the agent wrote
it to /tmp in chunks (the right move — composing it in one reply exhausts the output
token budget), then had no way to hand a local file to databricks_jobs. It submitted
`script: "PLACEHOLDER"` instead, which staged cleanly and ran, wasting cluster time.
"""

from __future__ import annotations

import pytest

from agent.tools.databricks_jobs_tool import _load_script_text

REAL = "import lightgbm as lgb\nprint('training')\n" * 3


def test_inline_script_passes_through():
    assert _load_script_text({"script": REAL}) == REAL


def test_reads_a_local_script_path(tmp_path):
    f = tmp_path / "train.py"
    f.write_text(REAL)
    assert _load_script_text({"script_path": str(f)}) == REAL


def test_script_path_sets_the_staged_filename(tmp_path):
    f = tmp_path / "my_trainer.py"
    f.write_text(REAL)
    args = {"script_path": str(f)}
    _load_script_text(args)
    assert args["filename"] == "my_trainer.py"


def test_explicit_filename_is_not_overridden(tmp_path):
    f = tmp_path / "my_trainer.py"
    f.write_text(REAL)
    args = {"script_path": str(f), "filename": "chosen.py"}
    _load_script_text(args)
    assert args["filename"] == "chosen.py"


def test_inline_script_wins_over_script_path(tmp_path):
    f = tmp_path / "train.py"
    f.write_text("print('from file')\n" * 5)
    assert _load_script_text({"script": REAL, "script_path": str(f)}) == REAL


def test_missing_file_is_a_clear_error():
    with pytest.raises(ValueError, match="not a readable file"):
        _load_script_text({"script_path": "/tmp/definitely-not-here-9f3a.py"})


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a readable file"):
        _load_script_text({"script_path": str(tmp_path)})


def test_nothing_supplied_returns_none():
    """`_run` turns this into the 'provide one of ...' error; the loader stays quiet."""
    assert _load_script_text({}) is None


def test_empty_file_is_rejected(tmp_path):
    f = tmp_path / "train.py"
    f.write_text("   \n\n")
    with pytest.raises(ValueError, match="empty|placeholder"):
        _load_script_text({"script_path": str(f)})


@pytest.mark.parametrize(
    "stub",
    ["PLACEHOLDER", "PLACEHOLDER...", "placeholder", "pass", "# TODO", "...", "TBD", "# FIXME"],
)
def test_placeholder_scripts_are_refused(stub):
    """The real incident: a staged 'PLACEHOLDER...' ran as a job."""
    with pytest.raises(ValueError, match="placeholder"):
        _load_script_text({"script": stub})


@pytest.mark.parametrize(
    "real",
    [
        "print('hi')",                                   # short but legitimate
        "x = 1",
        "# placeholder values are filled below\nrun()",   # the word, but not a stub
        "print('TODO: nothing, this actually runs')",
    ],
)
def test_guard_does_not_reject_real_scripts(real):
    """Length is not the signal — a character minimum would reject these."""
    assert _load_script_text({"script": real}) == real


def test_tool_schema_advertises_script_path():
    from agent.tools.databricks_jobs_tool import DATABRICKS_JOBS_TOOL_SPEC

    props = DATABRICKS_JOBS_TOOL_SPEC["parameters"]["properties"]
    assert "script_path" in props, "the agent cannot use what is not in the schema"
    desc = props["script_path"]["description"].lower()
    assert "local" in desc and "token" in desc
