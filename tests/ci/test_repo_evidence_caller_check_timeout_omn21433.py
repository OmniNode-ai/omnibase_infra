# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: the repo-evidence caller gives each dod_verify check a named ceiling.

The verifier's default ceiling is 30 s per check. A hosted runner starts every
``uv run pytest`` cold, so a check that finishes in seconds on a lab host is killed and
recorded unverifiable there (omnibase_infra#5448: five pull-request heads, each
red on a different test-passing check). The reusable takes the ceiling as an input
since omnibase_core#2085; evidence-autoclose-sweep.yml already names 180 s.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CALLER_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "call-repo-evidence-gate.yml"
)

CORE = "OmniNode-ai/omnibase_core"
RECEIPT_GATE_PATH = ".github/workflows/receipt-gate.yml"
INPUT_NAME = "check-timeout-seconds"

_VERIFIER_DEFAULT_SECONDS = 30
_RECEIPT_GATE_MAXIMUM_SECONDS = 900


def _job() -> dict[str, Any]:
    caller = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    job: dict[str, Any] = caller["jobs"]["repo-evidence"]
    return job


def _gh_core(endpoint: str, *flags: str) -> str:
    """Read one omnibase_core API endpoint; skip when the environment cannot reach it."""
    try:
        done = subprocess.run(
            ["gh", "api", f"repos/{CORE}/{endpoint}", *flags],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        done = None
    if done is None or done.returncode != 0:
        pytest.skip(f"omnibase_core {endpoint} is not readable from this environment")
    return done.stdout


def test_caller_pins_a_reusable_that_declares_the_check_timeout_input() -> None:
    """Read the reusable at the pinned sha and find the input under workflow_call.

    A caller passing an input the reusable does not declare is refused by GitHub
    before the job starts, so the declaration itself is what this reads.
    """
    pinned_sha = _job()["uses"].rsplit("@", 1)[1]
    text = _gh_core(
        f"contents/{RECEIPT_GATE_PATH}?ref={pinned_sha}",
        "-H",
        "Accept: application/vnd.github.raw",
    )
    reusable = yaml.safe_load(text)
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = reusable.get("on", reusable.get(True))
    declared = triggers["workflow_call"]["inputs"]
    assert INPUT_NAME in declared, sorted(declared)


def test_caller_names_a_ceiling_above_the_verifier_default() -> None:
    value = _job()["with"].get("check-timeout-seconds")
    assert isinstance(value, str), "the reusable's inputs are strings; quote the number"
    assert re.fullmatch(r"[1-9][0-9]{0,2}", value), value
    assert _VERIFIER_DEFAULT_SECONDS < int(value) <= _RECEIPT_GATE_MAXIMUM_SECONDS


def test_caller_sha_pin_is_the_commit_of_the_dev_tag_named_on_the_same_line() -> None:
    """A pull_request_target caller pins by 40-hex sha; the dev tag is named beside it.

    The tag named in the comment is resolved through the GitHub API and must point at
    the pinned sha, so a mispaired tag and sha fail here.
    """
    line = next(
        raw
        for raw in CALLER_PATH.read_text(encoding="utf-8").splitlines()
        if "uses:" in raw and "receipt-gate.yml@" in raw
    )
    pinned_sha = _job()["uses"].rsplit("@", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{40}", pinned_sha), pinned_sha
    tags = re.findall(r"dev-\d{14}-[0-9a-f]{12}", line.split("#", 1)[1])
    assert len(tags) == 1, line
    resolved = _gh_core(f"commits/{tags[0]}", "--jq", ".sha").strip()
    assert resolved == pinned_sha, (tags[0], resolved, pinned_sha)
