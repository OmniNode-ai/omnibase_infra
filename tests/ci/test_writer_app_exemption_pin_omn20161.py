# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20161 - occ-preflight and Receipt Gate callers must pin the writer-app exemption.

omnibase_core#1820 exempts the OCC writer app from occ-preflight and the
Receipt Gate only when the pin-only probe proves the producer's outcome.
Callers here pin those reusable workflows by sha, so the exemption reaches this
repository only once every pin moves. This test reads every such pin from the
workflows and follows it to the pinned workflow text.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CORE = "OmniNode-ai/omnibase_core"
EXPECTED_PIN = "52851458622f368c3b596c82bf810bc6acce1d5e"  # pragma: allowlist secret
WRITER_APP = "onexbot-occ-writer"
USES = re.compile(
    rf"uses:\s*{re.escape(CORE)}/\.github/workflows/"
    r"(?P<name>occ-preflight|receipt-gate)\.yml@(?P<ref>\S+)"
)


def _pins() -> list[tuple[str, str, str]]:
    found = [
        (path.name, m["name"], m["ref"])
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        for m in USES.finditer(path.read_text())
    ]
    assert found, "no occ-preflight / receipt-gate caller found in .github/workflows"
    return found


def _fetch_core_file(ref: str, path: str) -> str:
    url = f"https://api.github.com/repos/{CORE}/contents/{path}?ref={ref}"
    try:
        done = subprocess.run(
            ["gh", "api", url, "-H", "Accept: application/vnd.github.raw"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        done = None
    if done is None or done.returncode != 0:
        pytest.skip(f"omnibase_core {path}@{ref} is not readable from this environment")
    return done.stdout


def test_every_caller_pin_is_the_writer_app_exemption_sha() -> None:
    stale = [(f, n, r) for f, n, r in _pins() if r != EXPECTED_PIN]
    assert not stale, f"pins not at {EXPECTED_PIN}: {stale}"


def test_the_expected_pin_carries_both_reusable_workflows() -> None:
    assert {n for _, n, _ in _pins()} == {"occ-preflight", "receipt-gate"}


@pytest.mark.parametrize("name", ["occ-preflight", "receipt-gate"])
def test_pinned_workflow_contains_the_writer_app_pin_only_exemption(name: str) -> None:
    refs = {r for _, n, r in _pins() if n == name}
    assert len(refs) == 1, f"{name} callers disagree on the pin: {refs}"
    text = _fetch_core_file(refs.pop(), f".github/workflows/{name}.yml")
    assert WRITER_APP in text
    assert "--check-no-companion-required" in text


def test_positive_control_old_pin_lacks_the_exemption() -> None:
    old = "979c290a16f70fcd7e0b62600ac511140baab4a6"  # pragma: allowlist secret
    text = _fetch_core_file(old, ".github/workflows/occ-preflight.yml")
    assert WRITER_APP not in text
