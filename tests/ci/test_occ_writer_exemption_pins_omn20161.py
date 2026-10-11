# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20161: the remaining core receipt-gate pin carries the writer exemption.

OMN-20074 retired both OCC preflight callers. The caller-mode receipt gate
still delegates to core, so keep its pin coverage and assert that no preflight
caller remains in the live workflow inventory.
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
USES = re.compile(
    rf"^\s*uses:\s*{re.escape(CORE)}/\.github/workflows/"
    r"(?P<gate>occ-preflight|receipt-gate)\.yml@(?P<ref>\S+)",
    re.MULTILINE,
)
WRITER_APP = "onexbot-occ-writer"
PIN_PROBE_FLAG = "--check-no-companion-required"


def _callers() -> list[tuple[str, str, str]]:
    """Return (workflow file name, gate, ref) for every gate caller in the repo."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for match in USES.finditer(path.read_text()):
            found.append((path.name, match["gate"], match["ref"]))
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


def test_only_receipt_gate_has_a_caller() -> None:
    """Positive control: the scan finds the callers it is meant to police."""
    assert {gate for _, gate, _ in _callers()} == {"receipt-gate"}


def test_occ_preflight_callers_are_absent() -> None:
    files = {name for name, gate, _ in _callers() if gate == "occ-preflight"}
    assert files == set()


@pytest.mark.parametrize(
    ("name", "ref"),
    [(n, r) for n, g, r in _callers() if g == "receipt-gate"],
)
def test_receipt_gate_pin_carries_writer_app_exemption(name: str, ref: str) -> None:
    """The receipt-gate caller advances past #1820 (OMN-20375), so read its own pin."""
    assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{name} pins receipt-gate.yml at {ref}"
    text = _fetch_core_file(ref, ".github/workflows/receipt-gate.yml")
    assert WRITER_APP in text
    assert PIN_PROBE_FLAG in text
