# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20161 - every core reusable-gate caller pin must carry the writer-app exemption.

omnibase_core#1820 exempts the OCC writer app from ``occ-preflight`` and the
Receipt Gate, but only when the pin-only probe proves the producer's outcome.
Callers here pin those reusables by sha, so the exemption reaches this repo only
once every pin has moved. The two ``occ-preflight`` callers must move together.
This test follows the chain: caller pin, then the workflow text at that pin.
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
GATES = ("occ-preflight", "receipt-gate")
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


def test_both_gates_have_callers() -> None:
    """Positive control: the scan finds the callers it is meant to police."""
    assert {gate for _, gate, _ in _callers()} == set(GATES)


def test_occ_preflight_callers_are_all_present() -> None:
    files = {name for name, gate, _ in _callers() if gate == "occ-preflight"}
    assert files == {"ci.yml", "hostile-reviewer.yml"}


@pytest.mark.parametrize(
    ("name", "gate", "ref"),
    [c for c in _callers() if c[1] == "occ-preflight"],
)
def test_occ_preflight_pin_is_the_writer_exemption_sha(
    name: str, gate: str, ref: str
) -> None:
    assert ref == EXPECTED_PIN, (
        f"{name} pins {gate}.yml at {ref}; advance it to {EXPECTED_PIN} so the OCC "
        "writer app exemption (omnibase_core#1820) applies. occ-preflight callers "
        "must move together."
    )


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


def test_occ_preflight_pinned_workflow_carries_writer_app_exemption() -> None:
    text = _fetch_core_file(EXPECTED_PIN, ".github/workflows/occ-preflight.yml")
    assert WRITER_APP in text
    assert PIN_PROBE_FLAG in text
