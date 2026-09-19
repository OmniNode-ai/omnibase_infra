# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18776 — incident replay for the skip-count ratchet (OMN-15547 R1-R5).

The incident is OMN-18781, found by the OMN-18775 inventory: two integration
suites in this repo are gated on environment variables that no workflow here
sets, so they are collected on every pull request, skip in full, and contribute
to a green Tests job. The event-bus one is the load-bearing case — the bus is
the transport the whole platform rests on and its integration proof has never
executed in CI.

Nothing refused them when they were added, and nothing has refused them since,
because no surface on the fleet reads a skip count at all.

The artifact is the real JUnit report from split 2 of run 35402899669, the bytes
GitHub Actions served, unedited. It carries all 27 of those never-executable
cases: 19 from ``test_kafka_event_bus_integration`` and 8 from
``test_handler_qdrant_integration``.

The replay drives the REAL ratchet over those exact bytes against the baseline
as it stood before those suites existed, and requires the refusal that never
happened. The discriminator drives the same function over the same bytes against
the shipped baseline and requires acceptance, because a guard that refused every
report would replay this incident perfectly and block every pull request forever.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "skip_count_ratchet.py"
BASELINE = REPO_ROOT / "config" / "skip_count_baseline.yaml"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18776"
    / "run-35402899669-split-2-junit.xml.captured"
)
SUITE = "omnibase_infra/test-parallel"

# The two modules OMN-18781 measured. Named here rather than derived, because
# deriving them from the fixture would make the test agree with whatever the
# fixture happens to contain.
NEVER_EXECUTABLE_MODULES = (
    "tests.integration.event_bus.test_kafka_event_bus_integration",
    "tests.integration.handlers.test_handler_qdrant_integration",
)

pytestmark = pytest.mark.unit


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def _shipped() -> dict[str, object]:
    loaded = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _ids_from_never_executable_suites() -> list[str]:
    entry = _shipped()["suites"][SUITE]  # type: ignore[index]
    return [
        i
        for i in entry["node_ids"]
        if any(i.startswith(m) for m in NEVER_EXECUTABLE_MODULES)
    ]


def _baseline_without_those_suites(tmp_path: Path) -> Path:
    """The baseline as it stood before OMN-18781's two suites existed."""
    loaded = _shipped()
    entry = loaded["suites"][SUITE]  # type: ignore[index]
    dropped = set(_ids_from_never_executable_suites())
    kept = [i for i in entry["node_ids"] if i not in dropped]
    entry["node_ids"] = kept
    entry["max_skips"] = len(kept)
    path = tmp_path / "pre-incident-baseline.yaml"
    path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    return path


def test_the_captured_report_really_carries_the_never_executable_cases() -> None:
    """Neither assertion below can be vacuous if this one holds."""
    body = FIXTURE.read_text(encoding="utf-8")
    assert FIXTURE.stat().st_size > 100_000, "the capture is the whole split report"
    ids = _ids_from_never_executable_suites()
    assert len(ids) == 27, f"expected the 27 OMN-18781 cases, found {len(ids)}"
    for module in NEVER_EXECUTABLE_MODULES:
        assert module in body, f"{module} is absent from the captured bytes"


def test_the_real_ratchet_refuses_the_report_that_grew_the_never_run_set(
    tmp_path: Path,
) -> None:
    """R5, false_green: the refusal nobody got when those suites were added."""
    baseline = _baseline_without_those_suites(tmp_path)
    result = _run(
        "--baseline", str(baseline), "--suite", SUITE, "--junit", str(FIXTURE)
    )
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "omnibase_infra" in out
    assert "+27" in out, "the refusal must name the delta"
    for module in NEVER_EXECUTABLE_MODULES:
        assert module in out, f"the refusal must name {module}"


def test_the_same_ratchet_accepts_the_same_bytes_against_the_shipped_baseline() -> None:
    """Discriminator: a guard that refused every report would replay this perfectly."""
    result = _run(
        "--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(FIXTURE)
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "narrowed selection" in out.lower(), (
        "one split of a 15-split run is a narrowed view of the baseline, and the "
        "gate must say so rather than reading it as progress"
    )
