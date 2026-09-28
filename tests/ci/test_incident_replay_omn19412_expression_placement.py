# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: the probe-window check refused a host-agnostic C13 (OMN-19412).

THE INCIDENT. omninode_infra#1725 (OMN-17427) merged to ``dev`` at
2026-09-28T05:28:12Z and placed C13, C14 and C29 with
``runs-on: ${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON) }}``, as operator
rulings 2026-09-28T01:58:06Z and 01:58:17Z required. The check read literal
runner labels only, so from that minute the required "Lab Probe Windows
(OMN-19412)" job failed on every omnibase_infra pull request with "a job is
placed by an expression, so its runner label cannot be read" (for example
omnibase_infra run 36382911915, job 108802601005), and CI Summary with it.
The input was good; the verdict was wrong (false_red).

THE ARTIFACT is the C13 workflow exactly as it stood at the #1725 merge commit,
read with ``git show`` and committed unmodified. The guard is driven over those
bytes with the committed placement value (the value the live repository
variable held at the time) and must accept. THE DISCRIMINATOR drives the same
bytes with that placement declared null and requires a refusal naming the job,
so a guard stuck open cannot pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.ci import check_lab_probe_windows as plw

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests/fixtures/omn19412/c13-customer-local-delegation.yml.captured"
)
FIXTURE_SHA256 = "52d7cf3d3a45812f3e8ac3141a94f424177cb66756e008e634db037653ffc2a7"
C13_PATH = ".github/workflows/c13-customer-local-delegation.yml"
# gh variable list --repo OmniNode-ai/omninode_infra, 2026-09-28T06:27Z.
LIVE_POOL = '["self-hosted","omnibase-verify"]'


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "omninode_infra"
    target = root / C13_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(FIXTURE.read_bytes())
    return root


def _committed_c13() -> list[plw.ProbeWindow]:
    windows = plw.load_windows(REPO_ROOT / "config/lab_probe_windows.yaml")
    return [w for w in windows if w.id == "C13"]


@pytest.mark.unit
def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.unit
def test_the_captured_c13_is_placed_by_the_variable() -> None:
    """Not vacuous: the captured job really is expression-placed."""
    jobs = plw.read_workflow(FIXTURE)["jobs"]
    assert [job["runs-on"] for job in jobs.values()] == [
        "${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON) }}"
    ]


@pytest.mark.unit
def test_the_real_guard_accepts_the_captured_c13_on_the_live_pool(
    tmp_path: Path,
) -> None:
    variables = plw.CommittedPlacements(
        {"omninode_infra": {"CUSTOMER_MACHINE_RUNS_ON_JSON": LIVE_POOL}}
    )
    assert (
        plw.check(_committed_c13(), {"omninode_infra": _root(tmp_path)}, variables)
        == []
    )


@pytest.mark.unit
def test_the_same_guard_refuses_the_captured_c13_with_the_variable_unset(
    tmp_path: Path,
) -> None:
    variables = plw.CommittedPlacements(
        {"omninode_infra": {"CUSTOMER_MACHINE_RUNS_ON_JSON": None}}
    )
    errors = plw.check(_committed_c13(), {"omninode_infra": _root(tmp_path)}, variables)
    assert len(errors) == 1, errors
    assert "c13-customer-local" in errors[0]
    assert "CUSTOMER_MACHINE_RUNS_ON_JSON" in errors[0]
    assert "committed as unset" in errors[0]
