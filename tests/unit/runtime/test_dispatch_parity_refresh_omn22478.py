# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22580: one command refreshes every dispatch-parity fixture, and the gate still bites.

A pull request that adds a node used to edit about ten corpus counts by hand in six
files. ``--refresh-fixtures`` rewrites the oracle, the OCC-free header and the recorded
readback from the live corpus, and ``--check`` is the gate's staleness step. These tests
plant a stale count and a stale probe in a scratch copy of the fixtures and show that the
check refuses them, that the refresh repairs them without a hand edit, and that a refresh
on matching fixtures writes nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _ROOT / ".github/workflows/dispatch-parity-gate.yml"


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    return harness.build_snapshot()


@pytest.fixture
def scratch(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "oracle_path": tmp_path / "selection-oracle-v2.json",
        "no_occ_header_path": tmp_path / "selection-oracle-no-occ-v2.header.json",
        "readback_path": tmp_path / "readback.json",
    }
    shutil.copy(harness.ORACLE_PATH, paths["oracle_path"])
    shutil.copy(harness.NO_OCC_HEADER_PATH, paths["no_occ_header_path"])
    shutil.copy(harness.READBACK_PATH, paths["readback_path"])
    return paths


def test_committed_fixtures_match_the_live_engine(live: dict[str, Any]) -> None:
    assert harness.fixture_problems(live) == {}


def test_refresh_on_matching_fixtures_writes_nothing(
    live: dict[str, Any], scratch: dict[str, Path]
) -> None:
    before = {name: path.read_bytes() for name, path in scratch.items()}
    assert harness.refresh_fixtures(live, **scratch) == []
    assert {name: path.read_bytes() for name, path in scratch.items()} == before


def test_planted_stale_count_is_refused_then_refreshed(
    live: dict[str, Any], scratch: dict[str, Path]
) -> None:
    header_path = scratch["no_occ_header_path"]
    header = json.loads(header_path.read_text())
    header["corpus"]["contracts_discovered"] += 1
    header["corpus"]["contracts_by_package"]["omnibase_infra"] += 1
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")

    problems = harness.fixture_problems(live, **scratch)
    assert "oracle" in problems
    assert "corpus" in problems["oracle"][0]

    written = harness.refresh_fixtures(live, **scratch)
    assert header_path in written
    assert harness.fixture_problems(live, **scratch) == {}


def test_planted_stale_probe_is_refused_then_refreshed(
    live: dict[str, Any], scratch: dict[str, Path]
) -> None:
    oracle_path = scratch["oracle_path"]
    oracle = json.loads(oracle_path.read_text())
    probe = next(iter(oracle["probes"].values()))
    probe["selection"]["status"] = "planted-regression"
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")

    problems = harness.fixture_problems(live, **scratch)
    assert "probes" in problems["oracle"][0]

    assert oracle_path in harness.refresh_fixtures(live, **scratch)
    assert harness.fixture_problems(live, **scratch) == {}


def test_planted_stale_readback_digest_is_refused_then_refreshed(
    live: dict[str, Any], scratch: dict[str, Path]
) -> None:
    readback_path = scratch["readback_path"]
    readback = json.loads(readback_path.read_text())
    readback["retired"]["snapshot_sha256"] = "0" * 64
    readback["retired"]["corpus"]["probe_count"] -= 1
    readback_path.write_text(json.dumps(readback, indent=2) + "\n")

    problems = harness.fixture_problems(live, **scratch)
    assert set(problems) == {"readback"}

    assert harness.refresh_fixtures(live, **scratch) == [readback_path]
    refreshed = json.loads(readback_path.read_text())
    assert refreshed["retired"]["snapshot_sha256"] == harness.snapshot_digest(live)
    assert refreshed["legacy"] == readback["legacy"]
    assert harness.fixture_problems(live, **scratch) == {}


def test_refresh_refuses_while_occ_is_a_declared_dependency(
    live: dict[str, Any], scratch: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="OCC-free closure only"):
        harness.refresh_fixtures(
            live,
            required_packages=(
                "omnibase-core",
                "omnibase-infra",
                "onex-change-control",
            ),
            **scratch,
        )


def test_stale_report_message_names_the_refresh_command() -> None:
    report = harness.format_stale_report({"oracle": ["differs in: probes"]})
    assert harness.REFRESH_COMMAND in report
    assert "--refresh-fixtures" in report
    assert "[oracle] differs in: probes" in report


def test_gate_step_runs_the_check_and_never_tells_authors_to_regenerate_with_out() -> (
    None
):
    text = _WORKFLOW.read_text()
    assert "harness --check" in text
    assert "harness --out" not in text
    assert "dispatch_parity_readback_omn20074.json" in text
