# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The census CI wiring for the desired-state kinds carries recorded evidence (OMN-19414).

The tests that grade the desired-state kinds run in a dedicated CI step and a
pre-commit hook. That wiring is CI tooling, so it is admitted by one recorded
observation: the census driver run read-only on the lab host, once with no
desired state named and once naming a file that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_STEP_TESTS = (
    "tests/unit/scripts/test_lane_census_plan.py",
    "tests/unit/scripts/test_lane_census_dry_run.py",
)


@pytest.mark.live_contact("tests/ci/fixtures/lane_census_desired_state_omn19414.json")
def test_the_lab_host_census_fails_closed_on_a_removed_desired_state(
    recorded_response: dict[str, object],
) -> None:
    runs = recorded_response["runs"]
    assert isinstance(runs, dict)
    assert runs["desired_state_unset"] == {
        "exit_code": 0,
        "lab_sync_evaluated": False,
        "has_drift": False,
        "finding_kinds": [],
    }
    removed = runs["desired_state_removed"]
    assert removed["exit_code"] == 30
    assert removed["lab_sync_evaluated"] is True
    assert removed["has_drift"] is True
    assert removed["finding_kinds"] == ["desired_state_unreadable"]
    assert removed["severity"] == "critical"


@pytest.mark.parametrize(
    "relative", [".github/workflows/ci.yml", ".pre-commit-config.yaml"]
)
def test_the_ci_step_and_the_hook_still_run_the_desired_state_tests(
    relative: str,
) -> None:
    text = (_ROOT / relative).read_text(encoding="utf-8")
    assert 'test_lane_census_plan.py -k "lab_sync or unreadable"' in text, relative
    assert "test_lane_census_dry_run.py" in text, relative
    for test in _STEP_TESTS:
        assert (_ROOT / test).is_file(), test
