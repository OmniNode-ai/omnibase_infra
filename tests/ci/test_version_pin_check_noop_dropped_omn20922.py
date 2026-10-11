# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20922: the Version Pin Compliance job no longer touches onex_change_control.

The job's onex_change_control step looked for ``scripts/check_version_pins.py``,
a path that does not exist at the pinned onex_change_control sha (the script
lives at ``standards/check_version_pins.py``), printed "not found -- skipping"
and exited 0 on every run. The retirement deletes the app-token mint, the
checkout and that step. The Dockerfile.runtime plugin pin check and the runtime
image identity check stay, and so does the job name (a required context).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_JOB_ID = "version-pin-check"
_JOB_NAME = "Version Pin Compliance"
_FORBIDDEN = ("onex_change_control", "check_version_pins", "app-token")

_KEPT_STEPS = {
    "Check Dockerfile.runtime plugin pins": (
        'echo "================================================================"\n'
        'echo "Dockerfile.runtime Plugin Pin Check (OMN-5108)"\n'
        'echo "================================================================"\n'
        "uv run python scripts/check_dockerfile_pins.py\n"
    ),
    "Check runtime image identity stamping": (
        'echo "================================================================"\n'
        'echo "Runtime Image Identity Check (OMN-12965)"\n'
        'echo "================================================================"\n'
        "uv run python scripts/check_runtime_image_identity.py\n"
    ),
}


def _job_text(text: str) -> str:
    """Return the raw ``version-pin-check`` job block of a ci.yml text."""
    match = re.search(
        rf"^  {re.escape(_JOB_ID)}:\n(?P<body>(?:(?:    .*)?\n)+)", text, re.MULTILINE
    )
    assert match is not None, f"{_JOB_ID} job block not found"
    return match.group("body")


def _job() -> dict[str, Any]:
    job = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))["jobs"][_JOB_ID]
    assert isinstance(job, dict)
    return job


def _forbidden_in(block: str) -> list[str]:
    return [token for token in _FORBIDDEN if token in block]


def test_job_block_names_no_onex_change_control_step() -> None:
    assert _forbidden_in(_job_text(CI_YML.read_text(encoding="utf-8"))) == []


def test_forbidden_token_scan_flags_a_planted_step() -> None:
    """Positive control: the scan above is not vacuous."""
    planted = _job_text(CI_YML.read_text(encoding="utf-8")) + (
        "      - name: Checkout onex_change_control standards\n"
    )
    assert _forbidden_in(planted) == ["onex_change_control"]


def test_no_skip_when_script_missing_guard_remains() -> None:
    """The deleted step was the one that skipped itself when its script was absent."""
    assert "skipping version pin check" not in _job_text(
        CI_YML.read_text(encoding="utf-8")
    )


@pytest.mark.live_contact("tests/ci/fixtures/version_pin_check_noop_run_omn20922.json")
def test_recorded_dev_run_shows_the_deleted_step_was_a_noop(
    recorded_response: dict[str, object],
) -> None:
    """The recorded dev run is the evidence that the deleted step never checked anything."""
    response = cast("dict[str, Any]", recorded_response["response"])
    lines = response["log_lines"]
    pinned = response["onex_change_control_tree_at_pin"]
    # The checkout fetched the pin, and the step looked for a script path that the
    # pin does not contain, so it skipped itself with a warning and exited 0.
    assert pinned["pin"] in lines["checkout_ref"]
    assert "scripts/check_version_pins.py" in lines["step_group"]
    assert lines["skip_warning"].endswith("not found \u2014 skipping version pin check")
    assert (
        "scripts/check_version_pins.py" not in pinned["paths_named_check_version_pins"]
    )
    assert "standards/check_version_pins.py" in pinned["paths_named_check_version_pins"]
    # The two steps this job keeps passed in the same run.
    assert "Dockerfile pin check passed" in lines["dockerfile_pin_check"]
    assert "identity check passed" in lines["image_identity_check"]


def test_job_name_is_unchanged() -> None:
    assert _job()["name"] == _JOB_NAME


def test_dockerfile_pin_and_image_identity_steps_are_kept_verbatim() -> None:
    steps = {step["name"]: step for step in _job()["steps"] if "name" in step}
    for name, run in _KEPT_STEPS.items():
        assert steps[name]["run"] == run, name


def test_job_still_sets_up_uv_before_the_kept_steps() -> None:
    names = [step.get("name") for step in _job()["steps"]]
    assert names == [
        "Checkout code",
        "Setup Python and uv",
        *_KEPT_STEPS,
    ]
