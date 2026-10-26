# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ticket-contract criterion-label check is wired into one test split (OMN-21279).

omnibase_infra#5370 repeated criterion label AC7 that omnibase_infra#5365 had
already merged; Git merged the two contract files with no text conflict, the PR
runs stayed green, and the merge-queue full suite removed the PR three times.
The change-aware selector does not pick ``tests/scripts/`` for a
``contracts/`` change, so the check only ran in the full suite. The wiring
asserted below names the check in one split of the ``test-parallel`` job of
``.github/workflows/ci.yml``, so a PR whose contract repeats a criterion
label goes red on its own run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = ".github/workflows/ci.yml"
_JOB_NAME = "test-parallel"
_CHECK_FILE = "tests/scripts/ci/test_contract_criterion_labels_distinct_omn17427.py"


def _job_steps() -> list[dict[str, Any]]:
    text = (_ROOT / _CI_WORKFLOW).read_text(encoding="utf-8")
    data = cast("dict[str, Any]", yaml.safe_load(text))
    job = cast("dict[str, Any]", data["jobs"][_JOB_NAME])
    return cast("list[dict[str, Any]]", job["steps"])


def _steps_naming_the_check() -> list[dict[str, Any]]:
    return [step for step in _job_steps() if _CHECK_FILE in str(step.get("run", ""))]


def test_exactly_one_step_names_the_criterion_label_check() -> None:
    named = _steps_naming_the_check()
    assert len(named) == 1, (
        f"{_CI_WORKFLOW} job {_JOB_NAME!r} names {_CHECK_FILE!r} in "
        f"{len(named)} steps; expected exactly one"
    )


def test_the_step_is_split_scoped_and_unconditional() -> None:
    (step,) = _steps_naming_the_check()
    assert step.get("if") == "matrix.split == 1"
    assert "continue-on-error" not in step


def test_the_named_check_file_exists() -> None:
    assert (_ROOT / _CHECK_FILE).is_file()
