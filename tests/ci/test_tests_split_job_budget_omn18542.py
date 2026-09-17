# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18542: the CI-contract half of the Tests-split timeout fix.

The sizing fix lives in `scripts/ci/detect_test_paths.py` and is pinned by
`tests/unit/scripts/ci/test_split_count_sizing_omn18542.py`. Two facts about the
job that runs the result are workflow facts, so they are pinned here, in the
CI-contract class that reads `.github/workflows/**` off disk.

Both were silently wrong before:

* The shard budget was 15 minutes while completed shards measured p95 12.3 and
  max 15.3 over the 134 Tests-split jobs of the 40 ci.yml runs to
  2026-09-16T22:51Z. A budget inside its own observed distribution cancels real
  work, which is what happened twice on `omnibase_infra#3652`.
* The durations upload had never produced an artifact. pytest-split writes the
  file and the step then reported "No files were found with the provided path:
  .test_durations" on every shard, because actions/upload-artifact has excluded
  dotfiles by default since v4. Run 35129170714 produced 15 `test-results-*`
  artifacts and zero `test-durations-*` ones, which is why every shard still
  logs "[pytest-split] No test durations found" and splits by case count rather
  than by time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"

# The maximum completed Tests-split duration observed over the 40 ci.yml runs to
# 2026-09-16T22:51Z, in minutes. The budget must exceed it: a budget at or below
# the observed maximum cancels work that was going to pass.
MEASURED_MAX_SHARD_MINUTES = 15.3
TESTS_SPLIT_JOB = "test-parallel"
DURATIONS_STEP = "Upload test durations (for split rebalancing)"


def _tests_split_job() -> dict[str, Any]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    jobs = workflow["jobs"]
    assert TESTS_SPLIT_JOB in jobs, (
        f"{TESTS_SPLIT_JOB} is gone from ci.yml; the shard budget and the "
        "durations upload this module pins have moved or been deleted"
    )
    job = jobs[TESTS_SPLIT_JOB]
    assert isinstance(job, dict)
    return job


def test_shard_budget_exceeds_the_measured_maximum() -> None:
    budget = _tests_split_job()["timeout-minutes"]
    assert budget > MEASURED_MAX_SHARD_MINUTES, (
        f"{TESTS_SPLIT_JOB} budget is {budget} min against a measured maximum "
        f"of {MEASURED_MAX_SHARD_MINUTES} min; a budget inside its own observed "
        "distribution cancels shards that were going to pass -- which is the "
        "defect OMN-18542 opened on"
    )


def test_the_job_still_declares_a_budget_at_all() -> None:
    """The positive control on the assertion above.

    `timeout-minutes` absent would make the comparison a KeyError rather than a
    pass, but an unbounded matrix job is its own incident class (OMN-18405), so
    it is named here rather than left to a traceback.
    """
    assert "timeout-minutes" in _tests_split_job(), (
        f"{TESTS_SPLIT_JOB} declares no timeout-minutes; an unbounded shard "
        "hangs and starves successor runs"
    )


def test_durations_upload_includes_the_dotfile_it_uploads() -> None:
    steps = _tests_split_job()["steps"]
    step = next((s for s in steps if s.get("name") == DURATIONS_STEP), None)
    assert step is not None, f"the '{DURATIONS_STEP}' step is gone from ci.yml"

    with_block = step["with"]
    assert with_block["path"] == ".test_durations"
    assert with_block.get("include-hidden-files") is True, (
        "the durations upload names a dotfile path; without "
        "include-hidden-files, actions/upload-artifact v4+ silently uploads "
        "nothing and the step passes -- measured on run 35129170714, which "
        "produced zero test-durations-* artifacts across all 15 shards"
    )


def test_the_durations_path_is_still_a_dotfile() -> None:
    """The control on WHY the flag is required.

    If the path ever stops being hidden the flag is redundant rather than
    load-bearing, and the assertion above would keep passing while pinning
    nothing. This makes that change fail loudly instead.
    """
    steps = _tests_split_job()["steps"]
    step = next(s for s in steps if s.get("name") == DURATIONS_STEP)
    assert Path(step["with"]["path"]).name.startswith(".")
