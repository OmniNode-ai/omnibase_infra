# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dependency-pins job reads the core policy that exempts one hook rev (OMN-23246).

RULING 2026-11-09T01:57:48Z keeps the check-public-runner-placement hook pinned at one
omnibase_core rev past the 14-tag bound for seven days. The exemption itself lives in
omnibase_core's dependency_pin_policy.yaml, which the reusable workflow reads from the
checkout it is given, so the caller job names that checkout. These tests hold the caller
to the ruling: one input, an immutable commit, the held-back rev unchanged and still
the only hook off the dev-merge tag.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
HELD_BACK_REV_PREFIX = "1c9566e2c249"
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _job() -> dict[str, object]:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    job = ci["jobs"]["dependency-pins"]
    assert isinstance(job, dict)
    return job


def test_dependency_pins_job_passes_one_immutable_core_commit() -> None:
    inputs = _job()["with"]
    assert isinstance(inputs, dict)
    assert set(inputs) == {"omnibase-core-ref"}
    assert FULL_SHA.fullmatch(str(inputs["omnibase-core-ref"])), inputs


def test_dependency_pins_job_still_calls_the_core_workflow_unconditionally() -> None:
    job = _job()
    assert str(job["uses"]).startswith(
        "OmniNode-ai/omnibase_core/.github/workflows/dependency-pins.yml@"
    )
    assert "if" not in job
    assert "needs" not in job


def test_held_back_hook_rev_is_the_ruled_one_and_alone() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    held = [
        (str(entry["rev"]), [str(hook["id"]) for hook in entry["hooks"]])
        for entry in config["repos"]
        if any(hook["id"] == "check-public-runner-placement" for hook in entry["hooks"])
    ]
    assert len(held) == 1
    rev, hooks = held[0]
    assert rev.startswith(HELD_BACK_REV_PREFIX)
    assert FULL_SHA.fullmatch(rev)
    assert hooks == ["check-public-runner-placement"]
