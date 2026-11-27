# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the failed-head gate against job records on disk (OMN-26493).

The recorded jobs of the 7b25f0ff incident are written into a real state directory, one
JSON file per job, and the gate reads them back as the deploy agent does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_failed_head_gate import (
    refusal,
)

pytestmark = pytest.mark.integration

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "scripts/deploy-agent/tests/unit/fixtures/recorded_jobs_7b25f0ff_omn26493.json"
)


def _state_dir(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for row in recorded["jobs"]:
        job = row["job"]
        (tmp_path / f"{job['correlation_id']}.json").write_text(
            json.dumps(job), encoding="utf-8"
        )
    return tmp_path, recorded


def test_failed_head_is_refused_from_recorded_job_files(tmp_path: Path) -> None:
    state_dir, recorded = _state_dir(tmp_path)
    found = refusal(state_dir, recorded["failed_head"])
    assert found is not None
    assert found.head == recorded["failed_head"]
    assert found.verdict_class.value


def test_newer_head_is_not_refused(tmp_path: Path) -> None:
    state_dir, recorded = _state_dir(tmp_path)
    assert refusal(state_dir, recorded["newer_head"]) is None


def test_empty_state_dir_refuses_nothing(tmp_path: Path) -> None:
    assert refusal(tmp_path, "7b25f0ffe92dbd8bf1284348a37f37351cc30b61") is None
