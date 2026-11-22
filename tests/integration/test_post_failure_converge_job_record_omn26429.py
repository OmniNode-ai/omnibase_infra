# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The post-failure converge rule reads a recorded agent job record (OMN-26429).

The fixture is a real failed dev job record. The rule must parse it, see a failed
dev job, and recover the infra commit the boot guard wrote into its errors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_post_failure_converge import (
    EnumJobStatus,
    failed_dev_job,
    latest_terminal_dev_job,
    read_job_record,
)

pytestmark = pytest.mark.integration

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dev_job_a7474c42"
    / "job_record.json"
)
_BUILT_COMMIT = "8b76765e2fbb1a3f761bbf1788d972ab71e24bd4"


def test_recorded_failed_dev_job_is_read_with_its_source() -> None:
    job = read_job_record(_FIXTURE)
    assert job is not None
    assert job.status is EnumJobStatus.FAILED
    failed = failed_dev_job(job)
    assert failed is not None
    assert failed.sources == {"omnibase_infra": _BUILT_COMMIT}


def test_latest_terminal_dev_job_finds_the_recorded_failure(tmp_path: Path) -> None:
    shutil.copy(_FIXTURE, tmp_path / "job.json")
    job = latest_terminal_dev_job(tmp_path)
    assert job is not None
    assert job.status is EnumJobStatus.FAILED
