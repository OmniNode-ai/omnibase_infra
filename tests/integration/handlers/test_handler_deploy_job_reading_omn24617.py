# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed deploy job's reading row reaches a real append command (OMN-24617).

The handler is built from the node's own contract settings and run against a real
stand-in append executable, so the argv, the stdin row and the request id cross a real
process boundary.
"""

from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.handlers.handler_deploy_job_reading import HandlerDeployJobReading
from omnibase_infra.models.model_deploy_job_reading_request import (
    ModelDeployJobReadingRequest,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_NOW = datetime(2026, 11, 15, 14, 40, 0, tzinfo=UTC)


def _handler(command: str) -> HandlerDeployJobReading:
    settings = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]
    return HandlerDeployJobReading(
        settings=settings,
        environ={
            settings["append_command_env"]: command,
            settings["host_name_env"]: "h201",
        },
    )


def _request(job_outcome: str, lane: str = "dev") -> ModelDeployJobReadingRequest:
    return ModelDeployJobReadingRequest(
        correlation_id=uuid4(),
        runtime_lane=lane,
        scope="full",
        job_outcome=job_outcome,
        phase_results={"preflight": "success", "git": "failed"},
        published_at=_NOW,
    )


def _append_script(tmp_path: Path) -> tuple[Path, Path]:
    sink = tmp_path / "sink.txt"
    script = tmp_path / "append.sh"
    script.write_text(f'#!/bin/sh\necho "$@" > {sink}\ncat >> {sink}\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, sink


def test_failed_job_row_is_appended_through_a_real_command(tmp_path: Path) -> None:
    script, sink = _append_script(tmp_path)
    request = _request("failed")
    result = _handler(str(script)).handle(request)
    assert result.outcome == "appended"
    written = sink.read_text(encoding="utf-8")
    assert "verdict=NOT_WORKING" in written
    assert "reason=job-failed:git" in written
    assert f"--request-id {result.append_key}" in written
    assert f"job={request.correlation_id}" in written


def test_prod_job_writes_nothing(tmp_path: Path) -> None:
    script, sink = _append_script(tmp_path)
    result = _handler(str(script)).handle(_request("failed", lane="prod"))
    assert result.outcome == "skipped"
    assert not sink.exists()
