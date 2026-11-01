# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise timeout configuration and its consumers from the built wheel."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_WHEEL_READBACK = """
import asyncio
from pathlib import Path
import yaml
from omnibase_infra.models.operational_timeouts import TIMEOUTS, ModelOperationalTimeouts
from omnibase_infra.models.operational_timeouts import model_operational_timeouts
from omnibase_infra.adapters.adapter_onex_tool_execution import AdapterONEXToolExecution
from omnibase_infra.cli import service_demo_reset
from omnibase_infra.handlers import done_write_receipt_guard, handler_dev_lane_pin, handler_dev_lane_branch_install
from omnibase_infra.runtime import freshness_monitor, runtime_scheduler, projector_shell
from omnibase_infra.runtime.transition_notification_outbox import TransitionNotificationOutbox
from omnibase_infra.runtime.transition_notification_publisher import TransitionNotificationPublisher
from omnibase_infra.utils.util_retry_optimistic import retry_on_optimistic_conflict
from omnibase_infra.utils.util_consumer_restart import run_with_restart

package = Path(model_operational_timeouts.__file__).resolve().parent
assert package.is_relative_to(Path.cwd())
declared = yaml.safe_load((package / 'operational_timeouts.yaml').read_text())['timeouts']
assert TIMEOUTS == ModelOperationalTimeouts.model_validate(declared)
assert TransitionNotificationOutbox.DEFAULT_QUERY_TIMEOUT_SECONDS == declared['database_query_seconds']
assert projector_shell.ProjectorShell.DEFAULT_QUERY_TIMEOUT_SECONDS == declared['database_query_seconds']
assert handler_dev_lane_pin._DOCKER_TIMEOUT_SECONDS == declared['dev_lane_docker_seconds']
assert handler_dev_lane_branch_install.RESTART_TIMEOUT_SECONDS == declared['dev_lane_restart_seconds']
assert done_write_receipt_guard.DEFAULT_DOD_VERIFY_TIMEOUT_SECONDS == declared['done_write_verify_seconds']
assert freshness_monitor._DEFAULT_INTERVAL == declared['freshness_interval_seconds']

async def exercise_consumers():
    attempts = 0
    async def conflict_then_success():
        nonlocal attempts
        attempts += 1
        return attempts >= 3
    assert await retry_on_optimistic_conflict(
        conflict_then_success, check_conflict=lambda result: not result, jitter=False
    ) is True
    assert attempts == 3

    shutdown = asyncio.Event()
    starts = 0
    async def transient_then_shutdown():
        nonlocal starts
        starts += 1
        if starts == 1:
            raise ConnectionError('transient integration-test failure')
        shutdown.set()
    await run_with_restart(transient_then_shutdown, name='timeout-wheel-readback', shutdown_event=shutdown)
    assert starts == 2

asyncio.run(exercise_consumers())
print('wheel timeout readback: declared values loaded; retry attempts=3; consumer starts=2')
"""


def test_built_wheel_loads_defaults_and_runs_consumers(tmp_path: Path) -> None:
    """Packaging must retain the required YAML for fresh-process consumers."""
    root = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--out-dir",
            str(wheels),
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = wheels.glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    result = subprocess.run(
        [sys.executable, "-c", _WHEEL_READBACK],
        cwd=installed,
        check=False,
        env={**os.environ, "PYTHONPATH": str(installed)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "retry attempts=3; consumer starts=2" in result.stdout


@pytest.mark.parametrize("failure", ["missing", "malformed", "invalid"])
def test_unusable_timeout_configuration_has_a_typed_error(
    tmp_path: Path, failure: str
) -> None:
    from omnibase_infra.errors import ProtocolConfigurationError
    from omnibase_infra.models.operational_timeouts import ModelOperationalTimeouts

    path = tmp_path / "operational_timeouts.yaml"
    if failure == "malformed":
        path.write_text("timeouts: [\n", encoding="utf-8")
    elif failure == "invalid":
        path.write_text("timeouts: {database_query_seconds: -1}\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError, match=r"operational_timeouts\.yaml"):
        ModelOperationalTimeouts.from_file(path)


def test_timeout_slice_has_no_literal_timeout_findings() -> None:
    from omnibase_core.validation.validator_contracts_and_di import (
        ValidatorContractsAndDi,
    )

    root = Path(__file__).resolve().parents[2] / "src/omnibase_infra"
    files = [
        "adapters/adapter_onex_tool_execution.py",
        "cli/service_demo_reset.py",
        "handlers/done_write_receipt_guard.py",
        "handlers/handler_dev_lane_branch_install.py",
        "handlers/handler_dev_lane_pin.py",
        "runtime/freshness_monitor.py",
        "runtime/projector_shell.py",
        "runtime/runtime_scheduler.py",
        "runtime/transition_notification_outbox.py",
        "runtime/transition_notification_publisher.py",
        "utils/util_consumer_restart.py",
        "utils/util_retry_optimistic.py",
    ]
    targets = [root / file for file in files]
    assert all(path.is_file() for path in targets)
    result = ValidatorContractsAndDi().validate(targets)
    assert result.metadata is not None
    assert result.metadata.files_processed == len(targets)
    findings = [
        issue for issue in result.issues if issue.rule_name == "literal_timeout"
    ]
    assert findings == [], findings
