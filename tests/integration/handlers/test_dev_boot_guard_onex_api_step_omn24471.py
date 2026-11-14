# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed onex-api roll step crosses the boot guard and keeps the runtime (OMN-24471).

The guard and ``run_runtime_roll_step`` are real. Only the compose command and the
host container list are substituted; no Docker daemon or declared lane is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from tests.unit.handlers.test_handler_dev_boot_guard_companion_lane import (
    NEW_REVISION,
    _family,
    _guard,
    _Host,
    _incident,
)
from tests.unit.handlers.test_handler_dev_boot_guard_onex_api_step import (
    _roll_execution,
)

pytestmark = pytest.mark.integration


def test_failed_onex_api_roll_step_keeps_the_healthy_runtime(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError, match="onex-api is unhealthy"):
        with guard.protect("job-integration", "c" * 40, lambda error: None):
            host.containers = _family(NEW_REVISION)
            guard_mod.run_runtime_roll_step(
                ["docker", "compose", "up", "-d", "--no-deps", "onex-api"],
                30,
                execution=_roll_execution(),
            )

    assert not host.restored
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["runtime_kept_revision"] == NEW_REVISION
