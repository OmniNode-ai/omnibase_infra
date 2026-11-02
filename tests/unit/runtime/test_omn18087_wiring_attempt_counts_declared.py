# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The auto-wiring attempt counts come from the declared operational file (OMN-18087).

The consume-boundary DLQ retry count and the NOT_READY reattach count are read
from ``operational_timeouts.yaml``; ``handler_wiring`` holds no literal for them.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.operational_timeouts import (
    TIMEOUTS,
    model_operational_timeouts,
)
from omnibase_infra.runtime.auto_wiring import handler_wiring

pytestmark = pytest.mark.unit

_DECLARED = Path(model_operational_timeouts.__file__).with_name(
    "operational_timeouts.yaml"
)


def test_the_declared_file_carries_both_attempt_counts() -> None:
    block = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))["timeouts"]

    assert TIMEOUTS.boundary_dlq_max_attempts == block["boundary_dlq_max_attempts"]
    assert (
        TIMEOUTS.not_ready_retry_max_attempts == block["not_ready_retry_max_attempts"]
    )


def test_reconciliation_loop_defaults_to_the_declared_count() -> None:
    parameter = inspect.signature(handler_wiring.run_not_ready_reconciliation_loop)
    default = parameter.parameters["max_attempts"].default

    assert default == TIMEOUTS.not_ready_retry_max_attempts


def test_handler_wiring_binds_the_declared_counts() -> None:
    assert (
        handler_wiring._BOUNDARY_DLQ_MAX_ATTEMPTS == TIMEOUTS.boundary_dlq_max_attempts
    )
    assert not hasattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_MAX_ATTEMPTS")
