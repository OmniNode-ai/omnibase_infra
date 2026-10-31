# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Operational timeouts are read from the declared file, not from call-site literals (OMN-21802)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.adapters.adapter_onex_tool_execution import (
    AdapterONEXToolExecution,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers import (
    done_write_receipt_guard,
    handler_dev_lane_branch_install,
    handler_dev_lane_pin,
)
from omnibase_infra.models.operational_timeouts import (
    TIMEOUTS,
    ModelOperationalTimeouts,
    load_operational_timeouts,
    model_operational_timeouts,
)
from omnibase_infra.runtime import freshness_monitor, projector_shell
from omnibase_infra.runtime.models.model_runtime_scheduler_config import (
    ModelRuntimeSchedulerConfig,
)
from omnibase_infra.runtime.transition_notification_outbox import (
    TransitionNotificationOutbox,
)
from omnibase_infra.runtime.transition_notification_publisher import (
    TransitionNotificationPublisher,
)
from omnibase_infra.utils.util_consumer_restart import run_with_restart
from omnibase_infra.utils.util_retry_optimistic import retry_on_optimistic_conflict

pytestmark = pytest.mark.unit

_DECLARED = (
    Path(model_operational_timeouts.__file__).parent / "operational_timeouts.yaml"
)


def _block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))
    return dict(document["timeouts"])


def _default(function: Callable[..., object], name: str) -> object:
    return inspect.signature(function).parameters[name].default


def test_loaded_timeouts_equal_the_declared_file() -> None:
    assert load_operational_timeouts() is TIMEOUTS
    assert ModelOperationalTimeouts.model_validate(_block()) == TIMEOUTS


@pytest.mark.parametrize("key", sorted(_block()))
def test_a_block_missing_a_value_fails_to_load(key: str) -> None:
    block = _block()
    del block[key]
    with pytest.raises(ValidationError):
        ModelOperationalTimeouts.model_validate(block)


def test_a_file_without_a_timeouts_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "operational_timeouts.yaml"
    path.write_text("other: 1\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelOperationalTimeouts.from_file(path)


@pytest.mark.parametrize(
    ("initial_key", "max_key"),
    [
        (
            "optimistic_retry_initial_backoff_seconds",
            "optimistic_retry_max_backoff_seconds",
        ),
        (
            "consumer_restart_initial_backoff_seconds",
            "consumer_restart_max_backoff_seconds",
        ),
    ],
)
def test_an_initial_backoff_above_its_ceiling_is_refused(
    initial_key: str, max_key: str
) -> None:
    block = _block()
    block[initial_key] = getattr(TIMEOUTS, max_key) + 1.0
    with pytest.raises(ValidationError):
        ModelOperationalTimeouts.model_validate(block)


def test_outbox_and_projector_defaults_are_the_declared_values() -> None:
    outbox = TransitionNotificationOutbox
    assert TIMEOUTS.outbox_poll_interval_seconds == outbox.DEFAULT_POLL_INTERVAL_SECONDS
    assert TIMEOUTS.database_query_seconds == outbox.DEFAULT_QUERY_TIMEOUT_SECONDS
    assert TIMEOUTS.outbox_shutdown_seconds == outbox.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    assert (
        TIMEOUTS.database_query_seconds
        == projector_shell.ProjectorShell.DEFAULT_QUERY_TIMEOUT_SECONDS
    )


def test_freshness_and_publisher_defaults_are_the_declared_values() -> None:
    assert TIMEOUTS.freshness_interval_seconds == freshness_monitor._DEFAULT_INTERVAL
    assert (
        _default(
            TransitionNotificationPublisher.__init__, "circuit_breaker_reset_timeout"
        )
        == TIMEOUTS.circuit_breaker_reset_seconds
    )
    assert (
        _default(AdapterONEXToolExecution.__init__, "circuit_breaker_reset_timeout")
        == TIMEOUTS.circuit_breaker_reset_seconds
    )
    assert (
        _default(AdapterONEXToolExecution.__init__, "default_timeout")
        == TIMEOUTS.tool_execution_seconds
    )


def test_scheduler_default_config_carries_the_declared_values() -> None:
    config = ModelRuntimeSchedulerConfig.default()
    assert config.tick_interval_ms == TIMEOUTS.scheduler_tick_interval_ms
    assert (
        config.circuit_breaker_reset_timeout_seconds
        == TIMEOUTS.circuit_breaker_reset_seconds
    )
    assert config.valkey_timeout_seconds == TIMEOUTS.scheduler_valkey_seconds


def test_backoff_defaults_are_the_declared_values() -> None:
    retry = retry_on_optimistic_conflict
    assert (
        _default(retry, "initial_backoff")
        == TIMEOUTS.optimistic_retry_initial_backoff_seconds
    )
    assert (
        _default(retry, "max_backoff") == TIMEOUTS.optimistic_retry_max_backoff_seconds
    )
    assert (
        _default(retry, "backoff_multiplier")
        == TIMEOUTS.optimistic_retry_backoff_multiplier
    )
    assert (
        _default(run_with_restart, "initial_backoff_s")
        == TIMEOUTS.consumer_restart_initial_backoff_seconds
    )
    assert (
        _default(run_with_restart, "max_backoff_s")
        == TIMEOUTS.consumer_restart_max_backoff_seconds
    )
    assert (
        _default(run_with_restart, "backoff_factor")
        == TIMEOUTS.consumer_restart_backoff_factor
    )


def test_dev_lane_and_done_write_waits_are_the_declared_values() -> None:
    assert (
        TIMEOUTS.dev_lane_ledger_seconds == handler_dev_lane_pin._LEDGER_TIMEOUT_SECONDS
    )
    assert (
        TIMEOUTS.dev_lane_docker_seconds == handler_dev_lane_pin._DOCKER_TIMEOUT_SECONDS
    )
    assert (
        TIMEOUTS.dev_lane_docker_seconds
        == handler_dev_lane_branch_install.DOCKER_TIMEOUT_SECONDS
    )
    assert (
        TIMEOUTS.dev_lane_restart_seconds
        == handler_dev_lane_branch_install.RESTART_TIMEOUT_SECONDS
    )
    assert (
        TIMEOUTS.done_write_verify_seconds
        == done_write_receipt_guard.DEFAULT_DOD_VERIFY_TIMEOUT_SECONDS
    )
