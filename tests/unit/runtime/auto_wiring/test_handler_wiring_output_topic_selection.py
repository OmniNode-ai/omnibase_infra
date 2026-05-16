# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for generic dispatch-result output topic selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _select_dispatch_result_output_topic,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)

pytestmark = pytest.mark.unit


def _contract(
    *,
    publish_topics: tuple[str, ...],
    terminal_event: str | None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="contract_sweep",
        node_type="compute",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract_sweep/contract.yaml"),
        entry_point_name="node_contract_sweep",
        package_name="omnimarket",
        package_version="0.2.0",
        terminal_event=terminal_event,
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.contract-sweep-start.v1",),
            publish_topics=publish_topics,
        ),
    )


def test_prefers_terminal_event_when_published() -> None:
    contract = _contract(
        publish_topics=(
            "onex.evt.omnimarket.contract-sweep-violation.v1",
            "onex.evt.omnimarket.contract-sweep-completed.v1",
        ),
        terminal_event="onex.evt.omnimarket.contract-sweep-completed.v1",
    )

    assert (
        _select_dispatch_result_output_topic(contract)
        == "onex.evt.omnimarket.contract-sweep-completed.v1"
    )


def test_falls_back_to_first_publish_topic_without_terminal_event() -> None:
    contract = _contract(
        publish_topics=("onex.evt.omnimarket.contract-sweep-violation.v1",),
        terminal_event=None,
    )

    assert (
        _select_dispatch_result_output_topic(contract)
        == "onex.evt.omnimarket.contract-sweep-violation.v1"
    )
