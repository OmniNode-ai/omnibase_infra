# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: a lazy wake of a contract the boot never subscribes does not fail.

The dev runtime of 2026-11-24T00:37:39Z logged ``did not wake for a message`` for
``node_projection_skill_executions`` (2 messages), ``node_projection_cost_by_repo`` (3) and
``projection_llm_cost`` (1, plus the ``node_claude_hook_event_effect`` closure that held it). Each of the
three wires a standalone runner, so ``subscribe_wired_contract_topics`` excludes it as
``NO_LIVE_DISPATCHER`` (OMN-17562): the boot skips the subscription and keeps the contract. The lazy
transaction read the empty attach result as a failure (``did not attach: [] []``), rolled the contract
back and retried it four times per message.

An exclusion the boot honours is not a failed attach. Any other empty attach still raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers import lazy_start_runtime_transaction as transaction_module
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
    NodeStart,
    RecordingBus,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

NAME = "projection_fixture_standalone_runner"


def _contract(tmp_path: Path) -> ModelDiscoveredContract:
    path = tmp_path / NAME / "contract.yaml"
    path.parent.mkdir()
    path.write_text(f"name: {NAME}\n", encoding="utf-8")
    return ModelDiscoveredContract(
        name=NAME,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name=NAME,
        package_name="omnimarket",
    )


def _transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: EnumContractAttachExclusionReason | None,
) -> LazyStartRuntimeTransaction:
    """A transaction whose subscribe step excludes the contract for ``reason`` and attaches nothing."""

    async def _subscribe(*_args: Any, **kwargs: Any) -> dict[str, tuple[str, ...]]:
        exclusions = kwargs.get("exclusions_out")
        if reason is not None and exclusions is not None:
            exclusions.append(
                ModelContractAttachExclusion(contract_name=NAME, reason=reason)
            )
        return {}

    monkeypatch.setattr(
        transaction_module, "subscribe_wired_contract_topics", _subscribe
    )
    bus = cast("ProtocolLazyStartBus", MagicMock())
    contract = _contract(tmp_path)
    transaction = LazyStartRuntimeTransaction(
        {NAME: contract},
        bus,
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
    )
    transaction._starts[NAME] = NodeStart(
        engine=MessageDispatchEngine(),
        manifest=ModelAutoWiringManifest(contracts=(contract,)),
        report=MagicMock(),
        bus=RecordingBus(bus),
    )
    return transaction


@pytest.mark.asyncio
async def test_attach_accepts_a_contract_the_boot_excludes_as_having_no_live_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _transaction(
        tmp_path, monkeypatch, EnumContractAttachExclusionReason.NO_LIVE_DISPATCHER
    )

    await transaction.attach(NAME)  # raised "did not attach: [] []" before OMN-26594


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        None,
        EnumContractAttachExclusionReason.NOT_WIRED,
        EnumContractAttachExclusionReason.RAW_EVENT_PROJECTION_WITHOUT_APPLIER,
    ],
)
async def test_attach_still_fails_when_nothing_attached_and_the_boot_would_have_subscribed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: EnumContractAttachExclusionReason | None,
) -> None:
    transaction = _transaction(tmp_path, monkeypatch, reason)

    with pytest.raises(RuntimeError, match="did not attach"):
        await transaction.attach(NAME)
