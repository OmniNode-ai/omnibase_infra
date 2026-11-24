# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: a lazy start wires the real ``node_runtime_manifest_reducer`` contract with the kernel's dependency map.

Nothing is faked but the Postgres pool: the contract is read from disk, the dependency map comes from
the kernel's own builder, and the real ``LazyStartRuntimeTransaction.build`` wires the handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartHandlerDependencies,
)
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.integration

NAME = "node_runtime_manifest_reducer"
CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes"
    / NAME
    / "contract.yaml"
)


@pytest.mark.asyncio
async def test_real_contract_wires_with_the_recorded_kernel_dependencies() -> None:
    StoreLazyStartHandlerDependencies.record(None)
    try:
        built = _build_runtime_handler_dependencies(MagicMock(name="postgres_pool"))
        assert built is not None
        recorded = StoreLazyStartHandlerDependencies.dependencies()
        assert recorded is built

        manifest = discover_contracts_from_paths([CONTRACT])
        assert manifest.total_errors == 0
        contract = next(c for c in manifest.contracts if c.name == NAME)
        transaction = LazyStartRuntimeTransaction(
            {NAME: contract},
            cast("ProtocolLazyStartBus", MagicMock()),
            "local",
            None,
            ModelONEXContainer(),
            ModelTopicReadinessConfig(),
            handler_dependencies=recorded,
        )
        await transaction.build(NAME)
        assert transaction.node_engine(NAME) is not None
    finally:
        StoreLazyStartHandlerDependencies.record(None)
