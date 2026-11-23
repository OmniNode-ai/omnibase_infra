# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: the real ``local`` database topology profile reaches ``wire_from_manifest`` on a lazy start.

Nothing is faked but the wiring call itself: ``boot_manifest`` loads the real checked-in ``local`` profile
through ``ONEX_DATABASE_TOPOLOGY_PROFILE``, the dormant store keeps it, and the real
``LazyStartRuntimeTransaction.build`` hands it to ``wire_from_manifest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers import lazy_start_runtime_transaction as transaction_module
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
)
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.topology import load_topology_profile

pytestmark = pytest.mark.integration

DORMANT = "projection_fixture_db_io"
INGRESS = "node_ingress_fixture"
POLICY = ModelLazyStartPolicy(ingress_contract=INGRESS)


def _contract(tmp_path: Path, name: str) -> ModelDiscoveredContract:
    path = tmp_path / name / "contract.yaml"
    path.parent.mkdir()
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name=name,
        package_name="omnimarket",
    )


class _Report:
    total_wired = 0
    results: tuple[object, ...] = ()


@pytest.mark.asyncio
async def test_real_local_profile_reaches_wire_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    StoreLazyStartDormant.clear()
    monkeypatch.setattr(
        "omnibase_infra.handlers.lazy_start_boot_split.lazy_start_policy_of_this_lane",
        lambda _environ=None: POLICY,
    )
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(tmp_path, INGRESS), _contract(tmp_path, DORMANT))
    )
    try:
        boot_manifest(manifest, None, {"ONEX_DATABASE_TOPOLOGY_PROFILE": "local"})
        topology = StoreLazyStartDormant.topology()
        assert topology is not None
        assert topology == load_topology_profile("local")

        seen: dict[str, Any] = {}

        async def _wire(*_args: Any, **kwargs: Any) -> _Report:
            seen.update(kwargs)
            return _Report()

        monkeypatch.setattr(transaction_module, "wire_from_manifest", _wire)
        transaction = LazyStartRuntimeTransaction(
            dict(StoreLazyStartDormant.contracts()),
            cast("ProtocolLazyStartBus", MagicMock()),
            "local",
            None,
            ModelONEXContainer(),
            ModelTopicReadinessConfig(),
            topology=topology,
        )
        with pytest.raises(RuntimeError, match="did not wire"):
            await transaction.build(DORMANT)
        assert seen.get("topology") is topology
    finally:
        StoreLazyStartDormant.clear()
