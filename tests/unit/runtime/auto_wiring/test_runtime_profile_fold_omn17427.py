# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime loads the folded roles' nodes, and one node failing stops no other (OMN-17427).

A lane folds the effects, worker, tenant-projection and projection-writer roles
into its main runtime by naming them in ``ONEX_RUNTIME_FOLDED_PROFILES``. The
ownership filter the kernel and the health monitor both read must then attach
every contract those roles own, alongside main's own, in the one process.

The second half pins the isolation the container split used to give: nodes in
one process share the bus, so a node whose handler raises must not stop the
delivery to any other node.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.runtime.runtime_profile import resolve_runtime_folded_profiles

pytestmark = pytest.mark.unit

_FOLDED = (
    "effects",
    "workers",
    "tenant-projection",
    "projection-writer-delegation",
    "projection-writer-registration",
)
_FOLD_ENV = {"ONEX_RUNTIME_FOLDED_PROFILES": ",".join(_FOLDED)}

_NODES_DIR = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"


def _contract(
    name: str, runtime_profiles: tuple[str, ...] = ()
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_profiles=runtime_profiles,
    )


def _manifest() -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(
        contracts=(
            _contract("legacy_node"),
            _contract("main_node", ("main",)),
            _contract("effect_node", ("effects",)),
            _contract("worker_node", ("workers",)),
            _contract("tenant_writer", ("tenant-projection",)),
            _contract("delegation_writer", ("projection-writer-delegation",)),
            _contract("registration_writer", ("projection-writer-registration",)),
            _contract("canary_node", ("canary",)),
        )
    )


def test_folded_main_runtime_loads_every_folded_node() -> None:
    result = filter_manifest_for_runtime_profile(_manifest(), "main", environ=_FOLD_ENV)

    loaded = {contract.name for contract in result.manifest.contracts}
    assert loaded == {
        "legacy_node",
        "main_node",
        "effect_node",
        "worker_node",
        "tenant_writer",
        "delegation_writer",
        "registration_writer",
    }
    assert result.folded_profiles == _FOLDED
    assert set(result.folded_contracts) == loaded - {"legacy_node", "main_node"}
    assert result.skipped_contracts == ("canary_node",)


def test_unfolded_main_runtime_is_unchanged() -> None:
    result = filter_manifest_for_runtime_profile(_manifest(), "main", environ={})

    assert {c.name for c in result.manifest.contracts} == {"legacy_node", "main_node"}
    assert result.folded_profiles == ()
    assert result.folded_contracts == ()


def test_fold_reaches_every_real_effects_and_worker_contract() -> None:
    """Against this repo's own contracts on disk, not a fixture."""
    paths = sorted(_NODES_DIR.glob("*/contract.yaml"))
    discovered = discover_contracts_from_paths(paths)
    folded_role_contracts = {
        contract.name
        for contract in discovered.contracts
        if set(contract.runtime_profiles) & {"effects", "workers"}
    }
    assert folded_role_contracts, "no effects/workers contract discovered"

    unfolded = filter_manifest_for_runtime_profile(discovered, "main", environ={})
    folded = filter_manifest_for_runtime_profile(
        discovered,
        "main",
        environ={"ONEX_RUNTIME_FOLDED_PROFILES": "effects,workers"},
    )

    unfolded_names = {c.name for c in unfolded.manifest.contracts}
    folded_names = {c.name for c in folded.manifest.contracts}
    lane_excluded = set(folded.lane_excluded_contracts)
    assert not folded_role_contracts & unfolded_names
    assert folded_role_contracts - lane_excluded <= folded_names
    assert unfolded_names <= folded_names


def test_raw_contract_ownership_honours_the_fold() -> None:
    raw = {"name": "effect_node", "runtime_profiles": ["effects"]}

    assert not runtime_profile_owns_contract(raw, "main", environ={})
    assert runtime_profile_owns_contract(raw, "main", environ=_FOLD_ENV)


def test_unknown_folded_profile_is_refused() -> None:
    with pytest.raises(
        ProtocolConfigurationError, match="ONEX_RUNTIME_FOLDED_PROFILES"
    ):
        resolve_runtime_folded_profiles({"ONEX_RUNTIME_FOLDED_PROFILES": "efects"})


def test_blank_folded_profiles_resolve_to_none() -> None:
    assert (
        resolve_runtime_folded_profiles({"ONEX_RUNTIME_FOLDED_PROFILES": " , "}) == ()
    )
    assert resolve_runtime_folded_profiles({}) == ()


def _identity(node_name: str) -> ModelNodeIdentity:
    return ModelNodeIdentity(
        env="test", service="omnibase-infra", node_name=node_name, version="1.0.0"
    )


@pytest.mark.asyncio
async def test_one_node_raising_does_not_stop_the_others() -> None:
    bus = EventBusInmemory(environment="test", group="fold")
    await bus.start()
    delivered: dict[str, list[bytes]] = {"writer_a": [], "writer_b": [], "effect": []}
    raised: list[bytes] = []

    async def crashing(message: ModelEventMessage) -> None:
        raised.append(message.value)
        raise RuntimeError("node handler crashed")

    def recorder(name: str) -> Callable[[ModelEventMessage], Awaitable[None]]:
        async def record(message: ModelEventMessage) -> None:
            delivered[name].append(message.value)

        return record

    shared_topic = "onex.evt.test.fold-shared.v1"
    other_topic = "onex.evt.test.fold-other.v1"
    await bus.subscribe(shared_topic, _identity("node_crashing"), crashing)
    await bus.subscribe(shared_topic, _identity("node_writer_a"), recorder("writer_a"))
    await bus.subscribe(shared_topic, _identity("node_writer_b"), recorder("writer_b"))
    await bus.subscribe(other_topic, _identity("node_effect"), recorder("effect"))
    try:
        for index in range(3):
            await bus.publish(shared_topic, None, f"s{index}".encode())
            await bus.publish(other_topic, None, f"o{index}".encode())
    finally:
        await bus.close()

    assert raised == [b"s0", b"s1", b"s2"]
    assert delivered["writer_a"] == [b"s0", b"s1", b"s2"]
    assert delivered["writer_b"] == [b"s0", b"s1", b"s2"]
    assert delivered["effect"] == [b"o0", b"o1", b"o2"]


_POLICY_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "contracts"
    / "services"
    / "runtime_policy.contract.yaml"
)


def _policy_with_dev_main_fold(folded: list[str]) -> ModelRuntimePolicyContract:
    raw = yaml.safe_load(_POLICY_CONTRACT.read_text(encoding="utf-8"))
    raw["profiles"]["dev"]["services"]["main"]["folded_profiles"] = folded
    return ModelRuntimePolicyContract.model_validate(raw)


def test_policy_contract_renders_a_declared_fold() -> None:
    from scripts.render_runtime_policy_env import render_env

    env = render_env(_policy_with_dev_main_fold(["effects", "workers"]))

    assert env["DEV_RUNTIME_MAIN_FOLDED_PROFILES"] == "effects,workers"
    assert resolve_runtime_folded_profiles(
        {"ONEX_RUNTIME_FOLDED_PROFILES": env["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]}
    ) == ("effects", "workers")


def test_policy_contract_without_a_fold_renders_no_fold_key() -> None:
    from scripts.render_runtime_policy_env import render_env

    raw = yaml.safe_load(_POLICY_CONTRACT.read_text(encoding="utf-8"))
    env = render_env(ModelRuntimePolicyContract.model_validate(raw))

    assert not [key for key in env if key.endswith("_FOLDED_PROFILES")]


def test_policy_contract_refuses_an_unregistered_fold() -> None:
    with pytest.raises(ValidationError, match="folded_profiles"):
        _policy_with_dev_main_fold(["efects"])
