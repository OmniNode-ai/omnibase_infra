# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot step 0 builds the node catalog before the container exists (OMN-21590, A2).

Every rule has a planted violation beside a control. The fixture package set is two
fake distributions whose metadata lists three contracts, one with Windows line endings.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
import yaml

import omnibase_infra
from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_announce_refusal_reason import EnumAnnounceRefusalReason
from omnibase_core.enums.enum_catalog_entry_state import EnumCatalogEntryState
from omnibase_core.models.contracts.subcontracts.model_fsm_subcontract import (
    ModelFSMSubcontract,
)
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.nodes.package_announce.model_catalog_entry import (
    ModelCatalogEntry,
)
from omnibase_core.models.nodes.package_announce.model_node_announce_policy import (
    ModelNodeAnnouncePolicy,
)
from omnibase_core.validation.validator_fsm_analysis import analyze_fsm
from omnibase_infra.handlers.handler_package_announcer import (
    HandlerPackageAnnouncer,
    announcer_identity,
    build_catalog_direct,
    load_boot_runtime_config,
    read_package_inventory,
    run_boot_step0,
)
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.models.node_announce.model_package_announce_request import (
    ModelPackageAnnounceRequest,
)
from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)
from omnibase_infra.topics import SUFFIX_CONTRACT_REGISTERED

_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_ANNOUNCER = announcer_identity("test.step0")


def _contract(name: str, *, handler_module: str = "omnibase_infra.fixture") -> str:
    return textwrap.dedent(f"""\
        name: "{name}"
        node_type: "EFFECT_GENERIC"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        node_version: "1.0.0"
        description: "fixture"
        runtime_profiles: [main]
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-in.v1"
          publish_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-out.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - operation: run
              handler:
                name: HandlerFixture
                module: {handler_module}
    """)


def _entry_point(
    name: str, dist: str, version: str, contract_path: Path | None
) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist
    ep.dist.version = version
    node = type("FixtureNode", (), {"contract_path": contract_path})
    ep.load.return_value = node
    return ep


@pytest.fixture
def three_contracts(tmp_path: Path) -> list[MagicMock]:
    """Two fixture distributions, three contracts; one file has Windows line endings."""
    paths = {}
    for name in ("node_alpha_effect", "node_beta_effect", "node_gamma_effect"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "contract.yaml"
        text = _contract(name)
        if name == "node_gamma_effect":
            path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        else:
            path.write_text(text, encoding="utf-8")
        paths[name] = path
    return [
        _entry_point(
            "node_alpha_effect", "pkg-one", "1.0.0", paths["node_alpha_effect"]
        ),
        _entry_point("node_beta_effect", "pkg-one", "1.0.0", paths["node_beta_effect"]),
        _entry_point(
            "node_gamma_effect", "pkg-two", "2.1.0", paths["node_gamma_effect"]
        ),
    ]


def _request(
    *,
    workload: str = "main",
    announcer_id: UUID = _ANNOUNCER,
    policy: ModelNodeAnnouncePolicy | None = None,
) -> ModelPackageAnnounceRequest:
    return ModelPackageAnnounceRequest(
        topic=ModelNodeAnnouncementTopic.suffix_for(workload),
        announcer_id=announcer_id,
        generation=1,
        announced_at=datetime.now(UTC),
        policy=policy,
    )


def _entry(result: ModelPackageAnnounceResult, name: str) -> ModelCatalogEntry:
    entry = result.catalog.lookup(name)
    assert entry is not None, f"no catalog entry for {name}"
    return entry


class RecordingBus:
    """A second bus binding: records every publish and delivers like a durable bus would."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []
        self._subscribers: dict[str, list[Callable[[object], Awaitable[None]]]] = {}

    async def start(self) -> None:
        return None

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        self.published.append((topic, key, value))
        for callback in list(self._subscribers.get(topic, [])):
            await callback(SimpleNamespace(value=value, topic=topic))

    async def subscribe(
        self,
        topic: str,
        *,
        on_message: Callable[[object], Awaitable[None]],
        group_id: str,
    ) -> Callable[[], Awaitable[None]]:
        self._subscribers.setdefault(topic, []).append(on_message)

        async def unsubscribe() -> None:
            self._subscribers[topic].remove(on_message)

        return unsubscribe


@pytest.mark.unit
@pytest.mark.asyncio
async def test_step0_announces_exactly_three_with_canonical_hashes(
    three_contracts: list[MagicMock],
) -> None:
    with patch(_EP, return_value=three_contracts):
        result = await run_boot_step0(workload="main")

    assert len(result.announcements) == 3
    for announcement in result.announcements:
        path = next(
            r.contract_path
            for r in result.inventory.records
            if r.contract_name == announcement.entity_id
        )
        assert path is not None
        assert (
            announcement.content_hash
            == ModelNodeAnnouncement.canonical_content_hash(path.read_bytes())
        )
    held = {name: e.content_hash for name, e in result.catalog.entries.items()}
    assert set(held) == {"node_alpha_effect", "node_beta_effect", "node_gamma_effect"}
    assert all(
        e.state is EnumCatalogEntryState.ADMITTED
        for e in result.catalog.entries.values()
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_step0_completes_with_the_container_constructor_replaced_by_one_that_raises(
    three_contracts: list[MagicMock],
) -> None:
    """Falsifier: the container constructor is called while step 0 runs."""
    boom = MagicMock(side_effect=AssertionError("container constructed in step 0"))
    with (
        patch(_EP, return_value=three_contracts),
        patch.object(ModelONEXContainer, "__init__", boom),
    ):
        result = await run_boot_step0(workload="main")

    assert len(result.catalog.entries) == 3
    boom.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_boot_publishes_nothing_on_contract_registered(
    three_contracts: list[MagicMock],
) -> None:
    """AC4: step 0 publishes on the announcement topic and on no other."""
    bus = RecordingBus()
    with patch(_EP, return_value=three_contracts):
        await HandlerPackageAnnouncer().handle(
            _request(),
            bus=bus,
            settle_timeout_seconds=load_boot_runtime_config().catalog_settle_timeout_seconds,
        )

    topics = {topic for topic, _, _ in bus.published}
    assert topics == {ModelNodeAnnouncementTopic.suffix_for("main")}
    assert SUFFIX_CONTRACT_REGISTERED not in topics
    assert len(bus.published) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_catalog_is_the_same_on_a_second_bus_binding(
    three_contracts: list[MagicMock],
) -> None:
    with patch(_EP, return_value=three_contracts):
        on_memory = await run_boot_step0(workload="main")
        on_recording = await HandlerPackageAnnouncer().handle(
            _request(announcer_id=on_memory.announcements[0].announcer_id),
            bus=RecordingBus(),
            settle_timeout_seconds=load_boot_runtime_config().catalog_settle_timeout_seconds,
        )
        direct = build_catalog_direct(
            read_package_inventory(None),
            _request(announcer_id=on_memory.announcements[0].announcer_id),
        )

    assert on_memory.catalog.entries == on_recording.catalog.entries
    assert on_memory.catalog.entries == direct.catalog.entries


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_malformed_contract_is_a_parse_error_with_no_catalog_entry(
    three_contracts: list[MagicMock], tmp_path: Path
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "contract.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    eps = [
        *three_contracts,
        _entry_point("node_bad", "pkg-bad", "0.1.0", bad / "contract.yaml"),
    ]
    discover_contracts_cache_clear()

    with patch(_EP, return_value=eps):
        result = await run_boot_step0(workload="main")
        manifest = discover_contracts(result)

    assert len(result.catalog.entries) == 3
    assert [e.reason for e in manifest.errors] == ["parse_error"]
    assert "Expected YAML dict, got list" in manifest.errors[0].error
    assert manifest.total_discovered == 3
    discover_contracts_cache_clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_handler_outside_the_trusted_prefixes_is_refused_with_a_typed_reason(
    three_contracts: list[MagicMock], tmp_path: Path
) -> None:
    outsider = tmp_path / "outsider"
    outsider.mkdir()
    (outsider / "contract.yaml").write_text(
        _contract("node_outsider_effect", handler_module="evil_pkg.handlers"),
        encoding="utf-8",
    )
    eps = [
        *three_contracts,
        _entry_point(
            "node_outsider_effect", "pkg-x", "9.9.9", outsider / "contract.yaml"
        ),
    ]
    policy = ModelNodeAnnouncePolicy(
        trusted_handler_prefixes=("omnibase_infra",),
        active_profiles=("main",),
        authoritative_announcers={},
    )
    discover_contracts_cache_clear()

    with patch(_EP, return_value=eps):
        result = await HandlerPackageAnnouncer().handle(
            _request(policy=policy),
            bus=RecordingBus(),
            settle_timeout_seconds=load_boot_runtime_config().catalog_settle_timeout_seconds,
        )
        manifest = discover_contracts(result)

    entry = result.catalog.lookup("node_outsider_effect")
    assert entry is not None
    assert entry.state is EnumCatalogEntryState.REFUSED
    assert (
        entry.refusal_reason
        is EnumAnnounceRefusalReason.HANDLER_OUTSIDE_TRUSTED_PREFIXES
    )
    assert _entry(result, "node_alpha_effect").state is EnumCatalogEntryState.ADMITTED
    assert "refused by catalog admission" in manifest.errors[0].error
    assert {c.name for c in manifest.contracts} == {
        "node_alpha_effect",
        "node_beta_effect",
        "node_gamma_effect",
    }
    discover_contracts_cache_clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_duplicate_name_keeps_the_first_package_and_reports_the_second(
    three_contracts: list[MagicMock],
) -> None:
    duplicate = _entry_point(
        "node_alpha_effect_copy",
        "pkg-copy",
        "3.0.0",
        three_contracts[0].load().contract_path,
    )
    discover_contracts_cache_clear()
    with patch(_EP, return_value=[*three_contracts, duplicate]):
        result = await run_boot_step0(workload="main")
        manifest = discover_contracts(result)

    assert _entry(result, "node_alpha_effect").package.distribution == "pkg-one"
    assert result.catalog.stale_generation_rejections == 1
    assert [e.reason for e in manifest.errors] == ["duplicate_contract_name"]
    discover_contracts_cache_clear()


@pytest.mark.unit
def test_discovery_with_no_step0_reads_the_same_catalog(
    three_contracts: list[MagicMock],
) -> None:
    discover_contracts_cache_clear()
    with patch(_EP, return_value=three_contracts):
        manifest = discover_contracts()
    assert [c.name for c in manifest.contracts] == [
        "node_alpha_effect",
        "node_beta_effect",
        "node_gamma_effect",
    ]
    discover_contracts_cache_clear()


@pytest.mark.unit
def test_no_module_the_container_side_imports_names_an_entry_point_call() -> None:
    """AC3: the scan lives in the announcer's metadata reader and nowhere the container imports."""
    root = str(Path(omnibase_infra.__file__).parent.parent)
    probe = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {root!r})
        import importlib
        importlib.import_module("omnibase_infra.runtime.auto_wiring")
        importlib.import_module("omnibase_core.container")
        for name, module in sorted(sys.modules.items()):
            if name.startswith("omnibase_infra.runtime") and getattr(module, "__file__", None):
                print(name, module.__file__)
    """)
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out, "the positive control: the probe must list the imported runtime modules"
    assert all(line.split(" ", 1)[1].startswith(root) for line in out)
    offenders = [
        f"{name}:{number}"
        for line in out
        for name, file in [line.split(" ", 1)]
        for number, text in enumerate(Path(file).read_text().splitlines(), 1)
        if "entry_points(" in text
    ]
    assert offenders == []


@pytest.mark.unit
def test_discovery_module_names_no_entry_point_group() -> None:
    from omnibase_infra.runtime.auto_wiring import discovery

    source = Path(discovery.__file__).read_text()
    assert "import entry_points" not in source
    assert "entry_points(" not in source


@pytest.mark.unit
def test_the_kernel_runs_step0_before_the_container_and_reads_the_catalog() -> None:
    from omnibase_infra.runtime import service_kernel

    source = Path(service_kernel.__file__).read_text()
    step0 = source.index("await run_boot_step0(")
    container = source.index("container = ModelONEXContainer()")
    discovery = source.index("discover_contracts(node_catalog_result)")
    assert step0 < container < discovery
    assert "publish_contract_registrations" not in source


@pytest.mark.unit
def test_the_catalog_entry_lifecycle_is_a_loadable_fsm_with_every_state_reachable() -> (
    None
):
    contract = yaml.safe_load(
        (
            Path(omnibase_infra.__file__).parent
            / "nodes"
            / "node_package_announce_effect"
            / "contract.yaml"
        ).read_text()
    )
    fsm = ModelFSMSubcontract.model_validate(contract["state_transitions"])
    assert fsm.initial_state == "absent"
    assert not analyze_fsm(fsm).unreachable_states
    assert [t.transition_name for t in fsm.transitions].count("withdraw_unknown") == 1


@pytest.mark.unit
def test_planted_violation_a_catalog_state_with_no_inbound_edge_is_caught() -> None:
    contract = yaml.safe_load(
        (
            Path(omnibase_infra.__file__).parent
            / "nodes"
            / "node_package_announce_effect"
            / "contract.yaml"
        ).read_text()
    )
    block = contract["state_transitions"]
    block["states"].append(
        {
            "version": block["state_machine_version"],
            "state_name": "orphan",
            "state_type": "operational",
            "description": "planted",
        }
    )
    assert analyze_fsm(
        ModelFSMSubcontract.model_validate(block)
    ).unreachable_states == ["orphan"]


def test_announcer_identity_is_stable_and_a_uuid() -> None:
    assert announcer_identity("a") == announcer_identity("a")
    assert announcer_identity("a") != announcer_identity("b")
    assert isinstance(announcer_identity("a"), UUID)
    assert uuid4() != announcer_identity("a")
