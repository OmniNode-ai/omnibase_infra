# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex lease acquire|renew|release|show`` against a three-member etcd (OMN-21424).

The three commands travel the bus: the verb publishes each on the orchestrator's
command topic and prints the orchestrator's answer from its event topics. The
bus here is an in-memory bus on which the kernel's ``wire_from_manifest`` has
wired the real ``node_lease_orchestrator`` contract, so the consumer is the
orchestrator's own handlers over a real etcd. Each simulated host runs its
orchestrator against its own etcd member, so a grant made through one host is
seen, and refused to a second holder, through another. ``show`` reads etcd
directly and writes nothing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import entry_points
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.lease import lease_lane_bus
from omnibase_infra.handlers.lease.lease_cli import lease_group
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.integration.runtime.placement.conftest import EtcdCluster

pytestmark = [pytest.mark.integration]

_ORCHESTRATOR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_lease_orchestrator"
    / "contract.yaml"
)


@asynccontextmanager
async def _orchestrator_bus(
    lane: str, omnibase_path: Path | None
) -> AsyncIterator[EventBusInmemory]:
    """The lane's bus with the real lease orchestrator wired onto it by the kernel."""
    manifest = discover_contracts_from_paths([_ORCHESTRATOR])
    assert manifest.errors == ()
    bus = EventBusInmemory(environment="test", group="omn-21424-cli")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        report = await wire_from_manifest(
            manifest, engine, event_bus=bus, environment="local"
        )
        engine.freeze()
        assert report.quarantined_handlers == (), report
        yield bus
    finally:
        await bus.close()


@pytest.fixture(autouse=True)
def _lane_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lease_lane_bus, "open_lane_bus", _orchestrator_bus)


KINDS = "exclusive:1:30"


def _run(endpoint: str, *args: str) -> Result:
    return CliRunner().invoke(
        lease_group,
        ["--lane", "dev", *args],
        env={"ONEX_LEASE_ETCD_ENDPOINT": endpoint, "ONEX_LEASE_KINDS": KINDS},
        catch_exceptions=False,
    )


def _reply(result: Result) -> dict[str, object]:
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(loaded, dict)
    return loaded


def test_the_verb_is_registered_on_the_onex_cli() -> None:
    names = {
        ep.name: ep.value for ep in entry_points(group="onex.cli") if ep.name == "lease"
    }
    assert names == {"lease": "omnibase_infra.handlers.lease.lease_cli:lease_group"}


def test_acquire_show_renew_release_across_hosts(etcd_cluster: EtcdCluster) -> None:
    host0, host1, host2 = (m.endpoint for m in etcd_cluster.members)
    request_id = str(uuid4())
    acquire = ("acquire", "--kind", "exclusive", "--resource", "pr:cli-omn21424")

    first = _run(host0, *acquire, "--holder", "lane-a", "--request-id", request_id)
    assert first.exit_code == 0, first.output
    granted = _reply(first)
    assert granted["outcome"] == "lease-granted"
    assert int(str(granted["fence"])) > 0

    again = _run(host1, *acquire, "--holder", "lane-a", "--request-id", request_id)
    assert again.exit_code == 0, again.output
    assert _reply(again)["lease_id"] == granted["lease_id"]
    assert _reply(again)["fence"] == granted["fence"]

    other = _run(host1, *acquire, "--holder", "lane-b")
    assert other.exit_code == 3, other.output
    assert _reply(other)["outcome"] == "lease-refused"

    shown = _run(host2, "show", "--kind", "exclusive", "--resource", "pr:cli-omn21424")
    assert shown.exit_code == 0, shown.output
    held = [json.loads(line) for line in shown.stdout.strip().splitlines()]
    assert [(h["lease_id"], h["holder"]) for h in held] == [
        (granted["lease_id"], "lane-a")
    ]

    lease = (
        "--kind",
        "exclusive",
        "--resource",
        "pr:cli-omn21424",
        "--lease-id",
        str(granted["lease_id"]),
    )
    renewed = _run(host2, "renew", *lease)
    assert renewed.exit_code == 0, renewed.output
    assert _reply(renewed)["outcome"] == "lease-renewed"
    assert _reply(renewed)["fence"] == granted["fence"]

    released = _run(host0, "release", *lease)
    assert released.exit_code == 0, released.output
    assert _reply(released)["outcome"] == "lease-released"

    gone = _run(host1, "renew", *lease)
    assert gone.exit_code == 3, gone.output
    assert _reply(gone)["outcome"] == "lease-expired"

    empty = _run(host2, "show", "--kind", "exclusive", "--resource", "pr:cli-omn21424")
    assert empty.exit_code == 0, empty.output
    assert empty.stdout.strip() == ""


def test_an_undeclared_kind_is_refused_by_the_orchestrator_and_a_missing_endpoint_fails_loudly(
    etcd_cluster: EtcdCluster,
) -> None:
    undeclared = _run(
        etcd_cluster.members[0].endpoint,
        "acquire",
        "--kind",
        "nope",
        "--resource",
        "r",
        "--holder",
        "h",
    )
    assert undeclared.exit_code == 3, undeclared.output
    assert _reply(undeclared)["outcome"] == "lease-refused"
    assert "nope" in str(_reply(undeclared)["reason"])

    missing = CliRunner().invoke(
        lease_group,
        ["show", "--kind", "exclusive", "--resource", "r"],
        env={"ONEX_LEASE_ETCD_ENDPOINT": "", "ONEX_LEASE_KINDS": KINDS},
    )
    assert missing.exit_code == 2, missing.output
    assert "ONEX_LEASE_ETCD_ENDPOINT" in missing.output
