# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Broker ListGroups codes reach the CLI retry gate before dispatch (OMN-20376).

Only the broker admin boundary and terminal receipt writer are stand-ins. The
CLI, contract resolution, response decoding, liveness probes and retry/refusal
path run together, including the downstream subscription footprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from aiokafka.protocol.admin import DescribeGroupsResponse_v0

from omnibase_infra.backends import backend_probe
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.event_bus import kafka_auth

pytestmark = pytest.mark.integration

_COMMAND = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL = "onex.evt.omnimarket.delegate-skill-completed.v1"
_CHAIN = "onex.cmd.synthetic.delegation-request.v1"
_ROUTING = "onex.evt.synthetic.routing.v1"
_CHAIN_COMPLETED = "onex.evt.synthetic.delegation-completed.v1"
_CHAIN_FAILED = "onex.evt.synthetic.delegation-failed.v1"
_FIRST_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.0.0.__t." + _COMMAND
)
_CHAIN_BASE = "local.runtime_config.delegation-orchestrator"
_CHAIN_GROUP = f"{_CHAIN_BASE}.__t.{_CHAIN}"
_GROUPS = (_FIRST_GROUP, _CHAIN_GROUP, f"{_CHAIN_BASE}.__t.{_ROUTING}")


class _BrokerAdmin:
    def __init__(self, error_code: int) -> None:
        self.error_code = error_code
        self.closed = False
        self.described: list[str] = []
        self.listings = 0

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def describe_cluster(self) -> dict[str, Any]:
        return {"brokers": [{"node_id": 1}]}

    async def _send_request(self, request: Any, node_id: int) -> Any:
        assert node_id == 1
        prepared = request.prepare({16: (0, 4)})
        assert prepared.states_filter == ["Stable"]
        self.listings += 1
        response_type: Any = backend_probe._list_groups_v4_structs()[1]
        response = response_type(
            0,
            self.error_code,
            [(group, "consumer", "Stable", {}) for group in _GROUPS],
            {},
        )
        return response_type.decode(response.encode())

    async def describe_consumer_groups(self, group_ids: list[str]) -> list[Any]:
        self.described.extend(group_ids)
        return [
            DescribeGroupsResponse_v0(
                [(0, group, "Stable", "consumer", "", []) for group in group_ids]
            )
        ]


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    chain: bool,
    errors: list[int],
) -> tuple[list[_BrokerAdmin], list[float], list[dict[str, Any]]]:
    import aiokafka.admin

    contract: dict[str, Any] = {
        "name": "node_delegate_skill_orchestrator",
        "terminal_event": _TERMINAL,
        "event_bus": {"subscribe_topics": [_COMMAND], "publish_topics": [_TERMINAL]},
    }
    if chain:
        contract["delegation_runtime_dispatch"] = {
            "topics": {
                "command": _CHAIN,
                "completed": _CHAIN_COMPLETED,
                "failed": _CHAIN_FAILED,
            }
        }
        sibling = tmp_path / "node_delegation_orchestrator" / "contract.yaml"
        sibling.parent.mkdir()
        sibling.write_text(
            yaml.safe_dump(
                {
                    "name": "node_delegation_orchestrator",
                    "event_bus": {
                        "subscribe_topics": [_CHAIN, _ROUTING],
                        "publish_topics": [_CHAIN_COMPLETED, _CHAIN_FAILED],
                    },
                }
            ),
            encoding="utf-8",
        )
    path = tmp_path / "node_delegate_skill_orchestrator" / "contract.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")

    clients: list[_BrokerAdmin] = []
    sleeps: list[float] = []
    dispatched: list[dict[str, Any]] = []

    def admin(**_: Any) -> _BrokerAdmin:
        # Exhausting the script is a failure, rather than silently returning
        # success if the retry bound changes or a new probe is introduced.
        client = _BrokerAdmin(errors[len(clients)])
        clients.append(client)
        return client

    def receipt(**kwargs: Any) -> int:
        dispatched.append(kwargs)
        return 0

    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", admin)
    monkeypatch.setattr(kafka_auth, "build_aiokafka_auth_kwargs_for", lambda _: {})
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: path)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", receipt)
    monkeypatch.setattr(delegate_locus, "_sleep", sleeps.append)
    monkeypatch.setattr(delegate_locus, "_monotonic", lambda: sum(sleeps))
    return clients, sleeps, dispatched


def _delegate(tmp_path: Path) -> int:
    return run_delegate(
        prompt="coordinator retry integration probe",
        task_type="research",
        max_tokens=None,
        bus="kafka",
        locus=EnumDelegateLocus.DEPLOYED_LANE,
        kafka_bootstrap="broker.invalid:19092",
        state_root=tmp_path / "state",
        timeout=None,
        verbose=False,
        emit_socket=tmp_path / "emit.sock",
    )


@pytest.mark.parametrize("chain", [False, True], ids=["first-hop", "downstream"])
@pytest.mark.parametrize("error_code", [14, 15, 16])
def test_coordinator_recovery_dispatches_only_after_a_live_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chain: bool, error_code: int
) -> None:
    errors = ([0] if chain else []) + [error_code, 0]
    clients, sleeps, dispatched = _prepare(
        tmp_path, monkeypatch, chain=chain, errors=errors
    )

    assert _delegate(tmp_path) == 0

    assert len(clients) == len(errors)
    assert all(client.closed and client.listings == 1 for client in clients)
    assert clients[-2].described == []
    assert sleeps == [2.0]
    assert len(dispatched) == 1
    assert dispatched[0]["host_handlers"] is False
    decision = dispatched[0]["locus_decision"]
    assert decision.lane_consumer_groups == (_FIRST_GROUP,)
    assert decision.downstream_consumer_groups == ((_CHAIN_GROUP,) if chain else ())
    assert clients[-1].described == [(_CHAIN_GROUP if chain else _FIRST_GROUP)]


@pytest.mark.parametrize("chain", [False, True], ids=["first-hop", "downstream"])
@pytest.mark.parametrize("error_code", [14, 15, 16, 31])
def test_unanswered_or_denied_listing_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chain: bool, error_code: int
) -> None:
    attempts = 1 if error_code == 31 else 3
    errors = ([0] if chain else []) + [error_code] * attempts
    clients, sleeps, dispatched = _prepare(
        tmp_path, monkeypatch, chain=chain, errors=errors
    )

    with pytest.raises(click.ClickException, match="cannot confirm") as caught:
        _delegate(tmp_path)

    assert f"Error {error_code}" in str(caught.value)
    if error_code != 31:
        assert "3 attempts" in str(caught.value)
    assert len(clients) == len(errors)
    assert all(client.closed and client.listings == 1 for client in clients)
    assert all(client.described == [] for client in clients[int(chain) :])
    assert sleeps == ([] if error_code == 31 else [2.0, 4.0])
    assert dispatched == []
