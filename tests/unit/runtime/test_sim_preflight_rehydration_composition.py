# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host replay refuses a wrong target and preserves exact raw publisher bytes."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    _RECEIPT_MINT,
    VerifiedSimArchiveRehydrationPlan,
)
from omnibase_infra.runtime.sim_preflight_rehydration import (
    KafkaSimPreflightRawPublisher,
    ModelSimPreflightRehydrationConfig,
    rehydrate_verified_sim_preflight_chain,
    verify_live_sim_preflight_target,
)
from tests.unit.runtime.test_sim_archive_source_receipt_omn19728 import (
    _fixture,
    _topology,
)


def _config(**overrides: object) -> ModelSimPreflightRehydrationConfig:
    raw: dict[str, object] = {
        "source_databases": ModelExecutionGraphReadDatabases(
            analytics_dsn=SecretStr(
                "postgresql://readonly@source.example:5432/omnidash_analytics"
            ),
            ledger_dsn=SecretStr(
                "postgresql://readonly@source.example:5432/omnibase_infra"
            ),
        ),
        "source_kafka": ModelKafkaEventBusConfig(
            bootstrap_servers="source.example:9092", environment="source"
        ),
        "source_topic_namespace": "",
        "target_ledger_dsn": SecretStr(
            "postgresql://target@127.0.0.1:65036/omnibase_infra"
        ),
        "topology_version": _topology().version,
    }
    raw.update(overrides)
    return ModelSimPreflightRehydrationConfig.model_validate(raw)


def _inspect_document() -> list[dict[str, object]]:
    return [
        {
            "Name": f"/omnibase-infra-sim-preflight-{service}",
            "Config": {
                "Labels": {
                    "com.docker.compose.project": "omnibase-infra-sim-preflight",
                    "com.docker.compose.service": service,
                }
            },
            "State": {"Running": True, "Status": "running"},
            "NetworkSettings": {
                "Ports": {
                    container_port: [{"HostIp": "127.0.0.1", "HostPort": host_port}]
                }
            },
        }
        for service, container_port, host_port in (
            ("postgres", "5432/tcp", "65036"),
            ("redpanda", "19092/tcp", "65092"),
        )
    ]


@pytest.fixture(autouse=True)
def _sim_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "sim-202")


@pytest.mark.unit
def test_config_rejects_nonlocal_or_same_source_target_without_exposing_dsn() -> None:
    with pytest.raises(ValidationError, match="exact local preflight ledger") as exc:
        _config(target_ledger_dsn=SecretStr("postgresql://secret@other:5432/db"))
    assert "secret" not in str(exc.value)
    with pytest.raises(ValidationError, match="source broker must differ"):
        _config(
            source_kafka=ModelKafkaEventBusConfig(
                bootstrap_servers="127.0.0.1:65092", environment="source"
            )
        )
    with pytest.raises(ValidationError, match="exact local preflight ledger"):
        _config(
            target_ledger_dsn=SecretStr(
                "postgresql://target@127.0.0.1:65036/omnibase_infra?host=other"
            )
        )
    with pytest.raises(ValidationError, match="forbids overrides"):
        _config(
            source_databases=ModelExecutionGraphReadDatabases(
                analytics_dsn=SecretStr(
                    "postgresql://source:5432/analytics?port=65036"
                ),
                ledger_dsn=SecretStr("postgresql://source:5432/ledger"),
            )
        )


@pytest.mark.unit
def test_target_inspection_requires_running_exact_project_and_loopback_ports() -> None:
    document = _inspect_document()
    with patch(
        "omnibase_infra.runtime.sim_preflight_rehydration.subprocess.run",
        return_value=SimpleNamespace(stdout=json.dumps(document)),
    ) as run:
        verify_live_sim_preflight_target()
    assert run.call_args.args[0] == [
        "docker",
        "inspect",
        "omnibase-infra-sim-preflight-postgres",
        "omnibase-infra-sim-preflight-redpanda",
    ]

    for mutation in ("project", "status", "port"):
        changed = _inspect_document()
        if mutation == "project":
            changed[0]["Config"]["Labels"]["com.docker.compose.project"] = "dev"  # type: ignore[index]
        elif mutation == "status":
            changed[1]["State"]["Running"] = False  # type: ignore[index]
        else:
            changed[1]["NetworkSettings"]["Ports"]["19092/tcp"][0]["HostIp"] = (
                "192.0.2.1"  # type: ignore[index]
            )
        with patch(
            "omnibase_infra.runtime.sim_preflight_rehydration.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps(changed)),
        ):
            with pytest.raises(ValueError, match="exact running project"):
                verify_live_sim_preflight_target()
    malformed = _inspect_document()
    malformed[0]["Config"] = None
    with patch(
        "omnibase_infra.runtime.sim_preflight_rehydration.subprocess.run",
        return_value=SimpleNamespace(stdout=json.dumps(malformed)),
    ):
        with pytest.raises(ValueError, match="malformed"):
            verify_live_sim_preflight_target()
    with patch(
        "omnibase_infra.runtime.sim_preflight_rehydration.subprocess.run",
        side_effect=subprocess.TimeoutExpired("docker inspect", 10),
    ):
        with pytest.raises(ValueError, match="inspection failed"):
            verify_live_sim_preflight_target()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_raw_publisher_preserves_duplicate_header_order_and_awaits_ack() -> None:
    producer = MagicMock()
    producer.send_and_wait = AsyncMock(return_value=object())
    publisher = KafkaSimPreflightRawPublisher(producer)
    headers = [("first", b"one"), ("first", b"two")]

    await publisher.publish(
        "onex.cmd.omnimarket.delegate-skill.v1",
        key=b"\x00key",
        value=b"\xffvalue",
        headers=headers,
        timestamp_ms=1_800_000_000_123,
    )

    producer.send_and_wait.assert_awaited_once_with(
        "onex.cmd.omnimarket.delegate-skill.v1",
        key=b"\x00key",
        value=b"\xffvalue",
        headers=headers,
        timestamp_ms=1_800_000_000_123,
    )
    producer.send_and_wait.side_effect = asyncio.TimeoutError
    with pytest.raises(asyncio.TimeoutError):
        await publisher.publish(
            "onex.cmd.omnimarket.delegate-skill.v1",
            key=None,
            value=b"value",
            headers=[],
            timestamp_ms=1,
        )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("use_relay", [False, True])
async def test_host_composition_verifies_five_before_any_target_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_relay: bool,
) -> None:
    from omnibase_infra.runtime import sim_preflight_rehydration as composition

    authority, _verifier, records, _rows, _source, _calls = _fixture()
    plans = tuple(
        VerifiedSimArchiveRehydrationPlan(record, _mint=_RECEIPT_MINT)
        for record in records
    )
    steps: list[str] = []

    class FakePool:
        async def close(self) -> None:
            steps.append("pool-close")

    async def open_pool(*, dsn: str, server_settings: object = None) -> FakePool:
        if server_settings is None:
            assert "127.0.0.1:65036" in dsn
            steps.append("target-pool")
        else:
            assert server_settings == {"default_transaction_read_only": "on"}
            steps.append("readonly-source-pool")
        return FakePool()

    class FakeVerifier:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def verify(self, actual_authority: object, actual_records: object):
            assert actual_authority is authority
            assert actual_records is records
            steps.append("verified")
            return plans

    class FakeProducer:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["bootstrap_servers"] == "127.0.0.1:65092"

        async def start(self) -> None:
            steps.append("producer-start")

        async def stop(self) -> None:
            steps.append("producer-stop")

    class FakeOutbox:
        def __init__(self, _pool: object, _publisher: object, **kwargs: object) -> None:
            assert kwargs["rehydration_targets"] == frozenset(
                hop.topic for hop in _topology().declared_chain
            )

        async def enqueue_once(self, plan: object) -> bool:
            assert type(plan) is VerifiedSimArchiveRehydrationPlan
            steps.append("enqueue")
            return True

        async def require_only_selected_keys(self, actual_plans: object) -> None:
            assert actual_plans is plans
            steps.append("target-key-guard")

        async def relay_selected_once(self, actual_plans: object) -> int:
            assert actual_plans is plans
            steps.append("relay-selected")
            return 5

    target_receipt = tmp_path / "relay-receipt.json" if use_relay else None
    checked_receipts: list[Path | None] = []

    def inspect_target(receipt: Path | None = None) -> None:
        checked_receipts.append(receipt)
        steps.append("inspect")

    monkeypatch.setattr(composition, "verify_live_sim_preflight_target", inspect_target)
    monkeypatch.setattr(composition.asyncpg, "create_pool", open_pool)
    monkeypatch.setattr(composition, "SimArchiveSourceReceiptVerifier", FakeVerifier)
    monkeypatch.setattr(composition, "AIOKafkaProducer", FakeProducer)
    monkeypatch.setattr(composition, "PostgresSimArchiveRehydrationOutbox", FakeOutbox)

    delivered = await rehydrate_verified_sim_preflight_chain(
        authority=authority,
        archive_records=records,
        config=_config(target_relay_receipt=target_receipt),
    )

    assert delivered == 5
    assert steps[:5] == [
        "inspect",
        "readonly-source-pool",
        "readonly-source-pool",
        "verified",
        "inspect",
    ]
    assert steps.count("enqueue") == 5
    assert steps.index("target-pool") > steps.index("verified")
    assert steps.index("target-key-guard") < steps.index("enqueue")
    assert steps.index("relay-selected") > steps.index("enqueue")
    assert steps.count("inspect") == 3
    assert checked_receipts == [target_receipt] * 3


def test_relative_relay_receipt_is_refused() -> None:
    with pytest.raises(ValidationError, match="absolute private path"):
        _config(target_relay_receipt="relative.json")


def test_relay_target_requires_live_verifier(tmp_path: Path) -> None:
    receipt = tmp_path / "relay-receipt.json"
    with (
        patch(
            "omnibase_infra.runtime.sim_preflight_loopback_relay.verify_live_relay_receipt"
        ) as verify,
        patch("subprocess.run") as docker,
    ):
        verify_live_sim_preflight_target(receipt)
        verify.assert_called_once_with(
            receipt, required_services=("postgres", "redpanda")
        )
        docker.assert_not_called()
        verify.side_effect = ValueError("stale relay")
        with pytest.raises(ValueError, match="stale relay"):
            verify_live_sim_preflight_target(receipt)
        docker.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wrong_target_stops_before_source_or_target_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.runtime import sim_preflight_rehydration as composition

    authority, _verifier, records, _rows, _source, _calls = _fixture()
    monkeypatch.setattr(
        composition,
        "verify_live_sim_preflight_target",
        lambda receipt=None: (_ for _ in ()).throw(ValueError("wrong target")),
    )
    pool = AsyncMock()
    monkeypatch.setattr(composition.asyncpg, "create_pool", pool)

    with pytest.raises(ValueError, match="wrong target"):
        await rehydrate_verified_sim_preflight_chain(
            authority=authority, archive_records=records, config=_config()
        )
    pool.assert_not_awaited()
