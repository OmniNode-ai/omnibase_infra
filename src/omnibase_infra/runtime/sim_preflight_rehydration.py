# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-only, exact-chain replay into the disposable sim-preflight project."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
from aiokafka import AIOKafkaProducer
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidenceReader,
    PostgresDelegationOwnerReader,
)
from omnibase_infra.runtime.db.sim_archive_source_ledger_reader import (
    PostgresSimArchiveSourceLedgerReader,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    resolve_declared_runtime_lane,
)
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.sim_archive_rehydration_outbox import (
    PostgresSimArchiveRehydrationOutbox,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    KafkaSimArchiveSourceReader,
    RawSimArchiveRecord,
    SimArchiveSourceReceiptVerifier,
)

_PROJECT = "omnibase-infra-sim-preflight"
_TARGET_BROKER = "127.0.0.1:65092"
_TARGET_DB_HOST = "127.0.0.1"
_TARGET_DB_PORT = 65036
_TARGET_DB_NAME = "omnibase_infra"
_TARGETS = (
    ("omnibase-infra-sim-preflight-postgres", "postgres", "5432/tcp", "65036"),
    ("omnibase-infra-sim-preflight-redpanda", "redpanda", "19092/tcp", "65092"),
)


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.rstrip(".").lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _host_port(raw: str) -> tuple[str | None, int | None]:
    """Parse one broker coordinate without reflecting malformed input."""
    try:
        parsed = urlparse(f"//{raw.strip()}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError
        return parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError("invalid sim source broker coordinate") from exc


class ModelSimPreflightRehydrationConfig(BaseModel):
    """Separate read-only source and exact disposable target coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    source_databases: ModelExecutionGraphReadDatabases
    source_kafka: ModelKafkaEventBusConfig
    source_topic_namespace: str
    target_ledger_dsn: SecretStr
    topology_version: ModelExecutionGraphTopologyVersion
    target_relay_receipt: Path | None = None

    @model_validator(mode="after")
    def require_isolated_target(self) -> ModelSimPreflightRehydrationConfig:
        if (
            self.target_relay_receipt is not None
            and not self.target_relay_receipt.is_absolute()
        ):
            raise ValueError("relay receipt must be an absolute private path")
        try:
            target = urlparse(self.target_ledger_dsn.get_secret_value())
            target_port = target.port
        except ValueError as exc:
            raise ValueError("invalid sim target ledger coordinate") from exc
        if (
            target.scheme not in {"postgres", "postgresql"}
            or target.hostname != _TARGET_DB_HOST
            or target_port != _TARGET_DB_PORT
            or target.path != f"/{_TARGET_DB_NAME}"
            or target.params
            or target.query
            or target.fragment
        ):
            raise ValueError("sim target must be the exact local preflight ledger")
        for endpoint in self.source_kafka.bootstrap_servers.split(","):
            host, port = _host_port(endpoint)
            if _is_loopback_host(host) and port == 65092:
                raise ValueError("sim source broker must differ from disposable target")
        for source in (
            self.source_databases.analytics_dsn,
            self.source_databases.ledger_dsn,
        ):
            try:
                parsed = urlparse(source.get_secret_value())
                source_port = parsed.port
            except ValueError as exc:
                raise ValueError("invalid sim source database coordinate") from exc
            if parsed.params or parsed.query or parsed.fragment:
                raise ValueError("sim source database coordinate forbids overrides")
            if _is_loopback_host(parsed.hostname) and source_port == _TARGET_DB_PORT:
                raise ValueError(
                    "sim source database must differ from disposable target"
                )
        return self


def verify_live_sim_preflight_target(relay_receipt: Path | None = None) -> None:
    """Read live Docker state; config text alone cannot prove target ownership."""
    if relay_receipt is not None:
        from omnibase_infra.runtime.sim_preflight_loopback_relay import (
            verify_live_relay_receipt,
        )

        verify_live_relay_receipt(
            relay_receipt, required_services=("postgres", "redpanda")
        )
        return
    try:
        result = subprocess.run(
            ["docker", "inspect", *(target[0] for target in _TARGETS)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        raw = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as exc:
        raise ValueError("disposable sim target inspection failed") from exc
    if not isinstance(raw, list) or len(raw) != len(_TARGETS):
        raise ValueError("disposable sim target inspection is incomplete")
    by_name = {item.get("Name"): item for item in raw if isinstance(item, dict)}
    for container_name, service, container_port, host_port in _TARGETS:
        item = by_name.get(f"/{container_name}")
        if not isinstance(item, dict):
            raise ValueError("disposable sim target container identity differs")
        container_config = item.get("Config")
        state = item.get("State", {})
        network_settings = item.get("NetworkSettings")
        if not isinstance(container_config, dict) or not isinstance(
            network_settings, dict
        ):
            raise ValueError("disposable sim target inspection is malformed")
        labels = container_config.get("Labels", {})
        ports = network_settings.get("Ports", {})
        if (
            not isinstance(labels, dict)
            or labels.get("com.docker.compose.project") != _PROJECT
            or labels.get("com.docker.compose.service") != service
            or not isinstance(state, dict)
            or state.get("Running") is not True
            or state.get("Status") != "running"
            or not isinstance(ports, dict)
            or ports.get(container_port)
            != [{"HostIp": "127.0.0.1", "HostPort": host_port}]
        ):
            raise ValueError("disposable sim target is not the exact running project")


class KafkaSimPreflightRawPublisher:
    """Preserve original bytes, ordered headers, and timestamp until broker ack."""

    def __init__(self, producer: AIOKafkaProducer) -> None:
        self._producer = producer

    async def publish(
        self,
        topic: str,
        *,
        key: bytes | None,
        value: bytes,
        headers: list[tuple[str, bytes | None]],
        timestamp_ms: int,
    ) -> None:
        await asyncio.wait_for(
            self._producer.send_and_wait(
                topic,
                key=key,
                value=value,
                headers=headers,
                timestamp_ms=timestamp_ms,
            ),
            timeout=30,
        )


async def rehydrate_verified_sim_preflight_chain(
    *,
    authority: VerifiedExecutionGraphReadAuthority,
    archive_records: tuple[RawSimArchiveRecord, ...],
    config: ModelSimPreflightRehydrationConfig,
) -> int:
    """Reverify one complete current chain, enqueue, and relay only those rows."""
    if type(authority) is not VerifiedExecutionGraphReadAuthority:
        raise TypeError("sim rehydration requires signed gateway authority")
    if type(archive_records) is not tuple or any(
        type(record) is not RawSimArchiveRecord for record in archive_records
    ):
        raise TypeError("sim rehydration requires typed source records")
    if type(config) is not ModelSimPreflightRehydrationConfig:
        raise TypeError("sim rehydration requires typed source and target config")
    if resolve_declared_runtime_lane() != "sim-202":
        raise ValueError("sim rehydration requires sim-202 runtime lane")
    verify_live_sim_preflight_target(config.target_relay_receipt)
    topology = PackagedExecutionGraphTopologyContract().resolve(config.topology_version)
    if len(archive_records) != len(topology.declared_chain):
        raise ValueError("sim rehydration requires the exact declared chain")

    async with AsyncExitStack() as stack:
        source_analytics = await asyncpg.create_pool(
            dsn=config.source_databases.analytics_dsn.get_secret_value(),
            server_settings={"default_transaction_read_only": "on"},
        )
        stack.push_async_callback(source_analytics.close)
        source_ledger = await asyncpg.create_pool(
            dsn=config.source_databases.ledger_dsn.get_secret_value(),
            server_settings={"default_transaction_read_only": "on"},
        )
        stack.push_async_callback(source_ledger.close)
        verifier = SimArchiveSourceReceiptVerifier(
            current_reader=ExecutionGraphCurrentEvidenceReader(
                PostgresDelegationOwnerReader(source_analytics),
                PostgresSimArchiveSourceLedgerReader(source_ledger),
            ),
            source_reader=KafkaSimArchiveSourceReader(
                config.source_kafka,
                source_topic_namespace=config.source_topic_namespace,
            ),
            topology=topology,
        )
        plans = await verifier.verify(authority, archive_records)
        if len(plans) != len(topology.declared_chain):
            raise ValueError("sim rehydration receipt is not the complete chain")

        # Recheck after potentially slow source reads, immediately before any
        # target connection/write. A retry repeats the entire source receipt.
        verify_live_sim_preflight_target(config.target_relay_receipt)
        target_pool = await asyncpg.create_pool(
            dsn=config.target_ledger_dsn.get_secret_value()
        )
        stack.push_async_callback(target_pool.close)
        producer = AIOKafkaProducer(
            bootstrap_servers=_TARGET_BROKER,
            acks="all",
            enable_idempotence=True,
        )
        stack.push_async_callback(producer.stop)
        await producer.start()
        outbox = PostgresSimArchiveRehydrationOutbox(
            target_pool,
            KafkaSimPreflightRawPublisher(producer),
            rehydration_targets=frozenset(hop.topic for hop in topology.declared_chain),
        )
        await outbox.require_only_selected_keys(plans)
        for plan in plans:
            await outbox.enqueue_once(plan)
        verify_live_sim_preflight_target(config.target_relay_receipt)
        return await outbox.relay_selected_once(plans)


__all__ = [
    "KafkaSimPreflightRawPublisher",
    "ModelSimPreflightRehydrationConfig",
    "rehydrate_verified_sim_preflight_chain",
    "verify_live_sim_preflight_target",
]
