#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify the approved five source records; relay only with explicit --execute."""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import tarfile
from contextlib import AsyncExitStack
from pathlib import Path

import asyncpg
from pydantic import BaseModel, ConfigDict

from omnibase_core.crypto import FileKeyProvider
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidenceReader,
    PostgresDelegationOwnerReader,
)
from omnibase_infra.runtime.db.sim_archive_source_ledger_reader import (
    PostgresSimArchiveSourceLedgerReader,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    resolve_declared_runtime_lane,
)
from omnibase_infra.runtime.models.model_execution_graph_trusted_gateway_config import (
    ModelExecutionGraphTrustedGatewayConfig,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    KafkaSimArchiveSourceReader,
    RawSimArchiveRecord,
    SimArchiveSourceReceiptVerifier,
)
from omnibase_infra.runtime.sim_preflight_rehydration import (
    ModelSimPreflightRehydrationConfig,
    rehydrate_verified_sim_preflight_chain,
)

_APPROVED_ARCHIVE_SHA256 = (
    "ea8144141c8484e100cecde4452ebe461170d965b92d5d7cc006f4748fa3eac2"
)
_APPROVED_CORRELATION_SHA256 = (
    "0a06b1c90e4521ac00d14663ed2103d710ace3ca879e3cc42404d210ad2da978"
)
_SELECTED_COORDINATES = (
    ("onex.cmd.omnimarket.delegate-skill.v1", 0, 2676),
    ("onex.cmd.omnibase-infra.delegation-request.v1", 0, 2286),
    ("onex.cmd.omnibase-infra.delegation-routing-request.v1", 0, 4400),
    ("onex.evt.omnibase-infra.routing-decision.v1", 0, 4353),
    ("onex.evt.omnimarket.delegate-skill-completed.v1", 0, 1939),
)
_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"


class ModelSelectedChainHarnessConfig(BaseModel):
    """Private host inputs; source and target remain explicitly distinct."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    archive_path: Path
    signed_command_path: Path
    owner_metadata_path: Path
    owner_row_path: Path
    gateway: ModelExecutionGraphTrustedGatewayConfig
    rehydration: ModelSimPreflightRehydrationConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_optional(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("archive byte field is not base64 text")
    return base64.b64decode(value, validate=True)


def load_selected_archive(path: Path) -> tuple[RawSimArchiveRecord, ...]:
    """Read no payload into output; preserve original ordered duplicate headers."""
    if _sha256(path) != _APPROVED_ARCHIVE_SHA256:
        raise ValueError("archive differs from the approved capture")
    selected: dict[tuple[str, int, int], RawSimArchiveRecord] = {}
    with tarfile.open(path, mode="r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonl.gz"):
                continue
            if member.name.split("/", 1)[0] not in {
                c[0] for c in _SELECTED_COORDINATES
            }:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive object is unreadable")
            with source, gzip.GzipFile(fileobj=source) as lines:
                for line in lines:
                    raw = json.loads(line)
                    coordinate = (raw["topic"], raw["partition"], raw["offset"])
                    if coordinate not in _SELECTED_COORDINATES:
                        continue
                    if coordinate in selected:
                        raise ValueError("archive repeats a selected coordinate")
                    value = _decode_optional(raw["value_b64"])
                    if value is None:
                        raise ValueError("selected archive value is absent")
                    selected[coordinate] = RawSimArchiveRecord(
                        topic=coordinate[0],
                        partition=coordinate[1],
                        offset=coordinate[2],
                        key=_decode_optional(raw["key_b64"]),
                        value=value,
                        headers=tuple(
                            (h[0], _decode_optional(h[1])) for h in raw["headers"]
                        ),
                        timestamp_ms=raw["timestamp_ms"],
                    )
    if set(selected) != set(_SELECTED_COORDINATES):
        raise ValueError("archive lacks the exact approved five records")
    return tuple(selected[c] for c in _SELECTED_COORDINATES)


def load_signed_authority(
    config: ModelSelectedChainHarnessConfig,
) -> VerifiedExecutionGraphReadAuthority:
    """Only the existing signature verifier can produce the capability."""
    if config.gateway.command_topic != _COMMAND_TOPIC:
        raise ValueError("gateway command topic differs from the graph contract")
    command = ModelMessageEnvelope[dict[str, object]].model_validate_json(
        config.signed_command_path.read_bytes()
    )
    scope = TrustedGatewaySignerScope(
        config.gateway.runtime_id, config.gateway.realm, config.gateway.bus_id
    )
    authority = verify_signed_execution_graph_read_authority(
        command,
        FileKeyProvider(config.gateway.public_key_path),
        TrustedExecutionGraphGatewayPolicy(frozenset({scope})),
    )
    inner = ModelEventEnvelope[object].model_validate(command.payload)
    tags = inner.metadata.tags
    if (
        inner.event_type != "omnibase-infra.delegation-execution-graph-requested"
        or tags.get("workflow_type") != "delegation-execution-graph-read"
        or tags.get("contract_id") != "node_execution_graph_read_effect:1.0.0"
    ):
        raise ValueError("signed command differs from the reviewed Gateway workflow")
    return authority


def corroborate_owner_capture(
    config: ModelSelectedChainHarnessConfig,
    authority: VerifiedExecutionGraphReadAuthority,
) -> None:
    """Local owner capture corroborates selection; it never creates authority."""
    entries = [
        line.split("=", 1)
        for line in config.owner_metadata_path.read_text().splitlines()
    ]
    metadata = dict(entries)
    if len(metadata) != len(entries) or set(metadata) != {
        "correlation_id",
        "tenant_id",
        "source_db",
        "source_table",
        "row_sha256",
        "schema_sha256",
    }:
        raise ValueError("owner capture metadata is not the exact receipt shape")
    if (
        hashlib.sha256(str(authority.correlation_id).encode()).hexdigest()
        != _APPROVED_CORRELATION_SHA256
        or metadata.get("correlation_id") != str(authority.correlation_id)
        or metadata.get("tenant_id") != str(authority.tenant_id)
        or metadata.get("source_db") != "omnidash_analytics"
        or metadata.get("source_table") != "public.delegation_events"
        or metadata.get("row_sha256") != _sha256(config.owner_row_path)
    ):
        raise ValueError("signed request differs from the approved owner capture")


async def verify_source_only(
    config: ModelSelectedChainHarnessConfig,
    authority: VerifiedExecutionGraphReadAuthority,
    records: tuple[RawSimArchiveRecord, ...],
) -> int:
    """Fresh source reads only: no target connection, outbox, or producer."""
    source = config.rehydration
    topology = PackagedExecutionGraphTopologyContract().resolve(source.topology_version)
    async with AsyncExitStack() as stack:
        analytics = await asyncpg.create_pool(
            dsn=source.source_databases.analytics_dsn.get_secret_value(),
            server_settings={"default_transaction_read_only": "on"},
        )
        stack.push_async_callback(analytics.close)
        ledger = await asyncpg.create_pool(
            dsn=source.source_databases.ledger_dsn.get_secret_value(),
            server_settings={"default_transaction_read_only": "on"},
        )
        stack.push_async_callback(ledger.close)
        verifier = SimArchiveSourceReceiptVerifier(
            current_reader=ExecutionGraphCurrentEvidenceReader(
                PostgresDelegationOwnerReader(analytics),
                PostgresSimArchiveSourceLedgerReader(ledger),
            ),
            source_reader=KafkaSimArchiveSourceReader(
                source.source_kafka,
                source_topic_namespace=source.source_topic_namespace,
            ),
            topology=topology,
        )
        plans = await verifier.verify(authority, records)
        if len(plans) != len(_SELECTED_COORDINATES):
            raise ValueError("source receipt is not the exact approved chain")
        return len(plans)


async def run(
    config: ModelSelectedChainHarnessConfig, *, execute: bool = False
) -> dict[str, str | int]:
    """Execution repeats fresh admission through the existing typed composition."""
    if resolve_declared_runtime_lane() != "sim-202":
        raise ValueError("selected replay harness requires sim-202 lane")
    authority = load_signed_authority(config)
    corroborate_owner_capture(config, authority)
    records = load_selected_archive(config.archive_path)
    count = (
        await rehydrate_verified_sim_preflight_chain(
            authority=authority, archive_records=records, config=config.rehydration
        )
        if execute
        else await verify_source_only(config, authority, records)
    )
    return {
        "mode": "execute" if execute else "verify-only",
        "records": count,
        "archive_sha256": _APPROVED_ARCHIVE_SHA256,
        "correlation_sha256": _APPROVED_CORRELATION_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enqueue and relay only the approved five records after fresh verification",
    )
    args = parser.parse_args()
    try:
        config = ModelSelectedChainHarnessConfig.model_validate_json(
            args.config.read_bytes()
        )
        result = asyncio.run(run(config, execute=args.execute))
    except Exception as exc:  # noqa: BLE001 -- private configuration must never enter logs
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
