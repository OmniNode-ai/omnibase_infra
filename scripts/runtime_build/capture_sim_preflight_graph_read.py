# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""After operator auth proof, capture one actual authorized Gateway wire value.

This local validation tool never logs in, provisions services, creates signing
keys, or rehydrates data. Its HTTP call is a real workflow submission; run only
after the coordinator has passed the disposable realm's authorization proof.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import stat
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, cast
from uuid import UUID

import httpx
from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.crypto import FileKeyProvider
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.models.model_execution_graph_trusted_gateway_config import (
    ModelExecutionGraphTrustedGatewayConfig,
)
from omnibase_infra.runtime.sim_archive_source_receipt import RawSimArchiveRecord
from omnibase_infra.topics.topic_namespace import (
    TOPIC_NAMESPACE_ENV_VAR,
    apply_topic_namespace,
    resolve_topic_namespace,
)

_TOPIC = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
_WORKFLOW = "delegation-execution-graph-read"
_URL = "http://localhost:8090/v1/workflows"  # url-authority-ok: fixed approved disposable relay; arbitrary targets are forbidden.
_MAX_RECORDS = 64


class GatewayCaptureHTTPError(ValueError):
    def __init__(self, status_code: int) -> None:
        super().__init__("Gateway did not publish the authorized graph read")
        self.status_code = status_code


def _log_failure(phase: str, error: BaseException) -> None:
    status = error.status_code if isinstance(error, GatewayCaptureHTTPError) else None
    kind = (
        "http_refusal"
        if status is not None
        else "timeout"
        if isinstance(error, (TimeoutError, httpx.TimeoutException))
        else "cancelled"
        if isinstance(error, asyncio.CancelledError)
        else "io_failure"
        if isinstance(error, OSError)
        else "refusal"
        if isinstance(error, ValueError)
        else "operation_failure"
    )
    logging.getLogger(__name__).log(
        logging.ERROR,
        "capture_failed phase=%s error_kind=%s http_status=%s",
        phase,
        kind,
        status,
    )


class ModelGraphReadCaptureConfig(BaseModel):
    """Private local inputs; no ambient broker, HTTP, or signer authority."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    token_path: Path
    owner_metadata_path: Path
    signed_command_output: Path
    gateway: ModelExecutionGraphTrustedGatewayConfig
    kafka: ModelKafkaEventBusConfig
    topic_namespace: str
    timeout_seconds: float = Field(default=20, gt=0, le=30)

    @model_validator(mode="after")
    def isolated_targets(self) -> ModelGraphReadCaptureConfig:
        if self.kafka.bootstrap_servers != "127.0.0.1:65092":
            raise ValueError("capture requires the exact local sim broker")
        if (
            self.kafka.security_protocol != "PLAINTEXT"
            or self.kafka.sasl_mechanism is not None
        ):
            raise ValueError("capture requires the existing plain local sim broker")
        if self.gateway.command_topic != _TOPIC:
            raise ValueError("capture requires the declared graph command")
        resolve_topic_namespace({TOPIC_NAMESPACE_ENV_VAR: self.topic_namespace})
        return self


class ModelGraphReadAck(BaseModel):
    """The existing Gateway acknowledgement shape, not a new ingress API."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    workflow_id: UUID
    envelope_id: UUID
    correlation_id: UUID
    workflow_type: Literal["delegation-execution-graph-read"]
    status: Literal["published"]
    accepted_at: datetime


class ProtocolCaptureConsumer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def partitions_for_topic(self, topic: str) -> set[int] | None: ...
    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]: ...
    def assign(self, partitions: list[TopicPartition]) -> None: ...
    def seek(self, partition: TopicPartition, offset: int) -> None: ...
    async def getone(self) -> RawSimArchiveRecord: ...


class _KafkaCaptureConsumer:
    def __init__(self, config: ModelKafkaEventBusConfig, topic: str = _TOPIC) -> None:
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=config.bootstrap_servers,
            group_id=None,
            enable_auto_commit=False,
            **build_aiokafka_auth_kwargs(config),
        )

    async def start(self) -> None:
        await self.consumer.start()

    async def stop(self) -> None:
        await self.consumer.stop()

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        result = self.consumer.partitions_for_topic(topic)
        if result is None:
            return None
        if not isinstance(result, set) or any(
            type(value) is not int for value in result
        ):
            raise ValueError("broker partition metadata is malformed")
        return cast("set[int]", result)

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        result = await self.consumer.end_offsets(partitions)
        if not isinstance(result, dict) or any(
            not isinstance(key, TopicPartition) or type(value) is not int
            for key, value in result.items()
        ):
            raise ValueError("broker end metadata is malformed")
        return cast("dict[TopicPartition, int]", result)

    def assign(self, partitions: list[TopicPartition]) -> None:
        # Constructor subscription primes topic metadata during start; switch
        # to explicit manual assignment only after all end offsets are saved.
        self.consumer.unsubscribe()
        self.consumer.assign(partitions)

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.consumer.seek(partition, offset)

    async def getone(self) -> RawSimArchiveRecord:
        record = await self.consumer.getone()
        if not isinstance(record.value, bytes):
            raise ValueError("captured wire value must be bytes")
        return RawSimArchiveRecord(
            record.topic,
            record.partition,
            record.offset,
            record.key,
            record.value,
            tuple(record.headers),
            record.timestamp,
        )


def _private_file(path: Path) -> None:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("capture input must be a private regular file")


def _owner(path: Path) -> tuple[UUID, UUID]:
    _private_file(path)
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("owner metadata is malformed")
        fields[key] = value
    if set(fields) != {
        "correlation_id",
        "tenant_id",
        "source_db",
        "source_table",
        "row_sha256",
        "schema_sha256",
    }:
        raise ValueError("owner metadata is not the canonical capture shape")
    if (
        fields["source_db"] != "omnidash_analytics"
        or fields["source_table"] != "public.delegation_events"
    ):
        raise ValueError("owner metadata has the wrong relation scope")
    if not all(
        re.fullmatch(r"[0-9a-f]{64}", fields[key])
        for key in ("row_sha256", "schema_sha256")
    ):
        raise ValueError("owner metadata has invalid checksums")
    correlation = UUID(fields["correlation_id"])
    tenant = UUID(fields["tenant_id"])
    if (
        str(correlation) != fields["correlation_id"]
        or str(tenant) != fields["tenant_id"]
    ):
        raise ValueError("owner metadata identities must be canonical UUIDs")
    return correlation, tenant


async def _submit(
    token: str, body: dict[str, object], timeout: float
) -> ModelGraphReadAck:
    async with httpx.AsyncClient(
        trust_env=False, follow_redirects=False, timeout=timeout
    ) as client:
        response = await client.post(
            _URL, headers={"Authorization": "Bearer " + token}, json=body
        )
        if response.status_code != 202:
            raise GatewayCaptureHTTPError(response.status_code)
        return ModelGraphReadAck.model_validate(response.json())


async def capture(
    config: ModelGraphReadCaptureConfig,
    *,
    consumer_factory: Callable[[], ProtocolCaptureConsumer] | None = None,
    submit: Callable[
        [str, dict[str, object], float], Awaitable[ModelGraphReadAck]
    ] = _submit,
) -> dict[str, object]:
    """Submit once and save only matching, verified wire bytes; never commit offsets."""
    _private_file(config.token_path)
    correlation, tenant = _owner(config.owner_metadata_path)
    token = config.token_path.read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token):
        raise ValueError("private access token has an invalid wire shape")
    output = config.signed_command_output
    parent = output.parent
    if (
        not output.is_absolute()
        or parent.is_symlink()
        or not parent.is_dir()
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("capture output requires a new file in a private directory")
    namespace = resolve_topic_namespace(
        {TOPIC_NAMESPACE_ENV_VAR: config.topic_namespace}
    )
    physical_topic = apply_topic_namespace(_TOPIC, namespace=namespace)
    scope = TrustedGatewaySignerScope(
        config.gateway.runtime_id, config.gateway.realm, config.gateway.bus_id
    )
    policy = TrustedExecutionGraphGatewayPolicy(frozenset({scope}))
    keys = FileKeyProvider(config.gateway.public_key_path)
    consumer = (
        consumer_factory()
        if consumer_factory is not None
        else _KafkaCaptureConsumer(config.kafka, physical_topic)
    )
    captured: RawSimArchiveRecord | None = None
    ack: ModelGraphReadAck | None = None
    failure: BaseException | None = None
    phase = "metadata"
    try:
        async with asyncio.timeout(config.timeout_seconds):
            await consumer.start()
            partitions = consumer.partitions_for_topic(physical_topic)
            if not partitions or any(partition < 0 for partition in partitions):
                raise ValueError("graph command topic must exist before submission")
            assigned = [
                TopicPartition(physical_topic, partition)
                for partition in sorted(partitions)
            ]
            phase = "snapshot"
            ends = await consumer.end_offsets(assigned)
            if set(ends) != set(assigned) or any(
                offset < 0 for offset in ends.values()
            ):
                raise ValueError("graph command end snapshot is incomplete")
            consumer.assign(assigned)
            for partition in assigned:
                consumer.seek(partition, ends[partition])
            phase = "submit"
            ack = await submit(
                token,
                {
                    "workflow_type": _WORKFLOW,
                    "correlation_id": str(correlation),
                    "payload": {"cursor_mode": "latest"},
                },
                config.timeout_seconds,
            )
            phase = "ack_binding"
            if ack.correlation_id != correlation:
                raise ValueError(
                    "Gateway acknowledgement changed the owner correlation"
                )
            for _ in range(_MAX_RECORDS):
                phase = "wire_wait"
                record = await consumer.getone()
                phase = "wire_coordinate"
                coordinate = TopicPartition(record.topic, record.partition)
                if coordinate not in ends or record.offset < ends[coordinate]:
                    raise ValueError(
                        "capture escaped the snapshotted command coordinates"
                    )
                phase = "wire_schema"
                envelope = ModelMessageEnvelope[dict[str, object]].model_validate_json(
                    record.value
                )
                phase = "wire_signature"
                authority = verify_signed_execution_graph_read_authority(
                    envelope, keys, policy
                )
                if authority.workflow_id != ack.workflow_id:
                    continue
                phase = "wire_binding"
                inner = ModelEventEnvelope[object].model_validate(envelope.payload)
                if (
                    authority.correlation_id != correlation
                    or authority.tenant_id != tenant
                    or inner.envelope_id != ack.envelope_id
                    or inner.event_type
                    != "omnibase-infra.delegation-execution-graph-requested"
                    or inner.metadata.tags.get("workflow_type") != _WORKFLOW
                    or inner.metadata.tags.get("contract_id")
                    != "node_execution_graph_read_effect:1.0.0"
                    or authority.request.cursor_mode.value != "latest"
                ):
                    raise ValueError(
                        "verified wire differs from the real acknowledgement or owner"
                    )
                captured = record
                break
            if captured is None:
                raise ValueError(
                    "matching signed command was not captured within the record bound"
                )
    except BaseException as exc:
        failure = exc
        _log_failure(phase, exc)
        raise
    finally:
        try:
            await asyncio.wait_for(
                consumer.stop(), timeout=min(config.timeout_seconds, 5)
            )
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- preserve original refusal without exposing teardown details
            _log_failure("cleanup", exc)
            if failure is None:
                raise ValueError("capture consumer cleanup failed") from None
    if captured is None or ack is None:
        raise ValueError("authorized command capture is incomplete")
    descriptor = os.open(
        output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as target:
        target.write(captured.value)
    return {
        "status": "captured_authorized_wire",
        "bytes": len(captured.value),
        "wire_sha256": hashlib.sha256(captured.value).hexdigest(),
        "correlation_sha256": hashlib.sha256(str(correlation).encode()).hexdigest(),
        "workflow_sha256": hashlib.sha256(str(ack.workflow_id).encode()).hexdigest(),
        "partition": captured.partition,
        "offset": captured.offset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        _private_file(args.config)
        config = ModelGraphReadCaptureConfig.model_validate_json(
            args.config.read_bytes()
        )
        result = asyncio.run(capture(config))
    except (Exception, asyncio.CancelledError):  # noqa: BLE001 -- never log credential-bearing HTTP/config failures
        print(json.dumps({"status": "refused"}))
        return 65
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
