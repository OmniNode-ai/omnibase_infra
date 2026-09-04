# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Standalone process entrypoint for the hybrid gateway bus forwarder."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import random
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import yaml
from aiokafka.errors import KafkaError

from omnibase_core.protocols.runtime.protocol_transport_producer import (
    ProtocolTransportProducer,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.idempotency import StoreIdempotencySqlite
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayHttpsIngestConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    NodeGatewayDelivery,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ProtocolGatewayPublisher,
    ServiceGatewayForwarder,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)
from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)

logger = logging.getLogger(__name__)

_GATEWAY_CONTRACT_NAME = "node_bus_forwarder_effect"
_DEFAULT_GATEWAY_CONTRACT_PATH = (
    Path(__file__).parents[1] / "nodes" / _GATEWAY_CONTRACT_NAME / "contract.yaml"
)


def load_gateway_forwarder_runtime_config(
    config_path: Path,
    *,
    contract_path: Path = _DEFAULT_GATEWAY_CONTRACT_PATH,
    broker_ref_map_path: Path,
) -> ModelGatewayForwarderRuntimeConfig:
    """Load and validate one explicit two-leg forwarder configuration.

    ``broker_ref_map_path`` is required (no default): the cloud broker
    endpoint is resolved from the node contract's ``cloud_broker_ref`` at
    this effect boundary, never hardcoded into compose or tenant config. See
    ``_materialize_cloud_broker_ref`` for the fail-closed resolution rules.
    """
    raw_object: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_object, dict):
        raise ValueError("gateway forwarder config must be a YAML mapping")
    raw: dict[str, object] = {str(key): value for key, value in raw_object.items()}
    _materialize_contract_mirror_topics(raw, contract_path)
    _materialize_contract_canary_config(raw, contract_path)
    _materialize_contract_lane_mirror(raw, contract_path)
    _materialize_cloud_broker_ref(raw, contract_path, broker_ref_map_path)
    _materialize_contract_https_ingest(raw, contract_path, broker_ref_map_path)
    return ModelGatewayForwarderRuntimeConfig.model_validate(raw)


def _load_gateway_forwarder_config_block(
    contract_path: Path, selector: object
) -> dict[str, object]:
    """Read and validate ``config.gateway_forwarder`` from the node contract."""
    contract_object: object = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract_object, dict):
        raise ValueError("gateway node contract must be a YAML mapping")
    contract: dict[str, object] = {
        str(key): value for key, value in contract_object.items()
    }
    if contract.get("contract_name") != selector:
        raise ValueError("gateway node contract does not match mirror_topic_set")
    contract_config = contract.get("config")
    if not isinstance(contract_config, dict):
        raise ValueError("gateway node contract is missing config")
    gateway_config = contract_config.get("gateway_forwarder")
    if not isinstance(gateway_config, dict):
        raise ValueError("gateway node contract is missing gateway_forwarder")
    return {str(key): value for key, value in gateway_config.items()}


def _materialize_contract_mirror_topics(
    raw: dict[str, object],
    contract_path: Path,
) -> None:
    """Resolve the named fixed topic set from the node contract.

    Resolved deployment YAML intentionally cannot repeat raw topic literals.
    The node contract is their sole authority; the tenant config names that
    contract and this boundary copies its validated inbound/outbound set into
    the frozen runtime model before either broker starts.
    """
    forwarder_object = raw.get("forwarder")
    if not isinstance(forwarder_object, dict):
        raise ValueError("gateway forwarder config requires a forwarder mapping")
    forwarder: dict[str, object] = {
        str(key): value for key, value in forwarder_object.items()
    }
    raw["forwarder"] = forwarder
    if "mirror_topics" in forwarder:
        raise ValueError(
            "resolved gateway config must name mirror_topic_set instead of "
            "redeclaring topic literals"
        )
    selector = forwarder.pop("mirror_topic_set", None)
    if selector != _GATEWAY_CONTRACT_NAME:
        raise ValueError(
            f"mirror_topic_set must be {_GATEWAY_CONTRACT_NAME!r}, got {selector!r}"
        )

    gateway_config = _load_gateway_forwarder_config_block(contract_path, selector)
    mirror_topics_object = gateway_config.get("mirror_topics")
    if not isinstance(mirror_topics_object, dict):
        raise ValueError("gateway node contract mirror_topics must be a mapping")
    forwarder["mirror_topics"] = {
        str(key): value for key, value in mirror_topics_object.items()
    }


def _materialize_contract_canary_config(
    raw: dict[str, object],
    contract_path: Path,
) -> None:
    """Resolve the canary probe topic/cadence/deadlines from the node contract.

    Same authority pattern as ``_materialize_contract_mirror_topics``: resolved
    deployment YAML names the contract via ``canary_topic_set`` and may not
    redeclare the canary block inline, so the contract stays the sole source
    of the canary topic and its cadence/deadlines (OMN-15741).
    """
    forwarder_object = raw["forwarder"]
    if not isinstance(forwarder_object, dict):
        raise ValueError("gateway forwarder config requires a forwarder mapping")
    forwarder: dict[str, object] = forwarder_object
    if "canary" in forwarder:
        raise ValueError(
            "resolved gateway config must name canary_topic_set instead of "
            "redeclaring the canary block"
        )
    selector = forwarder.pop("canary_topic_set", None)
    if selector != _GATEWAY_CONTRACT_NAME:
        raise ValueError(
            f"canary_topic_set must be {_GATEWAY_CONTRACT_NAME!r}, got {selector!r}"
        )

    gateway_config = _load_gateway_forwarder_config_block(contract_path, selector)
    canary_object = gateway_config.get("canary")
    if not isinstance(canary_object, dict):
        raise ValueError(
            "gateway node contract is missing config.gateway_forwarder.canary"
        )
    forwarder["canary"] = {str(key): value for key, value in canary_object.items()}


def _materialize_contract_lane_mirror(
    raw: dict[str, object],
    contract_path: Path,
) -> None:
    """Resolve the lane-mirror lane names and topic set from the node contract.

    Same authority pattern as ``_materialize_contract_mirror_topics`` and
    ``_materialize_contract_canary_config`` (OMN-17034): the resolved
    deployment YAML names the contract via ``lane_mirror_set`` and may not
    redeclare the block inline, so the source lane, the mirror-lane set and
    the mirrored topics have exactly one home. The per-lane BROKER ADDRESSES
    are not resolved here -- those are this deployment's answer and stay in
    the resolved YAML's ``lane_mirror_source_bus``/``lane_mirror_buses``,
    exactly as ``local_bus`` already works.

    A contract with no ``lane_mirror`` block and a resolved file that does not
    name ``lane_mirror_set`` is a valid two-leg deployment and passes through
    untouched.
    """
    forwarder_object = raw.get("forwarder")
    if not isinstance(forwarder_object, dict):
        raise ValueError("gateway forwarder config requires a forwarder mapping")
    forwarder: dict[str, object] = forwarder_object
    # Key-presence alone is NOT a redeclaration: ``lane_mirror`` is an optional
    # field on ModelGatewayForwarderConfig, so any round-trip through
    # ``model_dump()`` emits an explicit ``lane_mirror: null`` for a two-leg
    # deployment. Rejecting that would refuse a legitimate operator artifact
    # (and did -- it broke three existing loader tests). Only a populated block
    # is an attempt to redeclare what the contract owns.
    if forwarder.get("lane_mirror") is not None:
        raise ValueError(
            "resolved gateway config must name lane_mirror_set instead of "
            "redeclaring the lane_mirror block"
        )
    selector = forwarder.pop("lane_mirror_set", None)
    if selector is None:
        return
    if selector != _GATEWAY_CONTRACT_NAME:
        raise ValueError(
            f"lane_mirror_set must be {_GATEWAY_CONTRACT_NAME!r}, got {selector!r}"
        )

    gateway_config = _load_gateway_forwarder_config_block(contract_path, selector)
    lane_mirror_object = gateway_config.get("lane_mirror")
    if not isinstance(lane_mirror_object, dict):
        raise ValueError(
            "resolved gateway config names lane_mirror_set but the node contract "
            "has no config.gateway_forwarder.lane_mirror block"
        )
    forwarder["lane_mirror"] = {
        str(key): value for key, value in lane_mirror_object.items()
    }


def _materialize_cloud_broker_ref(
    raw: dict[str, object],
    contract_path: Path,
    broker_ref_map_path: Path,
) -> None:
    """Resolve ``cloud_bus.bootstrap_servers`` from the contract's cloud broker ref.

    Mirrors ``_materialize_contract_mirror_topics``: the node contract's
    ``gateway_forwarder.cloud_leg.cloud_broker_ref`` is the sole authority for
    which cloud broker endpoint applies. Resolved tenant YAML may declare
    ``forwarder.cloud_bus.cloud_broker_ref`` (it must match the contract
    verbatim) but must never carry a ``bootstrap_servers`` literal -- the
    actual address is resolved here, at the effect boundary, from an
    operator-supplied broker-ref map. This replaces the previous hardcoded
    Docker ``extra_hosts``/``bootstrap_servers`` literal (OMN-15743).

    Fails closed: raises ``ValueError`` if the resolved config redeclares the
    literal, if the declared ref does not match the contract, or if the map
    is missing/unreadable/has no entry for the ref. There is no hardcoded
    fallback endpoint.
    """
    cloud_bus_object = raw.get("cloud_bus")
    if not isinstance(cloud_bus_object, dict):
        raise ValueError("gateway forwarder config requires a cloud_bus mapping")
    cloud_bus: dict[str, object] = {
        str(key): value for key, value in cloud_bus_object.items()
    }
    raw["cloud_bus"] = cloud_bus
    if "bootstrap_servers" in cloud_bus:
        raise ValueError(
            "resolved gateway config must not declare cloud_bus.bootstrap_servers "
            "as a literal; the cloud broker endpoint is resolved from the node "
            "contract's cloud_broker_ref at the effect boundary"
        )

    forwarder_object = raw.get("forwarder")
    if not isinstance(forwarder_object, dict):
        raise ValueError("gateway forwarder config requires a forwarder mapping")
    declared_cloud_bus_object = forwarder_object.get("cloud_bus")
    if not isinstance(declared_cloud_bus_object, dict):
        raise ValueError("gateway forwarder config requires forwarder.cloud_bus")
    declared_ref = declared_cloud_bus_object.get("cloud_broker_ref")
    if not isinstance(declared_ref, str) or not declared_ref:
        raise ValueError(
            "gateway forwarder config forwarder.cloud_bus.cloud_broker_ref must "
            "be a non-empty string"
        )

    contract_object: object = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract_object, dict):
        raise ValueError("gateway node contract must be a YAML mapping")
    contract: dict[str, object] = {
        str(key): value for key, value in contract_object.items()
    }
    contract_config = contract.get("config")
    if not isinstance(contract_config, dict):
        raise ValueError("gateway node contract is missing config")
    gateway_config = contract_config.get("gateway_forwarder")
    if not isinstance(gateway_config, dict):
        raise ValueError("gateway node contract is missing gateway_forwarder")
    cloud_leg = gateway_config.get("cloud_leg")
    if not isinstance(cloud_leg, dict):
        raise ValueError("gateway node contract is missing gateway_forwarder.cloud_leg")
    contract_ref = cloud_leg.get("cloud_broker_ref")
    if not isinstance(contract_ref, str) or not contract_ref:
        raise ValueError(
            "gateway node contract cloud_leg.cloud_broker_ref must be a "
            "non-empty string"
        )
    if declared_ref != contract_ref:
        raise ValueError(
            f"resolved forwarder.cloud_bus.cloud_broker_ref {declared_ref!r} "
            "does not match the node contract's cloud_leg.cloud_broker_ref "
            f"{contract_ref!r}"
        )

    if not broker_ref_map_path.is_file():
        raise ValueError(
            f"no broker-ref map was found at {broker_ref_map_path!s}; the "
            "gateway process refuses to start without a resolvable "
            "cloud_broker_ref (fail-closed -- there is no hardcoded fallback "
            "broker endpoint)"
        )
    map_object: object = yaml.safe_load(broker_ref_map_path.read_text(encoding="utf-8"))
    if not isinstance(map_object, dict):
        raise ValueError(
            f"broker-ref map at {broker_ref_map_path!s} must be a YAML mapping"
        )
    resolved = map_object.get(contract_ref)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(
            f"broker-ref map at {broker_ref_map_path!s} has no resolvable "
            f"entry for cloud_broker_ref={contract_ref!r}"
        )
    cloud_bus["bootstrap_servers"] = resolved.strip()


def _materialize_contract_https_ingest(
    raw: dict[str, object],
    contract_path: Path,
    broker_ref_map_path: Path,
) -> None:
    """Resolve the OUTBOUND HTTPS ingest leg from the node contract (OMN-16459).

    Same authority split ``_materialize_contract_mirror_topics`` uses: the node
    contract owns the refs, the batch bound and the idempotency key; the
    resolved deployment YAML only NAMES the contract via ``https_ingest_set``.
    The one value this boundary resolves -- the ingest route address -- comes
    from the SAME operator-supplied ref map that already resolves
    ``cloud_broker_ref``, so the leg adds no new mount and no new env var.

    Opt-in: a resolved deployment with no ``https_ingest_set`` keeps the
    direct-MSK Kafka outbound leg unchanged.
    """
    forwarder_object = raw.get("forwarder")
    if not isinstance(forwarder_object, dict):
        raise ValueError("gateway forwarder config requires a forwarder mapping")
    forwarder: dict[str, object] = {
        str(key): value for key, value in forwarder_object.items()
    }
    raw["forwarder"] = forwarder

    # ``https_ingest`` is optional on the frozen model, so any config round-tripped
    # through ``model_dump()`` emits an explicit ``https_ingest: null``. Refusing
    # on key PRESENCE would break every such round trip; only a POPULATED inline
    # block is a redeclaration of contract-owned values.
    inline = forwarder.get("https_ingest")
    if inline:
        raise ValueError(
            "resolved gateway config must name https_ingest_set instead of "
            "redeclaring the contract's ingest refs, batch bound or idempotency key"
        )
    forwarder.pop("https_ingest", None)

    selector = forwarder.pop("https_ingest_set", None)
    if selector is None:
        return
    if selector != _GATEWAY_CONTRACT_NAME:
        raise ValueError(
            f"https_ingest_set must be {_GATEWAY_CONTRACT_NAME!r}, got {selector!r}"
        )

    gateway_config = _load_gateway_forwarder_config_block(contract_path, selector)
    declared = gateway_config.get("https_ingest")
    if not isinstance(declared, dict):
        raise ValueError(
            "gateway node contract is missing config.gateway_forwarder.https_ingest "
            "but the resolved deployment opted in via https_ingest_set"
        )
    ingest: dict[str, object] = {str(key): value for key, value in declared.items()}
    url_ref = ingest.get("ingest_url_ref")
    if not isinstance(url_ref, str) or not url_ref.strip():
        raise ValueError(
            "gateway node contract https_ingest.ingest_url_ref must be a non-empty "
            "string"
        )
    ingest["ingest_url"] = _resolve_ref_from_map(
        broker_ref_map_path, url_ref.strip(), "https ingest url"
    )
    forwarder["https_ingest"] = ingest


def _resolve_ref_from_map(
    broker_ref_map_path: Path, reference: str, description: str
) -> str:
    """Resolve one reference from the operator-supplied ref map, fail-closed."""
    if not broker_ref_map_path.is_file():
        raise ValueError(
            f"no ref map was found at {broker_ref_map_path!s}; the gateway process "
            f"refuses to start without a resolvable {description} (fail-closed -- "
            "there is no hardcoded fallback endpoint)"
        )
    map_object: object = yaml.safe_load(broker_ref_map_path.read_text(encoding="utf-8"))
    if not isinstance(map_object, dict):
        raise ValueError(f"ref map at {broker_ref_map_path!s} must be a YAML mapping")
    resolved = map_object.get(reference)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(
            f"ref map at {broker_ref_map_path!s} has no resolvable entry for "
            f"{reference} ({description})"
        )
    return resolved.strip()


class TransportGatewayHttpsIngest:
    """Adapt the gateway's single HTTPS ingest route to the publish boundary.

    Sibling of ``TransportGatewayBus``: it satisfies the node's existing
    ``ProtocolGatewayPublisher`` structurally and introduces NO new protocol
    (the node already owns ``ProtocolGatewayPublisher`` and
    ``ProtocolGatewayConsumer``; OMN-12912's protocol-ownership gate refuses a
    duplicate surface). It is a transport adapter, not a bespoke client class.

    Delivery semantics deliberately match what the direct-MSK leg got for free
    from the broker: a transient failure raises ``InfraUnavailableError``, the
    forwarder's retryable class, so ``_publish_with_delivery_retry`` retains the
    source message and the delivery node never commits the source offset. A 4xx
    is the route REFUSING this record, which retrying cannot fix, so it is
    deliberately raised as a non-retryable class instead of spinning forever.
    """

    def __init__(
        self,
        *,
        config: ModelGatewayHttpsIngestConfig,
        tenant_slug: str,
        client: httpx.AsyncClient,
        auth_token: str,
    ) -> None:
        self._config = config
        self._tenant_slug = tenant_slug
        self._client = client
        self._auth_token = auth_token

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        idempotency_key = self._idempotency_key(value)
        record: dict[str, object] = {
            "topic": topic,
            "key": base64.b64encode(key).decode("ascii") if key is not None else None,
            "value": base64.b64encode(value).decode("ascii"),
            "envelope_id": idempotency_key,
        }
        encoded_headers = self._encode_headers(headers)
        if encoded_headers:
            record["headers"] = encoded_headers
        try:
            response = await self._client.post(
                self._config.ingest_url,
                json={"tenant_slug": self._tenant_slug, "records": [record]},
                headers={
                    "authorization": f"Bearer {self._auth_token}",
                    "idempotency-key": idempotency_key,
                    "content-type": "application/json",
                },
                timeout=self._config.request_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise InfraUnavailableError(
                "gateway https ingest route unreachable for topic "
                f"{topic}: {type(exc).__name__}"
            ) from exc
        if response.status_code >= 500:
            raise InfraUnavailableError(
                f"gateway https ingest route returned {response.status_code} for "
                f"topic {topic}; retaining the source message"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"gateway https ingest route rejected the record for topic {topic} "
                f"with status {response.status_code}; retrying a rejection cannot "
                "succeed, so this is not raised as the retryable class"
            )

    def _idempotency_key(self, value: bytes) -> str:
        """Read the content-addressed envelope id the route deduplicates on."""
        try:
            decoded: object = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "gateway https ingest requires a JSON envelope carrying "
                "envelope_id; the record could not be decoded"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError(
                "gateway https ingest requires a JSON object carrying envelope_id"
            )
        envelope_id = decoded.get("envelope_id")
        if not isinstance(envelope_id, str) or not envelope_id.strip():
            raise ValueError(
                "gateway https ingest record carries no envelope_id; without the "
                "content-addressed id the ingest route cannot deduplicate a "
                "redelivery, so the record is refused rather than sent unkeyed"
            )
        return envelope_id.strip()

    @staticmethod
    def _encode_headers(headers: object | None) -> dict[str, str]:
        if headers is None:
            return {}
        if isinstance(headers, ModelEventHeaders):
            return {
                header_key: str(header_value)
                for header_key, header_value in headers.model_dump(
                    mode="json", exclude_none=True
                ).items()
            }
        if isinstance(headers, Mapping):
            encoded: dict[str, str] = {}
            for header_key, header_value in headers.items():
                if not isinstance(header_key, str):
                    raise TypeError(
                        "gateway transport headers must map string keys to bytes"
                    )
                if isinstance(header_value, bytes):
                    encoded[header_key] = header_value.decode("utf-8", "replace")
                else:
                    raise TypeError(
                        "gateway transport headers must map string keys to bytes"
                    )
            return encoded
        raise TypeError("gateway transport headers must map string keys to bytes")


async def select_outbound_publish_transport(
    config: ModelGatewayForwarderRuntimeConfig,
    *,
    kafka_bus: ProtocolGatewayPublisher,
    resolve_secret: Callable[[str], Awaitable[str | None]],
    client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
) -> tuple[ProtocolGatewayPublisher, httpx.AsyncClient | None]:
    """Pick the outbound publish boundary from the RESOLVED CONTRACT.

    There is deliberately no flag and no environment variable for this: the
    transport is whatever ``config.forwarder.https_ingest`` says it is, which is
    whatever the node contract plus the resolved deployment's
    ``https_ingest_set`` opt-in produced. Returns the publisher plus the HTTP
    client that must be closed on shutdown (``None`` on the Kafka path).
    """
    ingest = config.forwarder.https_ingest
    if ingest is None:
        return kafka_bus, None
    token = await resolve_secret(ingest.ingest_auth_ref)
    if not token:
        raise ValueError(
            "gateway https ingest leg is declared but its credential reference "
            f"{ingest.ingest_auth_ref} did not resolve in the secret store; the "
            "process refuses to start rather than fall back to the direct-MSK "
            "leg the ruling retires (OMN-16459 / OMN-15692 ruling 39)"
        )
    client = client_factory()
    return (
        TransportGatewayHttpsIngest(
            config=ingest,
            tenant_slug=config.forwarder.tenant_identity.tenant_slug,
            client=client,
            auth_token=token,
        ),
        client,
    )


async def run_gateway_forwarder(
    config: ModelGatewayForwarderRuntimeConfig,
    *,
    shutdown_event: asyncio.Event,
    resolve_secret: Callable[[str], Awaitable[str | None]],
    ready_path: Path | None = None,
) -> None:
    """Run the bridge until ``shutdown_event`` is set, then close both legs.

    ``resolve_secret`` is required with no default: OMN-16459's outbound HTTPS
    leg authenticates with the forwarder's own verified actor credential, whose
    VALUE is resolved from the secret store by the reference the node contract
    declares. A default would let a mis-wired process silently fall back to the
    direct-MSK leg ruling 39 (OMN-15692) retires.
    """
    tenant_slug = config.forwarder.tenant_identity.tenant_slug
    local_transport = KafkaTransport(
        config=config.local_bus,
        group=f"tenant-{tenant_slug}-gateway-forwarder-outbound",
        topics=config.forwarder.mirror_topics.outbound,
        auto_offset_reset=config.local_bus.auto_offset_reset,
    )
    cloud_transport = KafkaTransport(
        config=config.cloud_bus,
        group=f"tenant-{tenant_slug}-gateway-forwarder-inbound",
        topics=tuple(
            prefix_topic(tenant_slug, topic)
            for topic in config.forwarder.mirror_topics.inbound
        ),
        auto_offset_reset=config.cloud_bus.auto_offset_reset,
    )
    # OMN-17034: the lane-mirror leg's own transports. Deliberately separate
    # KafkaTransport instances with their own consumer group: a single
    # transport backs one direction's consumer AND another direction's
    # producer (see NodeGatewayDelivery's note on restart_consumer), so
    # sharing one with the trust-boundary legs would couple a stability-lane
    # fault to the dev/cloud delegation path.
    lane_mirror_config = config.forwarder.lane_mirror
    lane_mirror_source: KafkaTransport | None = None
    lane_mirror_producers: dict[str, KafkaTransport] = {}
    if lane_mirror_config is not None:
        source_bus_config = config.lane_mirror_source_bus
        if source_bus_config is None:  # pragma: no cover - runtime config validates
            raise ValueError("lane_mirror declared without a resolved source bus")
        lane_mirror_source = KafkaTransport(
            config=source_bus_config,
            group=f"tenant-{tenant_slug}-gateway-lane-mirror-source",
            topics=lane_mirror_config.topics,
            auto_offset_reset=source_bus_config.auto_offset_reset,
        )
        for lane in lane_mirror_config.mirror_lanes:
            lane_mirror_producers[lane] = KafkaTransport(
                config=config.lane_mirror_buses[lane],
                group=f"tenant-{tenant_slug}-gateway-lane-mirror-{lane}",
                topics=(),
                auto_offset_reset=config.lane_mirror_buses[lane].auto_offset_reset,
            )

    local_bus = TransportGatewayBus(local_transport)
    # OMN-16459: the OUTBOUND publish boundary is whatever the resolved contract
    # says it is. ``cloud_transport`` stays a KafkaTransport either way -- it is
    # still the INBOUND consumer (mirror_topics.inbound is pulled from the cloud
    # broker), which is why this ticket alone does not retire the OMN-16449
    # bastion.
    cloud_outbound, ingest_client = await select_outbound_publish_transport(
        config,
        kafka_bus=TransportGatewayBus(cloud_transport),
        resolve_secret=resolve_secret,
    )
    idempotency_store = StoreIdempotencySqlite(config.forwarder.dedupe_store_path)
    forwarder = ServiceGatewayForwarder(
        config=config.forwarder,
        local_bus=local_bus,
        cloud_bus=cloud_outbound,
    )
    delivery = NodeGatewayDelivery(
        config=config.forwarder,
        forwarder=forwarder,
        local_consumer=local_transport,
        cloud_consumer=cloud_transport,
        idempotency_store=idempotency_store,
    )

    if ready_path is not None:
        ready_path.unlink(missing_ok=True)

    store_started = False
    started_transports: list[KafkaTransport] = []
    delivery_started = False
    heartbeat_task: asyncio.Task[None] | None = None
    lane_mirror_task: asyncio.Task[None] | None = None
    try:
        await idempotency_store.start()
        store_started = True
        await local_transport.start()
        started_transports.append(local_transport)
        await cloud_transport.start()
        started_transports.append(cloud_transport)
        if lane_mirror_source is not None:
            await lane_mirror_source.start()
            started_transports.append(lane_mirror_source)
            for lane_producer in lane_mirror_producers.values():
                await lane_producer.start()
                started_transports.append(lane_producer)
        await delivery.start()
        delivery_started = True
        heartbeat_task = asyncio.create_task(
            _run_heartbeat_loop(forwarder, config, shutdown_event),
            name="gateway-forwarder-heartbeat",
        )
        if lane_mirror_config is not None and lane_mirror_source is not None:
            lane_mirror_task = asyncio.create_task(
                NodeLaneMirror(
                    config=lane_mirror_config,
                    source_consumer=lane_mirror_source,
                    mirror_producers={
                        lane: TransportGatewayBus(transport)
                        for lane, transport in lane_mirror_producers.items()
                    },
                    idempotency_store=idempotency_store,
                ).run(shutdown_event),
                name="gateway-forwarder-lane-mirror",
            )

        if ready_path is not None:
            ready_path.write_text("ready\n", encoding="utf-8")
        identity = config.forwarder.tenant_identity
        logger.info(
            "Gateway forwarder ready for tenant_id=%s tenant_slug=%s",
            identity.tenant_id,
            identity.tenant_slug,
        )
        await _supervise_gateway_delivery(
            forwarder=forwarder,
            delivery=delivery,
            heartbeat_task=heartbeat_task,
            shutdown_event=shutdown_event,
            config=config.forwarder,
        )
    finally:
        if ready_path is not None:
            ready_path.unlink(missing_ok=True)
        if lane_mirror_task is not None and not lane_mirror_task.done():
            lane_mirror_task.cancel()
        if lane_mirror_task is not None:
            await asyncio.gather(lane_mirror_task, return_exceptions=True)
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
        if heartbeat_task is not None:
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if delivery_started:
            await delivery.stop()
        for transport in reversed(started_transports):
            await transport.close()
        if ingest_client is not None:
            await ingest_client.aclose()
        if store_started:
            await idempotency_store.close()


async def _supervise_gateway_delivery(
    *,
    forwarder: ServiceGatewayForwarder,
    delivery: NodeGatewayDelivery,
    heartbeat_task: asyncio.Task[None],
    shutdown_event: asyncio.Event,
    config: ModelGatewayForwarderConfig,
) -> None:
    """Keep the delivery loop alive across cloud-leg faults, no terminal exit.

    A delivery-loop failure (e.g. the cloud broker leg dropping) previously
    propagated straight out of ``run_gateway_forwarder`` and ended the
    process. It is now retried in place with bounded exponential backoff
    and jitter. Once the failure has persisted past the contract-declared
    ``degraded_after_seconds`` window, one ``DEGRADED`` status event is
    published (locally -- see ``ServiceGatewayForwarder.publish_status``)
    so the failure is observable on the bus rather than only in restart
    counts. A restart only clears the failure window once the delivery
    loop has stayed up for a full ``heartbeat_interval_seconds`` recovery
    window without failing again -- a bare ``delivery.start()`` call
    succeeding proves the coroutines were scheduled, not that the cloud
    leg is actually reachable again, so it is deliberately not treated as
    recovery on its own. The process still exits on shutdown, on the
    heartbeat task failing unexpectedly, or on the delivery loop returning
    without either an exception or a shutdown signal (both are
    unrecoverable/programmer errors, not connectivity faults).
    """
    consecutive_failures = 0
    first_failure_at: datetime | None = None
    degraded_emitted = False
    shutdown_wait_task = asyncio.create_task(
        shutdown_event.wait(), name="gateway-shutdown-wait"
    )
    try:
        while True:
            delivery_wait_task = asyncio.create_task(
                delivery.wait(), name="gateway-delivery-health"
            )
            recovery_task: asyncio.Task[None] | None = None
            if consecutive_failures > 0:
                recovery_task = asyncio.create_task(
                    asyncio.sleep(config.heartbeat_interval_seconds),
                    name="gateway-delivery-recovery-confirm",
                )
            waitables: set[asyncio.Task[object]] = {
                delivery_wait_task,
                shutdown_wait_task,
                heartbeat_task,
            }
            if recovery_task is not None:
                waitables.add(recovery_task)
            try:
                done, _ = await asyncio.wait(
                    waitables, return_when=asyncio.FIRST_COMPLETED
                )
                if shutdown_wait_task in done:
                    return
                if heartbeat_task in done:
                    await heartbeat_task
                    return
                if (
                    recovery_task is not None
                    and recovery_task in done
                    and delivery_wait_task not in done
                ):
                    # Survived a full heartbeat interval without a new
                    # failure -- treat the connection as recovered.
                    consecutive_failures = 0
                    first_failure_at = None
                    if degraded_emitted:
                        await _publish_gateway_status(forwarder, status="active")
                        degraded_emitted = False
                    continue
                exc = delivery_wait_task.exception()
                if exc is None:
                    raise RuntimeError(
                        "gateway delivery loop exited without a shutdown signal"
                    )
            finally:
                if recovery_task is not None and not recovery_task.done():
                    recovery_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await recovery_task
                if not delivery_wait_task.done():
                    delivery_wait_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await delivery_wait_task

            consecutive_failures += 1
            now = datetime.now(UTC)
            if first_failure_at is None:
                first_failure_at = now
            elapsed_seconds = (now - first_failure_at).total_seconds()
            logger.warning(
                "Gateway delivery loop failed; reconnect attempt=%d "
                "elapsed_seconds=%.1f error_type=%s error=%s",
                consecutive_failures,
                elapsed_seconds,
                type(exc).__name__,
                exc,
            )

            degraded_threshold = config.degraded_after_seconds
            if not degraded_emitted and elapsed_seconds >= degraded_threshold:
                await _publish_gateway_status(
                    forwarder,
                    status="degraded",
                    consecutive_failures=consecutive_failures,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                degraded_emitted = True

            delay = _compute_reconnect_delay_seconds(config, consecutive_failures)
            shutdown_fired = await _sleep_or_shutdown(delay, shutdown_event)
            if shutdown_fired:
                return

            await delivery.stop()
            try:
                await delivery.start()
            except Exception:
                logger.exception("Gateway delivery restart failed; will retry")
                continue
    finally:
        if not shutdown_wait_task.done():
            shutdown_wait_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_wait_task


def _compute_reconnect_delay_seconds(
    config: ModelGatewayForwarderConfig,
    attempt: int,
) -> float:
    """Bounded exponential backoff with additive jitter, contract-declared."""
    exponential = config.reconnect_backoff_initial_seconds * (2 ** (attempt - 1))
    capped = min(exponential, config.reconnect_backoff_max_seconds)
    jitter = random.uniform(0, config.reconnect_backoff_jitter_seconds)
    return capped + jitter


async def _sleep_or_shutdown(delay: float, shutdown_event: asyncio.Event) -> bool:
    """Sleep for ``delay`` seconds; return True if shutdown fired first."""
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
    except TimeoutError:
        return False
    return True


async def _publish_gateway_status(
    forwarder: ServiceGatewayForwarder,
    *,
    status: Literal["active", "degraded"],
    consecutive_failures: int = 0,
    detail: str = "",
) -> None:
    """Best-effort status publish -- must never itself take down supervision."""
    try:
        await forwarder.publish_status(
            status,
            consecutive_failures=consecutive_failures,
            detail=detail,
        )
    except Exception:
        logger.exception("Gateway %s status publish failed", status)


class TransportGatewayBus:
    """Adapt the pull transport producer to the forwarder's publish boundary."""

    def __init__(self, producer: ProtocolTransportProducer) -> None:
        self._producer = producer

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        encoded_headers: Mapping[str, bytes]
        if headers is None:
            encoded_headers = {}
        elif isinstance(headers, Mapping):
            if not all(
                isinstance(header_key, str) and isinstance(header_value, bytes)
                for header_key, header_value in headers.items()
            ):
                raise TypeError(
                    "gateway transport headers must map string keys to bytes"
                )
            encoded_headers = {
                header_key: header_value
                for header_key, header_value in headers.items()
                if isinstance(header_key, str) and isinstance(header_value, bytes)
            }
        elif isinstance(headers, ModelEventHeaders):
            encoded_headers = {
                header_key: str(header_value).encode("utf-8")
                for header_key, header_value in headers.model_dump(
                    mode="json",
                    exclude_none=True,
                ).items()
            }
        else:
            raise TypeError("gateway transport headers must map string keys to bytes")
        try:
            await self._producer.send(topic, key, value, encoded_headers)
        except KafkaError as exc:
            raise InfraUnavailableError(
                f"gateway destination broker unavailable for topic {topic}"
            ) from exc


async def _run_heartbeat_loop(
    forwarder: ServiceGatewayForwarder,
    config: ModelGatewayForwarderRuntimeConfig,
    shutdown_event: asyncio.Event,
) -> None:
    """Emit immediately, then at the contract-declared liveness cadence."""
    interval = config.forwarder.heartbeat_interval_seconds
    while not shutdown_event.is_set():
        await forwarder.publish_heartbeat()
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
        except TimeoutError:
            continue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one tenant-scoped local/cloud event-bus forwarder",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the resolved, typed gateway forwarder YAML",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help=(
            "Optional readiness sentinel written only after both broker transports "
            "and delivery loops start"
        ),
    )
    parser.add_argument(
        "--broker-ref-map",
        type=Path,
        required=True,
        help=(
            "Path to the operator-supplied broker-ref resolution map (YAML "
            "mapping of contract cloud_broker_ref names to resolved "
            "bootstrap_servers strings). Resolved at the effect boundary; "
            "required, no default -- the process fails closed without it"
        ),
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    config = load_gateway_forwarder_runtime_config(
        args.config,
        broker_ref_map_path=args.broker_ref_map,
    )
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)
    await run_gateway_forwarder(
        config,
        shutdown_event=shutdown_event,
        resolve_secret=AdapterEnvSecretStore().get_secret,
        ready_path=args.ready_file,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
