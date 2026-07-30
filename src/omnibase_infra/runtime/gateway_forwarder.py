# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Standalone process entrypoint for the hybrid gateway bus forwarder."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    NodeGatewayDelivery,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
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
) -> ModelGatewayForwarderRuntimeConfig:
    """Load and validate one explicit two-leg forwarder configuration."""
    raw_object: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_object, dict):
        raise ValueError("gateway forwarder config must be a YAML mapping")
    raw: dict[str, object] = {str(key): value for key, value in raw_object.items()}
    _materialize_contract_mirror_topics(raw, contract_path)
    return ModelGatewayForwarderRuntimeConfig.model_validate(raw)


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
    mirror_topics_object = gateway_config.get("mirror_topics")
    if not isinstance(mirror_topics_object, dict):
        raise ValueError("gateway node contract mirror_topics must be a mapping")
    forwarder["mirror_topics"] = {
        str(key): value for key, value in mirror_topics_object.items()
    }


async def run_gateway_forwarder(
    config: ModelGatewayForwarderRuntimeConfig,
    *,
    shutdown_event: asyncio.Event,
    ready_path: Path | None = None,
) -> None:
    """Run the bridge until ``shutdown_event`` is set, then close both legs."""
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
    local_bus = TransportGatewayBus(local_transport)
    cloud_bus = TransportGatewayBus(cloud_transport)
    idempotency_store = StoreIdempotencySqlite(config.forwarder.dedupe_store_path)
    forwarder = ServiceGatewayForwarder(
        config=config.forwarder,
        local_bus=local_bus,
        cloud_bus=cloud_bus,
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
    delivery_wait_task: asyncio.Task[None] | None = None
    shutdown_wait_task: asyncio.Task[bool] | None = None
    try:
        await idempotency_store.start()
        store_started = True
        await local_transport.start()
        started_transports.append(local_transport)
        await cloud_transport.start()
        started_transports.append(cloud_transport)
        await delivery.start()
        delivery_started = True
        heartbeat_task = asyncio.create_task(
            _run_heartbeat_loop(forwarder, config, shutdown_event),
            name="gateway-forwarder-heartbeat",
        )

        if ready_path is not None:
            ready_path.write_text("ready\n", encoding="utf-8")
        identity = config.forwarder.tenant_identity
        logger.info(
            "Gateway forwarder ready for tenant_id=%s tenant_slug=%s",
            identity.tenant_id,
            identity.tenant_slug,
        )
        delivery_wait_task = asyncio.create_task(
            delivery.wait(),
            name="gateway-delivery-health",
        )
        shutdown_wait_task = asyncio.create_task(
            shutdown_event.wait(),
            name="gateway-shutdown-wait",
        )
        done, _ = await asyncio.wait(
            {delivery_wait_task, shutdown_wait_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if delivery_wait_task in done:
            await delivery_wait_task
        if heartbeat_task in done:
            await heartbeat_task
    finally:
        if ready_path is not None:
            ready_path.unlink(missing_ok=True)
        for waiter in (delivery_wait_task, shutdown_wait_task):
            if waiter is not None and not waiter.done():
                waiter.cancel()
        waiters = [
            waiter
            for waiter in (delivery_wait_task, shutdown_wait_task)
            if waiter is not None
        ]
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if delivery_started:
            await delivery.stop()
        for transport in reversed(started_transports):
            await transport.close()
        if store_started:
            await idempotency_store.close()


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
        help="Optional readiness sentinel written only after both subscriptions start",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    config = load_gateway_forwarder_runtime_config(args.config)
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)
    await run_gateway_forwarder(
        config,
        shutdown_event=shutdown_event,
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
