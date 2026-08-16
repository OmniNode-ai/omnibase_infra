# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Standalone process entrypoint for the hybrid gateway bus forwarder."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import random
import signal
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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
    _materialize_cloud_broker_ref(raw, contract_path, broker_ref_map_path)
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
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
        if heartbeat_task is not None:
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if delivery_started:
            await delivery.stop()
        for transport in reversed(started_transports):
            await transport.close()
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
