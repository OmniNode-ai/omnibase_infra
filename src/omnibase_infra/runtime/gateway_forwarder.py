# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Standalone process entrypoint for the hybrid gateway bus forwarder."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence
from pathlib import Path

import yaml

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
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
    local_bus = EventBusKafka(config=config.local_bus)
    cloud_bus = EventBusKafka(config=config.cloud_bus)
    forwarder = ServiceGatewayForwarder(
        config=config.forwarder,
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )

    if ready_path is not None:
        ready_path.unlink(missing_ok=True)

    started_buses: list[EventBusKafka] = []
    forwarder_started = False
    heartbeat_task: asyncio.Task[None] | None = None
    try:
        await local_bus.start()
        started_buses.append(local_bus)
        await cloud_bus.start()
        started_buses.append(cloud_bus)
        await forwarder.start()
        forwarder_started = True
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
        await shutdown_event.wait()
    finally:
        if ready_path is not None:
            ready_path.unlink(missing_ok=True)
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        if forwarder_started:
            await forwarder.stop()
        for bus in reversed(started_buses):
            await bus.close()


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
