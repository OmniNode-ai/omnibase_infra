#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Publish a command that triggers NodeBaselinesBatchCompute.

This script is intentionally a thin bus publisher. It validates the existing
``ModelBaselinesBatchComputeCommand`` payload, wraps it in a
``ModelEventEnvelope``, publishes to the node command topic, then exits. The
runtime-owned node handler performs database work and emits the
baselines-computed event.

Usage:
    uv run python scripts/run_baselines_batch_compute.py
    uv run python scripts/run_baselines_batch_compute.py --dry-run

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS (optional, default: localhost:19092)
        Kafka bootstrap address for publishing the command.

Exit Codes:
    0  Command published, or dry-run payload validated
    1  Configuration, validation, or delivery error

Ticket: OMN-3335, OMN-11177
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

import yaml  # ONEX_EXCLUDE: manual_yaml - reads node contract for command topic

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092"
SOURCE_TOOL = "run_baselines_batch_compute"
TARGET_TOOL = "node_baselines_batch_compute"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(SOURCE_TOOL)


class _BaselinesBatchCommand(Protocol):
    """Structural protocol for the command model loaded from the node model file."""

    correlation_id: UUID

    def model_dump(self, *, mode: str) -> dict[str, object]:
        pass


def _command_model_module_path() -> Path:
    return _node_contract_dir() / "models" / "model_baselines_batch_compute_command.py"


def _node_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_baselines_batch_compute"
    )


def _load_command_topic() -> str:
    contract_path = _node_contract_dir() / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        message = f"Invalid node contract at {contract_path}"
        raise RuntimeError(message)
    event_bus = contract.get("event_bus")
    if not isinstance(event_bus, dict):
        message = f"Node contract has no event_bus section: {contract_path}"
        raise RuntimeError(message)
    subscribe_topics = event_bus.get("subscribe_topics")
    if not isinstance(subscribe_topics, list):
        message = f"Node contract has no event_bus.subscribe_topics: {contract_path}"
        raise RuntimeError(message)
    command_topics = [
        topic.strip()
        for topic in subscribe_topics
        if isinstance(topic, str) and topic.strip().startswith("onex.cmd.")
    ]
    if len(command_topics) != 1:
        message = (
            "Node contract must declare exactly one onex.cmd.* subscribe topic; "
            f"found {len(command_topics)} in {contract_path}"
        )
        raise RuntimeError(message)
    return command_topics[0]


def _load_command_model_class() -> type[Any]:
    """Load the command model without importing the node package root.

    The node package ``__init__`` currently re-exports its handler. Importing
    the model through the package path would therefore import the handler as a
    side effect, which defeats this script's bus-publisher boundary.
    """
    module_path = _command_model_module_path()
    spec = importlib.util.spec_from_file_location(
        "_onex_baselines_batch_compute_command_model",
        module_path,
    )
    if spec is None or spec.loader is None:
        message = f"Cannot load command model from {module_path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("type[Any]", module.ModelBaselinesBatchComputeCommand)


def build_command(correlation_id: UUID) -> _BaselinesBatchCommand:
    """Build and validate the typed baselines batch compute command."""
    command_model = _load_command_model_class()
    return cast("_BaselinesBatchCommand", command_model(correlation_id=correlation_id))


def build_envelope(correlation_id: UUID) -> ModelEventEnvelope[dict[str, object]]:
    """Build the command envelope published to the event bus."""
    command = build_command(correlation_id)
    command_topic = _load_command_topic()
    return ModelEventEnvelope[dict[str, object]](
        payload=command.model_dump(mode="json"),
        correlation_id=command.correlation_id,
        event_type=command_topic,
        source_tool=SOURCE_TOOL,
        target_tool=TARGET_TOOL,
        payload_type="ModelBaselinesBatchComputeCommand",
    )


def _publish(
    envelope: ModelEventEnvelope[dict[str, object]], bootstrap_servers: str
) -> None:
    """Publish a command envelope to Kafka synchronously."""
    topic = str(envelope.event_type or "").strip()
    if not topic:
        message = "Command envelope has no event_type topic"
        raise RuntimeError(message)
    try:
        from confluent_kafka import Producer
    except ImportError as exc:
        message = (
            "confluent-kafka is required. Install it with: pip install confluent-kafka"
        )
        raise RuntimeError(message) from exc

    producer = Producer({"bootstrap.servers": bootstrap_servers})
    delivery_error: list[BaseException] = []

    def _on_delivery(err: object, _msg: object) -> None:
        if err is not None:
            delivery_error.append(RuntimeError(f"Kafka delivery failed: {err}"))

    producer.produce(
        topic=topic,
        key=str(envelope.correlation_id).encode("utf-8"),
        value=envelope.model_dump_json().encode("utf-8"),
        on_delivery=_on_delivery,
    )
    remaining = producer.flush(timeout=10.0)
    if remaining:
        message = f"Kafka delivery timed out with {remaining} message(s) still queued"
        raise RuntimeError(message)
    if delivery_error:
        raise delivery_error[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a baselines batch compute command to the ONEX bus.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS),
        help="Kafka bootstrap servers (default: KAFKA_BOOTSTRAP_SERVERS or localhost:19092).",
    )
    parser.add_argument(
        "--correlation-id",
        type=UUID,
        default=None,
        help="Correlation UUID to use. Defaults to a generated UUID4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the command envelope without publishing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    correlation_id = args.correlation_id or uuid4()

    try:
        envelope = build_envelope(correlation_id)
    except Exception:
        logger.exception("Failed to build baselines batch compute command")
        return 1

    envelope_json = envelope.model_dump(mode="json")
    if args.dry_run:
        print(json.dumps(envelope_json, indent=2, sort_keys=True))
        print(f"(dry-run: skipping publish to {envelope.event_type})")
        return 0

    bootstrap_servers = str(args.bootstrap_servers or "").strip()
    if not bootstrap_servers:
        logger.error("Kafka bootstrap servers resolved empty; cannot publish command")
        return 1

    try:
        _publish(envelope, bootstrap_servers)
    except Exception:
        logger.exception("Failed to publish baselines batch compute command")
        return 1

    logger.info(
        "Published baselines batch compute command to %s (correlation_id=%s)",
        envelope.event_type,
        correlation_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
