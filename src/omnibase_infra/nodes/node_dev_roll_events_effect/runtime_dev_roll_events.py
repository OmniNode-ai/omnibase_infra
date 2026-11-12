# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root of the dev roll events effect (OMN-24013).

The deploy agent loads this by file path. The topics and the incident log byte cap are read
from the node's contract.yaml, located from the agent's own executor module (the way the
estimator build effect locates its declaration), and handed to the publisher; nothing here
or in the publisher holds a topic or a bound.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml
from pydantic import JsonValue

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_dev_roll_events import (
        ProtocolDevRollEvents,
    )

_NODE = "src/omnibase_infra/nodes/node_dev_roll_events_effect"


def _load_events() -> ModuleType:
    """The publisher module, loaded by file path under its dotted name, once."""
    name = "omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("dev_roll_events.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the dev roll events publisher")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


events = _load_events()
EnumDevStartCause = events.EnumDevStartCause
docker_started_at = events.docker_started_at


def declaration_at(repo_root: Path) -> dict[str, JsonValue]:
    """The node's declared topics and settings, from the checkout's contract."""
    contract = repo_root / _NODE / "contract.yaml"
    declared: dict[str, JsonValue] = yaml.safe_load(
        contract.read_text(encoding="utf-8")
    )
    return declared


def declaration(execution: ModuleType) -> dict[str, JsonValue]:
    """The node's declared topics and settings, from the agent checkout's contract."""
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("dev roll events need the deploy agent source location")
    return declaration_at(Path(module_file).resolve().parents[3])


def log_byte_cap(execution: ModuleType) -> int:
    """The incident log byte cap the contract declares."""
    settings = declaration(execution)["settings"]
    cap = settings.get("log_byte_cap") if isinstance(settings, dict) else None
    if not isinstance(cap, int):
        raise RuntimeError("the dev roll events contract declares no log_byte_cap")
    return cap


def agent_send() -> Callable[[str, str, dict[str, JsonValue]], bool]:
    """The real sender of the running deploy agent: its own producer and topic rule."""
    try:
        publisher = importlib.import_module("deploy_agent.publisher")
        kafka_config = importlib.import_module("deploy_agent.kafka_config")
    except ImportError:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[4] / "scripts" / "deploy-agent")
        )
        publisher = importlib.import_module("deploy_agent.publisher")
        kafka_config = importlib.import_module("deploy_agent.kafka_config")
    kafka = kafka_config.load_deploy_agent_kafka_config_from_env()

    def send(topic: str, key: str, payload: dict[str, JsonValue]) -> bool:
        from omnibase_infra.topics.topic_namespace import apply_topic_namespace

        publisher._ensure_topic(apply_topic_namespace(topic), kafka)
        return bool(publisher._produce(topic, key, payload, kafka))

    return send


def build_events(execution: ModuleType) -> object:
    """The publisher of the running deploy agent."""
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("dev roll events need the deploy agent source location")
    return build_events_at(Path(module_file).resolve().parents[3])


def build_events_at(repo_root: Path) -> ProtocolDevRollEvents:
    """The publisher built from the checkout's contract, from the repo root."""
    topics = declaration_at(repo_root)["topics"]
    if not isinstance(topics, dict):
        raise RuntimeError("the dev roll events contract declares no topics")
    publisher: ProtocolDevRollEvents = events.DevRollEventsPublisher(
        agent_send(), {str(k): str(v) for k, v in topics.items()}
    )
    return publisher
