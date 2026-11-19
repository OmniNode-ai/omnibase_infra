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
import json
import logging
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import yaml
from pydantic import JsonValue

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_dev_roll_events import (
        ProtocolDevRollEvents,
    )

logger = logging.getLogger(__name__)


class ProtocolRecreateGate(Protocol):
    """What the dev recreate window gate offers its callers."""

    def allow(
        self, requested_by: str, observed_last: datetime | None = None
    ) -> bool: ...


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


class RefusingRecreateGate:
    """The gate when the window policy cannot be read: every recreate waits, none is guessed."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def allow(self, requested_by: str, observed_last: datetime | None = None) -> bool:
        logger.error(
            "dev recreate refused for %s: the window policy is unreadable (%s) "
            "friction_type=dev_recreate_policy_unreadable",
            requested_by,
            self._reason,
        )
        return False


def recreate_gate_at(
    repo_root: Path,
    last_recreate_at: datetime | None,
    announced_for: datetime | None = None,
) -> ProtocolRecreateGate:
    """The dev recreate window gate: the policy from the checkout's rendered env, announcing
    through the node's own publisher (OMN-25912). An unreadable policy is a refusing gate,
    never an open window."""
    try:
        for relative in (
            "enums/enum_dev_recreate_verdict.py",
            "enums/enum_dev_recreate_reason.py",
            "models/model_dev_recreate_policy.py",
            "models/model_dev_recreate_request.py",
            "models/model_dev_recreate_decision.py",
            "models/model_dev_recreate_announcement.py",
            "handlers/dev_recreate_window/handler_dev_recreate_window.py",
        ):
            events._load_sibling(relative)
        window = sys.modules["omnibase_infra.models.model_dev_recreate_policy"]
        handler = sys.modules[
            "omnibase_infra.handlers.dev_recreate_window.handler_dev_recreate_window"
        ]
        env: dict[str, str] = {}
        policy_env = repo_root / "docker" / "runtime-policy.env"
        for line in policy_env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                env[key.strip()] = value.strip()
        return handler.DevRecreateWindowGate(
            window.ModelDevRecreatePolicy.from_env(env),
            build_events_at(repo_root).recreate_announced,
            clock=lambda: datetime.now(UTC),
            last_recreate_at=last_recreate_at,
            announced_for=announced_for,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return RefusingRecreateGate(f"{type(exc).__name__}: {exc}")


def newest_dev_recreate(state_dir: Path) -> datetime | None:
    """When the newest dev job that ran the runtime phase was accepted, from the job records."""
    newest: datetime | None = None
    for path in state_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            phases = data.get("phase_results") or {}
            if data["command"]["runtime_lane"] != "dev" or phases.get(
                "runtime"
            ) not in (
                "success",
                "failed",
            ):
                continue
            accepted = datetime.fromisoformat(data["accepted_at"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if newest is None or accepted > newest:
            newest = accepted
    return newest


def window_admits(agent: object, requested_by: str) -> bool:
    """Whether the dev recreate window admits ``requested_by`` now (OMN-25912).

    The agent holds one gate, built on first use from the checkout's rendered policy;
    an agent that carries none (a test double) admits. The newest recreate in the job
    records spends the window, so a merge's recreate counts too.
    """
    if not hasattr(agent, "_recreate_gate"):
        agent.__dict__["_recreate_gate"] = recreate_gate_at(
            Path(__file__).resolve().parents[4], None
        )
    gate = agent.__dict__["_recreate_gate"]
    store = getattr(agent, "job_store", None)
    observed = newest_dev_recreate(Path(store.state_dir)) if store else None
    return gate is None or bool(gate.allow(requested_by, observed))


def announce_preflight_failure(
    execution: ModuleType,
    command: Mapping[str, JsonValue],
    error: str,
    phase: str,
) -> None:
    """Publish the preflight-failure events for a dev roll; never fails or delays a roll."""
    try:
        if str(command.get("runtime_lane")) != "dev":
            return
        module_file = execution.__file__
        if not module_file:
            raise RuntimeError("dev roll events need the deploy agent source location")
        events = build_events_at(Path(module_file).resolve().parents[3])
        lane = execution.lane_config_for(execution.EnumRuntimeLane.DEV)
        events.preflight_failure(
            command, error=error, phase=phase, project=lane.compose_project
        )
    except Exception:
        logger.exception(
            "dev preflight incident could not be announced friction_type=dev_event_publish_failed"
        )
