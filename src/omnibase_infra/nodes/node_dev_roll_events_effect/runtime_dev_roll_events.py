# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root of the dev roll events effect (OMN-24013).

The deploy agent loads this by file path. The topics and the incident log byte cap are read
from the node's contract.yaml, located from the agent's own executor module (the way the
estimator build effect locates its declaration), and handed to the publisher; nothing here
or in the publisher holds a topic or a bound.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import json
import logging
import operator
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

import yaml
from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    from omnibase_infra.models.model_dev_container_reading import (
        ModelDevContainerReading,
    )
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


def registered_installs() -> ModuleType:
    """The registered branch-install rules (OMN-26300), loaded by file path under their dotted name."""
    name = "omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_registered_installs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("runtime_dev_registered_installs.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the registered branch-install rules")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def build_events(execution: ModuleType, state_dir: Path | None = None) -> object:
    """The publisher of the running deploy agent, journaling beneath ``state_dir`` when named."""
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("dev roll events need the deploy agent source location")
    return build_events_at(Path(module_file).resolve().parents[3], state_dir)


def journal_dir_at(repo_root: Path, state_dir: Path | None) -> Path | None:
    """The directory the agent journals its accepted events in (OMN-25912), or ``None`` when
    the caller names no state directory: the event is then published and not journaled."""
    if state_dir is None:
        return None
    return state_dir / _setting_str(_settings(repo_root), "event_journal_dir")


def journaled_send(
    send: Callable[[str, str, dict[str, JsonValue]], bool],
    journal_dir: Path | None,
    topics: Mapping[str, str],
    input_roles: tuple[str, ...],
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Callable[[str, str, dict[str, JsonValue]], bool]:
    """Wrap a sender so each event the bus accepts on an audit input topic is also appended
    to the day's journal file. The reader reads its input from that journal (OMN-25912)."""
    role_of = {topics[role]: role for role in input_roles if role in topics}

    def sender(topic: str, key: str, payload: dict[str, JsonValue]) -> bool:
        accepted = send(topic, key, payload)
        role = role_of.get(topic)
        if accepted and role is not None and journal_dir is not None:
            now = clock()
            try:
                journal_dir.mkdir(parents=True, exist_ok=True)
                line = json.dumps(
                    {
                        "role": role,
                        "journaled_at": now.isoformat(),
                        "payload": payload,
                    },
                    sort_keys=True,
                )
                with (journal_dir / f"{now:%Y-%m-%d}.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(line + "\n")
            except OSError:
                logger.exception(
                    "dev event journal write failed "
                    "friction_type=dev_event_publish_failed"
                )
        return accepted

    return sender


def build_events_at(
    repo_root: Path,
    state_dir: Path | None = None,
    *,
    send: Callable[[str, str, dict[str, JsonValue]], bool] | None = None,
) -> ProtocolDevRollEvents:
    """The publisher built from the checkout's contract, from the repo root.

    Every accepted event of an audit input role is journaled under the agent's job state
    directory ``state_dir`` when the caller names it. ``send`` stands in for the agent's
    producer in tests.
    """
    declared = declaration_at(repo_root)
    topics = declared["topics"]
    if not isinstance(topics, dict):
        raise RuntimeError("the dev roll events contract declares no topics")
    named = {str(k): str(v) for k, v in topics.items()}
    publisher: ProtocolDevRollEvents = events.DevRollEventsPublisher(
        journaled_send(
            send or agent_send(),
            journal_dir_at(repo_root, state_dir),
            named,
            _input_roles(declared),
        ),
        named,
    )
    return publisher


def _input_roles(declared: Mapping[str, JsonValue]) -> tuple[str, ...]:
    settings = declared["settings"]
    roles = settings.get("audit_input_topics") if isinstance(settings, dict) else None
    if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
        raise RuntimeError(
            "the dev roll events contract declares no audit_input_topics"
        )
    return tuple(str(r) for r in roles)


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


def policy_env_at(repo_root: Path) -> dict[str, str]:
    """The declared recreate window policy, as the checkout's rendered runtime-policy.env."""
    env: dict[str, str] = {}
    policy_env = repo_root / "docker" / "runtime-policy.env"
    for line in policy_env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return env


def recreate_gate_at(
    repo_root: Path,
    last_recreate_at: datetime | None,
    announced_for: datetime | None = None,
    *,
    announce: Callable[[object], bool] | None = None,
    clock: Callable[[], datetime] | None = None,
    state_dir: Path | None = None,
) -> ProtocolRecreateGate:
    """The dev recreate window gate: the policy from the checkout's rendered env, announcing
    through the node's own publisher (OMN-25912). An unreadable policy is a refusing gate,
    never an open window. ``announce`` and ``clock`` stand in for the publisher and the wall
    clock: the one-shot self-check of a candidate agent passes a recorder and a fake clock
    (OMN-26494)."""
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
        return handler.DevRecreateWindowGate(
            window.ModelDevRecreatePolicy.from_env(policy_env_at(repo_root)),
            announce or build_events_at(repo_root, state_dir).recreate_announced,
            clock=clock or (lambda: datetime.now(UTC)),
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


#: ``requested_by`` of the deploy agent's restore of a dev lane whose runtime is not running.
IDLE_RESTORE_REQUESTER = "agent/idle-restore"

#: OMN-26369: the requester a restore asks the window under when a runtime-image merge is
#: waiting. Declared in ``dev_recreate_window.pinned_fix_requester_prefixes`` of
#: ``contracts/services/runtime_policy.contract.yaml``, so the window admits it off the hour.
MERGE_RESTORE_REQUESTER = "agent/merge-restore"

#: The clone's own upstream: the branch the deploy agent's checkout tracks.
UPSTREAM_REF = "@{upstream}"

#: OMN-26369: what ``docker/Dockerfile.runtime`` copies from this repository into the runtime
#: image, plus the lane state a compose up applies. A commit that touches none of these
#: changes nothing the runtime runs. ``tests/unit/scripts/test_deploy_agent_merge_trigger.py``
#: fails when a ``COPY`` source of the Dockerfile is missing here. A trailing ``/**`` is a
#: whole directory; any other entry is one file or a glob over one path.
RUNTIME_IMAGE_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "README.md",
    "uv.lock",
    "src/**",
    "contracts/**",
    "config/runner_fleet.yaml",
    "docker/Dockerfile.runtime",
    "docker/entrypoint-runtime.sh",
    "docker/catalog/**",
    "docker/migrations/**",
    "docker/runtime-policy.env",
    "docker/docker-compose*.yml",
    "docker/docker-compose*.yaml",
    "scripts/run-forward-migrations.sh",
    "scripts/runtime_build/**",
)


def is_runtime_image_path(path: str) -> bool:
    """Whether a changed file is one the runtime image or its lane state is built from."""
    for pattern in RUNTIME_IMAGE_PATHS:
        if pattern.endswith("/**"):
            if path.startswith(pattern[:-2]):
                return True
        elif fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def newest_dev_job_accepted(state_dir: Path) -> datetime | None:
    """When the newest dev job was accepted, from the job records; ``None`` when none.

    That job staged its source when it started, so a runtime merge committed after it is a
    merge no job has built.
    """
    newest: datetime | None = None
    for path in state_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["command"]["runtime_lane"] != "dev":
                continue
            accepted = datetime.fromisoformat(data["accepted_at"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if newest is None or accepted > newest:
            newest = accepted
    return newest


def read_runtime_merge(
    clone: Path,
    ref: str,
    since: datetime,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> datetime | None:
    """The commit time of the newest commit on ``ref`` after ``since`` that changes the runtime image.

    Read from ``clone`` as it is, bounded by ``--max-count``. An unreadable clone is no merge,
    never a guess.
    """
    try:
        result = run(
            [
                "git",
                "-C",
                str(clone),
                "log",
                ref,
                "--max-count=200",
                "--format=%x01%ct",
                "--name-only",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    newest: datetime | None = None
    for block in result.stdout.split("\x01")[1:]:
        stamp, *paths = block.splitlines()
        try:
            committed = datetime.fromtimestamp(int(stamp), tz=UTC)
        except ValueError:
            return None
        if committed <= since:
            continue
        if any(is_runtime_image_path(p.strip()) for p in paths if p.strip()) and (
            newest is None or committed > newest
        ):
            newest = committed
    return newest


def _runtime_merge_waiting(state_dir: Path) -> datetime | None:
    """The newest runtime-image commit on the clone's tracked dev head that no dev job has built.

    The tracked head is the clone's own upstream: the deploy agent's checkout follows the
    branch ``DEPLOY_AGENT_TRACKING_REF`` declares, so no branch name is spelled here.
    """
    newest_job = newest_dev_job_accepted(state_dir)
    if newest_job is None:
        return None
    return read_runtime_merge(
        Path(__file__).resolve().parents[4], UPSTREAM_REF, newest_job
    )


def window_deferral(agent: object, requested_by: str) -> tuple[datetime, str] | None:
    """When and why the recreate window holds ``requested_by``, ``None`` when it admits (OMN-25912)."""
    if window_admits(agent, requested_by):
        return None
    decision = getattr(agent.__dict__.get("_recreate_gate"), "last_decision", None)
    if decision is None:
        return datetime.now(UTC) + timedelta(
            seconds=60
        ), "the window gate holds no decision"
    return decision.next_recreate_at, (
        f"{decision.reason.value}, next recreate at {decision.next_recreate_at.isoformat()}"
    )


def window_admits(agent: object, requested_by: str) -> bool:
    """Whether the dev recreate window admits ``requested_by`` now (OMN-25912).

    The agent holds one gate, built on first use from the checkout's rendered policy;
    an agent that carries none (a test double) admits. The newest recreate in the job
    records spends the window, so a merge's recreate counts too.

    OMN-26369: the restore of a lane whose runtime is down is the agent's only path to a merge
    it never hears about (the chain that publishes the merge's rebuild command runs inside
    that runtime). The window held it from 14:26Z to 15:00Z on 2026-11-21 while omnibase_infra
    merges of 13:45:48Z and 14:06:14Z waited. So an idle restore the window refuses is asked
    again as ``agent/merge-restore`` when a runtime-image commit is on the dev head that no dev
    job has built, and the window admits that requester off the hour. Nothing else changes:
    the restore's own bounds (down lane, quiet, freeze, load gate) were already met to get here.
    """
    store = getattr(agent, "job_store", None)
    if not hasattr(agent, "_recreate_gate"):
        agent.__dict__["_recreate_gate"] = recreate_gate_at(
            Path(__file__).resolve().parents[4],
            None,
            state_dir=Path(store.state_dir) if store else None,
        )
    gate = agent.__dict__["_recreate_gate"]
    observed = newest_dev_recreate(Path(store.state_dir)) if store else None
    if gate is None or bool(gate.allow(requested_by, observed)):
        return True
    if requested_by != IDLE_RESTORE_REQUESTER or store is None:
        return False
    merged = _runtime_merge_waiting(Path(store.state_dir))
    if merged is None:
        return False
    admitted = bool(gate.allow(MERGE_RESTORE_REQUESTER, observed))
    if admitted:
        logger.info(
            "dev recreate window admits the idle restore for a runtime-image merge committed "
            "%s that no dev job has built",
            merged.isoformat(),
        )
    return admitted


_DEV_TAG_REF = re.compile(r"(?:refs/tags/)?dev-[0-9]{14}-[0-9a-f]{7,40}")


def is_lane_build(cmd: object) -> bool:
    """A dev build a lane asked for under a rollback declaration (OMN-25912).

    The declaration is how a lane gets past the lineage fence, so a lane's build of its
    branch carries one. A real rollback goes to a pinned good build, a dev tag or an
    image, and only the first kind recreates the runtime on a lane's own say-so.
    """
    return bool(
        str(getattr(getattr(cmd, "runtime_lane", None), "value", "")) == "dev"
        and getattr(cmd, "rollback", None) is not None
        and not getattr(cmd, "image_ref", None)
        and not getattr(cmd, "image_digest", None)
        and not _DEV_TAG_REF.fullmatch(str(getattr(cmd, "git_ref", "")))
    )


def lane_build_gate(agent: object) -> LaneBuildWindowGate:
    """The consumer's gate for ``agent``: its load gate, if any, with the window folded in."""
    return LaneBuildWindowGate(agent, getattr(agent, "_load_gate", None))


class LaneBuildWindowGate:
    """The consumer's load gate with the dev recreate window folded in (OMN-25912).

    The consumer defers a command while ``check`` reads DEFER and resumes its partition
    once a ``check`` with no command reads OPEN. This gate answers the host's own load
    gate first, then asks the window for a lane's build request, which recreates the
    runtime as the idle paths do. A real rollback never asks. A held request is named,
    with the time of the next window, until that time passes.
    """

    def __init__(self, agent: object, inner: object | None) -> None:
        self._agent = agent
        self._inner = inner
        self._until: datetime | None = None
        self.is_lane_build = is_lane_build

    def recheck_due(self) -> bool:
        return (
            True
            if self._inner is None
            else bool(operator.attrgetter("recheck_due")(self._inner)())
        )

    def check(self, cmd: object | None = None) -> object:
        verdicts = importlib.import_module("deploy_agent.load_gate").EnumLoadGateVerdict
        decision = (
            operator.attrgetter("check")(self._inner)()
            if self._inner is not None
            else None
        )
        if getattr(decision, "verdict", None) is verdicts.DEFER:
            return decision
        held: tuple[datetime, str] | None = None
        if cmd is None:
            if self._until is not None and datetime.now(UTC) < self._until:
                held = (self._until, "the recreate window has not opened")
        elif is_lane_build(cmd):
            held = window_deferral(self._agent, str(getattr(cmd, "requested_by", "")))
        self._until = held[0] if held else None
        if held is None:
            return decision or SimpleNamespace(
                verdict=verdicts.OPEN, describe=lambda: "open"
            )
        text = f"deferred to the recreate window at {held[0].isoformat()}: {held[1]}"
        return SimpleNamespace(verdict=verdicts.DEFER, describe=lambda: text)


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


# --- OMN-25912: the container observer and the audit trigger (the deploy agent's idle tick) ---

_OBSERVER_MODULES = (
    "enums/enum_dev_recreate_cause.py",
    "enums/enum_dev_recreate_violation_kind.py",
    "models/model_dev_container_reading.py",
    "models/model_dev_container_recreated.py",
    "models/model_dev_container_recreate_request.py",
    "models/model_dev_container_recreate_result.py",
    "models/model_dev_container_observation.py",
    "models/model_dev_recreate_policy.py",
    "models/model_dev_recreate_announcement.py",
    "models/model_dev_recreate_violation_row.py",
    "models/model_dev_recreate_violation.py",
    "models/model_dev_recreate_audit_result.py",
    "models/model_dev_recreate_audit_request.py",
    "protocols/protocol_dev_roll_events.py",
    "handlers/dev_container_recreate/handler_dev_container_recreate.py",
    "models/model_dev_build_health.py",
    "models/model_dev_build_health_request.py",
    "handlers/dev_build_health/handler_dev_build_health.py",
)
_WINDOW_MODULES = (
    "enums/enum_dev_recreate_verdict.py",
    "enums/enum_dev_recreate_reason.py",
    "models/model_dev_recreate_request.py",
    "models/model_dev_recreate_decision.py",
    "handlers/dev_recreate_window/handler_dev_recreate_window.py",
)
_AUDIT_MODULE = "nodes/node_dev_roll_events_effect/dev_recreate_audit.py"
_DOCKER_NEVER_STARTED_YEAR = 2000


def _module(relative: str) -> ModuleType:
    return sys.modules[
        "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
    ]


def _load_observer() -> None:
    """Load the observer's models, handlers and the reader by file path, in order."""
    for relative in (*_OBSERVER_MODULES[:6], *_WINDOW_MODULES[:2]):
        events._load_sibling(relative)
    for relative in (*_OBSERVER_MODULES[6:], *_WINDOW_MODULES[2:], _AUDIT_MODULE):
        events._load_sibling(relative)


def _settings(repo_root: Path) -> dict[str, JsonValue]:
    settings = declaration_at(repo_root)["settings"]
    if not isinstance(settings, dict):
        raise RuntimeError("the dev roll events contract declares no settings")
    return settings


def _setting_int(settings: Mapping[str, JsonValue], name: str) -> int:
    value = settings.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"the dev roll events contract declares no {name}")
    return value


def _setting_str(settings: Mapping[str, JsonValue], name: str) -> str:
    value = settings.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"the dev roll events contract declares no {name}")
    return value


#: The image label that names the omnibase_infra commit an image was built from.
_REVISION_LABEL = "org.opencontainers.image.revision"


def _image_revision(
    image: str, run: Callable[..., subprocess.CompletedProcess[str]]
) -> str:
    """The image's revision label, or empty when docker cannot say (OMN-26703)."""
    try:
        answer = run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                '{{index .Config.Labels "' + _REVISION_LABEL + '"}}',
                image,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    text = str(answer.stdout).strip()
    return (
        text if answer.returncode == 0 and re.fullmatch(r"[0-9a-f]{7,40}", text) else ""
    )


def _health_reading(state: Mapping[str, Any]) -> dict[str, Any]:
    """Docker's health status and the end of the first passing check in its log (OMN-26703)."""
    health = state.get("Health")
    if not isinstance(health, dict):
        return {}
    sentinel = datetime(1, 1, 1, tzinfo=UTC)
    first: datetime | None = None
    for entry in health.get("Log") or []:
        if isinstance(entry, dict) and entry.get("ExitCode") == 0:
            stamp = docker_started_at(entry.get("End"), fallback=sentinel)
            if stamp.year >= _DOCKER_NEVER_STARTED_YEAR:
                first = stamp
                break
    return {"health_status": str(health.get("Status", "")), "first_healthy_at": first}


def revision_order(
    repo_root: Path,
    previous: str,
    current: str,
    timeout: float,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """How ``current`` stands against ``previous`` in the checkout's history (OMN-26703).

    ``first`` with no previous revision, ``same``, ``forward`` (previous is an ancestor),
    ``back`` (current is an ancestor of previous), ``unrelated`` (neither) and ``unread``
    when a revision is unknown to the checkout or git fails: never a guess.
    """
    if not previous:
        return "first"
    if not current:
        return "unread"
    if previous == current:
        return "same"

    def ancestor(older: str, newer: str) -> bool | None:
        try:
            answer = run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "merge-base",
                    "--is-ancestor",
                    older,
                    newer,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return {0: True, 1: False}.get(answer.returncode)

    forward = ancestor(previous, current)
    back = ancestor(current, previous)
    if forward is None or back is None:
        return "unread"
    return "forward" if forward else "back" if back else "unrelated"


def read_container(
    compose_project: str,
    service: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ModelDevContainerReading | None:
    """The one container of ``service`` in the compose project as docker inspect reports it.

    ``None`` when docker cannot be read, the service has no container or more than one, or
    the container never started: an unreadable container is no reading, never a guess.
    """
    try:
        listed = run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={compose_project}",
                "--filter",
                f"label=com.docker.compose.service={service}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ids = listed.stdout.split()
        if listed.returncode != 0 or len(ids) != 1:
            return None
        inspected = run(
            ["docker", "inspect", ids[0]],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspected.returncode != 0:
            return None
        raw = json.loads(inspected.stdout)[0]
        state = raw["State"]
        sentinel = datetime(1, 1, 1, tzinfo=UTC)
        started = docker_started_at(state.get("StartedAt"), fallback=sentinel)
        if started.year < _DOCKER_NEVER_STARTED_YEAR:
            return None
        exit_code = state.get("ExitCode")
        _load_observer()
        reading = sys.modules["omnibase_infra.models.model_dev_container_reading"]
        return reading.ModelDevContainerReading(
            container=str(raw["Name"]).lstrip("/"),
            container_digest=str(raw["Id"]),
            started_at=started,
            restart_count=int(raw.get("RestartCount", 0)),
            exit_code=exit_code if isinstance(exit_code, int) else None,
            image_digest=str(raw.get("Image", "")),
            image_ref=str(raw.get("Config", {}).get("Image", "")),
            image_revision=_image_revision(str(raw.get("Image", "")), run),
            **_health_reading(state),
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        logger.exception(
            "dev container could not be read friction_type=dev_container_unreadable"
        )
        return None


def read_journal(
    journal_dir: Path, since: datetime, until: datetime
) -> list[tuple[str, dict[str, JsonValue]]]:
    """The journaled events from ``since`` on, oldest first, as (role, payload) pairs.

    The files are one a day. A file whose day ended more than a day before ``since`` is
    removed; a line that does not parse is skipped.
    """
    if not journal_dir.is_dir():
        return []
    rows: list[tuple[datetime, str, dict[str, JsonValue]]] = []
    for path in sorted(journal_dir.glob("*.jsonl")):
        try:
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
        except ValueError:
            continue
        if day + timedelta(days=2) < since:
            path.unlink(missing_ok=True)
            continue
        if day > until:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                at = datetime.fromisoformat(row["journaled_at"])
                role, payload = str(row["role"]), row["payload"]
            except (ValueError, KeyError, TypeError):
                continue
            if at >= since and isinstance(payload, dict):
                rows.append((at, role, payload))
    rows.sort(key=lambda row: row[0])
    return [(role, payload) for _, role, payload in rows]


def _journal_size(journal_dir: Path) -> int:
    if not journal_dir.is_dir():
        return 0
    return sum(path.stat().st_size for path in journal_dir.glob("*.jsonl"))


def _atomic_write(path: Path, data: dict[str, JsonValue]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class NewViolationsOnly:
    """The reader's publish seam: a violation already published is not published again."""

    def __init__(self, publisher: ProtocolDevRollEvents, seen: set[str]) -> None:
        self._publisher = publisher
        self._seen = seen
        self.settled: set[str] = set()
        self.refused = 0

    def recreate_violation(self, violation: BaseModel) -> bool:
        event_id = str(violation.model_dump(mode="json")["event_id"])
        if event_id in self._seen:
            self.settled.add(event_id)
            return True
        accepted = bool(self._publisher.recreate_violation(violation))
        if accepted:
            self.settled.add(event_id)
        else:
            self.refused += 1
        return accepted


_BUILD_HEALTH = "handlers/dev_build_health/handler_dev_build_health.py"


def _publish_build_health(
    reading: ModelDevContainerReading,
    state: dict[str, Any],
    publisher: ProtocolDevRollEvents,
    settings: Mapping[str, JsonValue],
    repo_root: Path,
    run: Callable[..., subprocess.CompletedProcess[str]],
    handler_module: ModuleType,
) -> None:
    """Publish the container start's health clock and revision order once its health settles (OMN-26703).

    The judged start, the revision it ran and the start of the no-rollback clock are kept in
    ``state["build_health"]``; a start the bus did not accept is not marked, so it is retried.
    """
    judged: dict[str, Any] = state.get("build_health") or {}
    key = f"{reading.container_digest}:{reading.started_at.isoformat()}"
    if judged.get("judged") == key:
        return
    previous_revision = str(judged.get("image_revision", ""))
    since = judged.get("no_rollback_since")
    request_module = _module("models/model_dev_build_health_request.py")
    health_handler = handler_module.HandlerDevBuildHealth()
    event = health_handler.handle(
        request_module.ModelDevBuildHealthRequest(
            reading=reading,
            previous_image_revision=previous_revision,
            revision_order=revision_order(
                repo_root,
                previous_revision,
                reading.image_revision,
                float(_setting_int(settings, "build_health_git_timeout_seconds")),
                run,
            ),
            previous_no_rollback_since=datetime.fromisoformat(since) if since else None,
            limit_seconds=_setting_int(settings, "build_health_limit_seconds"),
        )
    )
    if event is None:
        return
    if not publisher.build_health(event):
        logger.error(
            "dev build health %s was not accepted by the bus %s",
            event.event_id,
            events.FRICTION,
        )
        return
    state["build_health"] = {
        "judged": key,
        "image_revision": reading.image_revision or previous_revision,
        "no_rollback_since": event.no_rollback_since.isoformat(),
    }


def observe_dev_container(
    state_dir: Path,
    compose_project: str,
    repo_root: Path,
    *,
    publisher: ProtocolDevRollEvents | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> object | None:
    """One tick of the dev container observer; returns the audit result when the reader ran.

    Reads the container, publishes a ``dev_container_recreated`` event when it started since
    the previous tick, and runs the reader (``audit_and_publish``) over the events the agent
    journaled in the declared read window whenever the journal changed since the last run.
    Nothing here may raise into the agent loop; the caller swallows and logs.
    """
    settings = _settings(repo_root)
    state_path = state_dir / _setting_str(settings, "container_reading_state_file")
    journal_dir = journal_dir_at(repo_root, state_dir)
    if journal_dir is None:
        raise RuntimeError("the dev container observer needs a state directory")
    _load_observer()
    reading_module = _module("models/model_dev_container_reading.py")
    request_module = _module("models/model_dev_container_recreate_request.py")
    started_module = sys.modules["omnibase_infra.models.model_dev_runtime_started"]
    manifest_module = sys.modules["omnibase_infra.models.model_dev_roll_manifest"]
    announcement_module = _module("models/model_dev_recreate_announcement.py")
    observation_module = _module("models/model_dev_container_observation.py")
    audit_request_module = _module("models/model_dev_recreate_audit_request.py")
    policy_module = _module("models/model_dev_recreate_policy.py")
    handler_module = _module(
        "handlers/dev_container_recreate/handler_dev_container_recreate.py"
    )
    audit_module = _module(_AUDIT_MODULE)

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    publisher = publisher or build_events_at(repo_root, state_dir)
    now = clock()
    since = now - timedelta(seconds=_setting_int(settings, "audit_read_window_seconds"))
    tolerance = _setting_int(settings, "audit_start_match_tolerance_seconds")

    reading = read_container(
        compose_project, _setting_str(settings, "observed_service"), run=run
    )
    previous = (
        reading_module.ModelDevContainerReading.model_validate(state["reading"])
        if "reading" in state
        else None
    )
    if reading is not None:
        journaled = read_journal(journal_dir, since, now)
        recreate_handler = handler_module.HandlerDevContainerRecreate()
        result = recreate_handler.handle(
            request_module.ModelDevContainerRecreateRequest(
                reading=reading,
                previous=previous,
                starts=tuple(
                    started_module.ModelDevRuntimeStarted.model_validate(p)
                    for r, p in journaled
                    if r == "dev_runtime_started"
                ),
                manifests=tuple(
                    manifest_module.ModelDevRollManifest.model_validate(p)
                    for r, p in journaled
                    if r == "dev_roll_manifest"
                ),
                start_match_tolerance_seconds=tolerance,
            )
        )
        # A start the bus did not accept is retried: the reading that would hide it is
        # not saved until the event is out.
        if result.event is None or publisher.container_recreated(result.event):
            state["reading"] = reading.model_dump(mode="json")
        else:
            logger.error(
                "dev container start %s was not accepted by the bus %s",
                result.event.event_id,
                events.FRICTION,
            )

    if reading is not None:
        _publish_build_health(
            reading, state, publisher, settings, repo_root, run, _module(_BUILD_HEALTH)
        )

    journal_size = _journal_size(journal_dir)
    audited: object | None = None
    if journal_size != state.get("journal_size"):
        entries = read_journal(journal_dir, since, now)

        def of(role: str) -> list[dict[str, JsonValue]]:
            return [p for r, p in entries if r == role]

        policy = policy_module.ModelDevRecreatePolicy.from_env(policy_env_at(repo_root))
        request = audit_request_module.ModelDevRecreateAuditRequest(
            policy=policy,
            announcement_lookback_seconds=_setting_int(
                settings, "audit_announcement_lookback_seconds"
            ),
            start_match_tolerance_seconds=tolerance,
            starts=tuple(
                started_module.ModelDevRuntimeStarted.model_validate(p)
                for p in of("dev_runtime_started")
            ),
            manifests=tuple(
                manifest_module.ModelDevRollManifest.model_validate(p)
                for p in of("dev_roll_manifest")
            ),
            announcements=tuple(
                announcement_module.ModelDevRecreateAnnouncement.model_validate(p)
                for p in of("dev_recreate_announced")
            ),
            container_observations=tuple(
                observation_module.ModelDevContainerObservation.model_validate(
                    {
                        "container": p["container"],
                        "started_at": p["started_at"],
                        "restart_count": p["restart_count"],
                        "exit_code": p.get("exit_code"),
                        "cause": p["cause"],
                        "same_build": p["same_build"],
                    }
                )
                for p in of("dev_container_recreated")
            ),
        )
        seen = {str(v) for v in state.get("published_violations", [])}
        seam = NewViolationsOnly(publisher, seen)
        audited = audit_module.audit_and_publish(
            audit_module.DevRecreateAuditReader(), request, seam
        )
        state["published_violations"] = sorted(seam.settled)
        # A violation the bus refused is read again at the next tick.
        state["journal_size"] = -1 if seam.refused else journal_size

    state_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(state_path, state)
    return audited


_observed_last: list[float] = []


def observe_if_due(
    state_dir: Path,
    compose_project: str,
    *,
    force: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """The deploy agent's idle-tick entry (OMN-25912): observe at most once per declared interval.

    ``audit_trigger: agent_tick`` in the contract. The tick is stamped before the read, so a
    failing read logs once per interval. Never raises into the agent loop. Returns whether an
    observation ran.
    """
    repo_root = Path(__file__).resolve().parents[4]
    try:
        interval = _setting_int(
            _settings(repo_root), "container_observation_interval_seconds"
        )
        tick = monotonic()
        if not force and _observed_last and tick - _observed_last[0] < interval:
            return False
        _observed_last[:] = [tick]
        observe_dev_container(state_dir, compose_project, repo_root)
    except Exception:
        logger.exception(
            "dev container observation failed friction_type=dev_event_publish_failed"
        )
        return False
    return True
