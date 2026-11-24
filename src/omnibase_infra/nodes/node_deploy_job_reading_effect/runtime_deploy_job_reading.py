# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root of the deploy job reading effect (OMN-24617, OMN-26431).

The deploy agent loads this by file path (``load_dev_boot_guard``). It loads the handler
and the two models the handler imports from the same checkout, reads the node's
``settings`` from the contract of the agent's own checkout, binds the handler to the process environment and
hands it the agent's settled job record. The handler and the contract hold the names, the
bounds and the append route; nothing here repeats them.

It also carries the build decision entries (OMN-26431): ``record_idle``, ``record_queued``,
``record_deferred`` and ``record_command`` write one row per dev merge sha decision, from the
agent's idle converge and its command execution. Each decision is written once: its key
(repository, sha, result, reason) is held in the agent's state directory, marked before the
append, so a refused append is logged and counted and never retried.
"""

from __future__ import annotations

import functools
import http.client
import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, JsonValue, ValidationError

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_failed_head_gate import (
        FailedHeadGate,
    )
    from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_post_failure_converge import (
        PostFailureConverge,
    )
    from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_untagged_head_gate import (
        IdleBuildGate,
        UntaggedHeadGate,
    )
    from omnibase_infra.protocols.protocol_command import ProtocolCommand
    from omnibase_infra.protocols.protocol_post_failure_host import (
        ProtocolPostFailureHost,
    )

logger = logging.getLogger(__name__)

NODE = "src/omnibase_infra/nodes/node_deploy_job_reading_effect"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
#: The modules the handler imports, then the handler, in import order.
_CHECKOUT_MODULES = (
    "models/model_deploy_job_reading_request.py",
    "models/model_deploy_job_reading_result.py",
    "handlers/handler_deploy_job_reading.py",
    "models/model_dev_merge_build_decision_request.py",
    "handlers/handler_dev_merge_build_decision.py",
)


def _load(relative: str) -> ModuleType:
    """One module of this checkout, loaded under its dotted name by file path, once."""
    name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PACKAGE_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the deploy job reading module: {relative}")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


(
    request_module,
    result_module,
    handler_module,
    decision_request_module,
    decision_handler_module,
) = (_load(relative) for relative in _CHECKOUT_MODULES)

_POST_FAILURE_MODULES = (
    "enums/enum_post_failure_verdict.py",
    "enums/enum_job_status.py",
    "models/model_job_command.py",
    "models/model_job_lineage.py",
    "models/model_job_record.py",
    "models/model_failed_dev_job.py",
    "models/model_post_failure_inputs.py",
    "models/model_post_failure_decision.py",
    "protocols/protocol_command.py",
    "protocols/protocol_dev_tag.py",
    "protocols/protocol_job_store.py",
    "protocols/protocol_lane_flags.py",
    "protocols/protocol_post_failure_host.py",
    "nodes/node_deploy_job_reading_effect/runtime_post_failure_converge.py",
)


def post_failure_converge(
    host: ProtocolPostFailureHost, converge_command: Callable[[], ProtocolCommand]
) -> PostFailureConverge:
    """The agent's post-failure converge rule (OMN-26429), loaded from this checkout once.

    It reads the same job records this node turns into readings, and shares nothing else.
    """
    *_, rule = (_load(relative) for relative in _POST_FAILURE_MODULES)
    converge: PostFailureConverge = rule.PostFailureConverge(host, converge_command)
    return converge


_FAILED_HEAD_MODULES = (
    *_POST_FAILURE_MODULES,
    "enums/enum_failed_head_verdict_class.py",
    "models/model_failed_head_refusal.py",
    "nodes/node_deploy_job_reading_effect/runtime_failed_head_gate.py",
)


def failed_head_gate(host: ProtocolPostFailureHost) -> FailedHeadGate:
    """The agent's failed-head rule for its idle builds (OMN-26493), loaded from this checkout once.

    It reads the same job records as the post-failure rule and writes its one row per head
    through this module's build decision entry.
    """
    *_, rule = (_load(relative) for relative in _FAILED_HEAD_MODULES)
    gate: FailedHeadGate = rule.FailedHeadGate(host, record_failed_head)
    return gate


def contract_path(execution: ModuleType) -> Path:
    """The node's contract in the checkout the deploy agent's own module was loaded from."""
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError(
            "the deploy job reading needs the deploy agent source location"
        )
    return Path(module_file).resolve().parents[3] / NODE / "contract.yaml"


def load_settings(execution: ModuleType) -> dict[str, JsonValue]:
    """The node contract's ``settings`` block: names, bounds and the append route variables."""
    return _contract_settings(contract_path(execution))


def _contract_settings(path: Path) -> dict[str, JsonValue]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = contract["settings"]
    if not isinstance(settings, dict):
        raise RuntimeError("the deploy job reading contract declares no settings")
    loaded: dict[str, JsonValue] = settings
    return loaded


def publish(job: BaseModel, execution: ModuleType) -> BaseModel | None:
    """Write the reading row of the deploy agent's settled ``job`` record; never raises.

    ``job`` is read through its JSON form so this module holds no import of the agent;
    ``execution`` is an agent module, which locates the checkout the contract is read from.
    Returns the handler's result, or ``None`` when the record could not be read.
    """
    try:
        record = json.loads(job.model_dump_json())
        request = request_module.ModelDeployJobReadingRequest(
            correlation_id=record["correlation_id"],
            runtime_lane=record["command"]["runtime_lane"],
            scope=record["command"]["scope"],
            job_outcome=record["status"],
            phase_results=record["phase_results"],
            not_started=record.get("not_started") or "",
            first_error=(record.get("errors") or [""])[0],
            completed_at=record.get("completed_at"),
            published_at=datetime.now(UTC),
        )
        handler = handler_module.HandlerDeployJobReading(
            settings=load_settings(execution), environ=os.environ
        )
        result: BaseModel = handler.handle(request)
        return result
    except (KeyError, TypeError, ValueError, ValidationError, RuntimeError, OSError):
        logger.exception("job reading skipped: the job record could not be read")
        return None


_pin_last_check: list[float] = []
_pin_settings: list[dict[str, JsonValue]] = []
_PIN_HANDLER = "omnibase_infra.handlers.handler_dev_lane_code_ref_pin"
#: Until the contract has been read once, the throttle holds the check this long between tries.
_PIN_RETRY_SECONDS = 300.0


def _idle_settings() -> dict[str, JsonValue]:
    """The node contract's settings of the agent's own checkout, read once per process."""
    if not _pin_settings:
        _pin_settings[:] = [
            _contract_settings(_PACKAGE_ROOT.parents[1] / NODE / "contract.yaml")
        ]
    return _pin_settings[0]


def check_pin(*, force: bool = False) -> BaseModel | None:
    """Alarm when the agent's live code-ref pin is inside the window before its expiry; never raises.

    Called from the agent's idle tick, so it runs on a schedule with no job. The pin record is
    read from the agent's own state directory (the pin handler's ``DEPLOY_AGENT_STATE_DIR``),
    through the pin handler the agent loaded before this module. The contract is read and
    parsed once per process; the check is throttled to its
    ``pin_alarm_check_interval_seconds`` (``force`` skips the throttle), and the throttle is
    stamped before any read, so a failing read logs once per interval, not once per tick.
    Returns the handler's result, or ``None`` when throttled, when no pin is live or on an
    unreadable record.

    It is the one entry the idle tick calls for this node, so it also runs the callers the
    contract's ``operation_callers`` declares (OMN-26524, OMN-26316): ``check_routing``,
    ``check_routed_commands`` and ``check_overdue_dev_merges``. Each is throttled on its own
    interval, never raises and leaves this function's result as it was.
    """
    check_routing(force=force)
    check_routed_commands(force=force)
    check_overdue_dev_merges(force=force)
    try:
        tick = time.monotonic()
        interval = (
            _pin_settings[0]["pin_alarm_check_interval_seconds"]
            if _pin_settings
            else _PIN_RETRY_SECONDS
        )
        if (
            not force
            and _pin_last_check
            and isinstance(interval, int | float)
            and tick - _pin_last_check[0] < interval
        ):
            return None
        _pin_last_check[:] = [tick]
        if not _pin_settings:
            _pin_settings[:] = [
                _contract_settings(_PACKAGE_ROOT.parents[1] / NODE / "contract.yaml")
            ]
        settings = _pin_settings[0]
        pin_handler = sys.modules[_PIN_HANDLER]
        handler = handler_module.HandlerDeployJobReading(
            settings=settings, environ=os.environ
        )
        state_dir = handler.pin_state_dir()
        now = datetime.now(UTC)
        pin = pin_handler.read_active_pin(state_dir, now) if state_dir else None
        if pin is None:
            return None
        result: BaseModel = handler.handle_pin(
            branch=pin.branch,
            pr=pin.pr,
            ticket=pin.ticket,
            registered_by=pin.registered_by,
            expires_at=pin.expires_at,
            now=now,
        )
        return result
    except (KeyError, TypeError, ValueError, ValidationError, RuntimeError, OSError):
        logger.exception(
            "pin alarm check skipped: the pin or contract could not be read"
        )
        return None


# --- OMN-26524: the callers of the two routing operations -------------------------------

_ROUTING_MODULES = (
    "models/model_deploy_lane_instance.py",
    "models/model_deploy_lane_route.py",
    "models/model_deploy_lane_routing_table.py",
    "models/model_routing_key_refusal.py",
    "models/model_routing_key_check_request.py",
    "models/model_routing_key_check_result.py",
    "handlers/handler_routing_key_check.py",
)
_ALARM_MODULES = (
    "enums/enum_job_status.py",
    "models/model_job_command.py",
    "models/model_job_lineage.py",
    "models/model_job_record.py",
    "models/model_routed_command.py",
    "models/model_routed_command_alarm_row.py",
    "models/model_routed_command_alarm_request.py",
    "models/model_routed_command_alarm_result.py",
    "handlers/handler_routed_command_alarm.py",
)
_ROUTING_TABLE = "config/deploy_lane_routing.yaml"
_routing_last: list[float] = []
_alarm_last: list[float] = []


def _due(
    last: list[float], settings: dict[str, JsonValue], key: str, force: bool
) -> bool:
    """Whether the interval the contract's ``key`` declares has passed; stamps the tick."""
    interval = settings[key]
    tick = time.monotonic()
    if (
        not force
        and last
        and isinstance(interval, int | float)
        and tick - last[0] < interval
    ):
        return False
    last[:] = [tick]
    return True


def _read_key_check(url: str, timeout: float) -> str:
    """The ``hmac_key_check`` of the agent health answer at ``url``; empty when unread.

    Only an ``http`` URL is read. A 503 health answer still carries the agent's key check.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        return ""
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("GET", parts.path or "/")
        body = json.loads(connection.getresponse().read())
    except (http.client.HTTPException, OSError, ValueError):
        return ""
    finally:
        connection.close()
    value = body.get("hmac_key_check") if isinstance(body, dict) else None
    return value if isinstance(value, str) else ""


def check_routing(*, force: bool = False) -> BaseModel | None:
    """Run ``check_routing_key_checks`` once: the routing table against the agents' key checks.

    Called from the idle tick (through ``check_pin``), so it runs with no job. The table is the
    one shipped with this agent's checkout. The reference key check is this agent's own: the
    sender signs each command once, so every available instance's agent must hold the key
    this one holds. The answers are this agent's own for its own instance and, for each
    other instance, a read of the health URL the contract's ``instance_health_urls`` declares
    for it; an instance with no URL or no answer is unread. Writes one run line per run
    (``routing key check outcome=...``); a mismatch is logged at WARNING with both values.
    Throttled to ``routing_check_interval_seconds`` (``force`` skips the throttle); never
    raises; returns the handler's result, or ``None`` when throttled or not runnable.
    """
    try:
        settings = _idle_settings()
        if not _due(_routing_last, settings, "routing_check_interval_seconds", force):
            return None
        *_, handler_mod = (_load(relative) for relative in _ROUTING_MODULES)
        request_mod, table_mod = (
            sys.modules["omnibase_infra.models.model_routing_key_check_request"],
            sys.modules["omnibase_infra.models.model_deploy_lane_routing_table"],
        )
        agent_routing = importlib.import_module("deploy_agent.routing")
        own_check = importlib.import_module("deploy_agent.auth").hmac_key_check()
        if not own_check:
            logger.warning(
                "routing key check skipped: this agent holds no signing key "
                "friction_type=routing_key_check_skipped"
            )
            return None
        table = table_mod.ModelDeployLaneRoutingTable.model_validate(
            yaml.safe_load(
                (_PACKAGE_ROOT.parents[1] / _ROUTING_TABLE).read_text(encoding="utf-8")
            )
        )
        own = agent_routing.resolve_instance(agent_routing.load_routing_table()).name
        urls = settings["instance_health_urls"]
        timeout = settings["routing_health_timeout_seconds"]
        if not isinstance(urls, dict) or not isinstance(timeout, int | float):
            raise RuntimeError("the deploy job reading contract's routing settings")
        answers = {own: own_check}
        for name, spec in table.instances.items():
            url = urls.get(name)
            if name != own and spec.available and isinstance(url, str) and url:
                answers[name] = _read_key_check(url, float(timeout))
        handler = handler_mod.HandlerRoutingKeyCheck()
        result: BaseModel = handler.handle(
            request_mod.ModelRoutingKeyCheckRequest(
                table=table, runtime_key_check=own_check, instance_key_checks=answers
            )
        )
        mismatches = getattr(result, "mismatches", ())
        logger.log(
            logging.WARNING if mismatches else logging.INFO,
            "routing key check outcome=%s instance=%s mismatches=%s",
            getattr(result, "outcome", ""),
            own,
            ",".join(
                f"{m.instance}:{m.reason}:{m.instance_key_check or '-'}"
                f"!={m.runtime_key_check}"
                for m in mismatches
            )
            or "none",
        )
        return result
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RuntimeError,
        OSError,
        ImportError,
        yaml.YAMLError,
    ):
        logger.exception("routing key check skipped: the table or contract was unread")
        return None


def _routed_commands(
    settings: dict[str, JsonValue], state_dir: Path, instance: str
) -> list[object]:
    """This instance's routed commands from the contract's ``routed_commands_file``; none when absent."""
    name = settings["routed_commands_file"]
    if not isinstance(name, str):
        raise RuntimeError("the deploy job reading contract's routed_commands_file")
    path = state_dir / name
    if not path.is_file():
        return []
    command_mod = sys.modules["omnibase_infra.models.model_routed_command"]
    commands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            command = command_mod.ModelRoutedCommand.model_validate_json(line)
        except ValidationError:
            logger.warning("routed commands: a line of %s is not a command", name)
            continue
        if command.instance == instance:
            commands.append(command)
    return commands


def _job_records(state_dir: Path) -> list[object]:
    """The agent's own job records (``<correlation_id>.json`` in its state directory)."""
    record_mod = sys.modules["omnibase_infra.models.model_job_record"]
    records = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            records.append(
                record_mod.ModelJobRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            )
        except (ValidationError, OSError):
            continue
    return records


def check_routed_commands(*, force: bool = False) -> BaseModel | None:
    """Run ``alarm_routed_commands_without_job_record`` once over this instance's job records.

    Called from the idle tick (through ``check_pin``), so it runs with no job. The commands
    are this instance's lines of the contract's ``routed_commands_file`` and the job records
    are the agent's own state directory's; a record on another instance's agent is not read
    here, so only commands routed to this instance are polled. A command with no record after
    ``routed_command_no_job_window_seconds`` writes its alarm row through the node's append.
    Writes one run line per run (``routed command alarm polled=...``). Throttled to
    ``routed_command_alarm_interval_seconds`` (``force`` skips the throttle); never raises;
    returns the handler's result, or ``None`` when throttled or not runnable.
    """
    try:
        settings = _idle_settings()
        if not _due(
            _alarm_last, settings, "routed_command_alarm_interval_seconds", force
        ):
            return None
        for relative in _ALARM_MODULES:
            _load(relative)
        alarm_mod = sys.modules["omnibase_infra.handlers.handler_routed_command_alarm"]
        request_mod = sys.modules[
            "omnibase_infra.models.model_routed_command_alarm_request"
        ]
        agent_routing = importlib.import_module("deploy_agent.routing")
        own = agent_routing.resolve_instance(agent_routing.load_routing_table()).name
        state_dir = Path(_agent().STATE_DIR)
        commands = _routed_commands(settings, state_dir, own)
        handler = alarm_mod.HandlerRoutedCommandAlarm(
            settings=settings,
            append=functools.partial(
                handler_module.append_ledger_row, settings, os.environ
            ),
        )
        result: BaseModel = handler.handle(
            request_mod.ModelRoutedCommandAlarmRequest(
                commands=tuple(commands),
                job_records={own: tuple(_job_records(state_dir))},
                now=datetime.now(UTC),
            )
        )
        alarms = getattr(result, "alarms", ())
        logger.log(
            logging.WARNING if alarms else logging.INFO,
            "routed command alarm polled=%d instance=%s alarms=%d waiting=%d",
            len(commands),
            own,
            len(alarms),
            len(getattr(result, "waiting", ())),
        )
        return result
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RuntimeError,
        OSError,
        ImportError,
        yaml.YAMLError,
    ):
        logger.exception(
            "routed command alarm skipped: the commands or contract were unread"
        )
        return None


# --- OMN-26316: the caller of the overdue dev merge build decision ------------------------

_OVERDUE_MODULES = (
    "models/model_overdue_dev_merge_job.py",
    "models/model_overdue_dev_merge_build_request.py",
    "models/model_overdue_dev_merge_build_decision.py",
    "handlers/handler_dev_merge_build_overdue.py",
)
_overdue_last: list[float] = []
#: The branch the agent's checkout tracks; no branch name is spelled here.
_TRACKED_HEAD = "@{upstream}"


def _seconds(settings: dict[str, JsonValue], key: str) -> int:
    value = settings[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"the deploy job reading contract's {key} is not a bound")
    return value


def _overdue_jobs(state_dir: Path) -> list[object]:
    """The agent's job records as the overdue decision reads them; a file that is none is skipped."""
    job_mod = sys.modules["omnibase_infra.models.model_overdue_dev_merge_job"]
    jobs: list[object] = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            jobs.append(
                job_mod.ModelOverdueDevMergeJob(
                    job_id=data["correlation_id"],
                    runtime_lane=data["command"]["runtime_lane"],
                    accepted_at=data["accepted_at"],
                    status=data["status"],
                    git_ref=data["command"]["git_ref"],
                )
            )
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            continue
    return jobs


def _dev_merges(
    execution: ModuleType, now: datetime, lookback_seconds: int, timeout: float
) -> list[tuple[str, datetime, str]]:
    """The checkout's recent first-parent dev commits as (sha, commit time, ``RUNTIME`` or ``NON_RUNTIME``).

    The commit time of a merge-queue squash is its queue entry, which can precede the merge
    by the queue's wait; a job accepted inside that wait reads as carrying a sha it could not
    have. An unread log is no merge and an unread diff is a runtime merge, never a guess.
    """
    clone = str(execution.REPO_DIR)
    since = (now - timedelta(seconds=lookback_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    log = subprocess.run(
        [
            "git",
            "-C",
            clone,
            "log",
            _TRACKED_HEAD,
            "--first-parent",
            f"--since={since}",
            "--max-count=100",
            "--format=%H %ct",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if log.returncode != 0:
        raise RuntimeError(f"the checkout's dev log was unread: {log.stderr.strip()}")
    predicate = _not_runtime(execution)
    merges: list[tuple[str, datetime, str]] = []
    for line in log.stdout.splitlines():
        sha, stamp = line.split()
        diff = subprocess.run(
            ["git", "-C", clone, "diff", "--name-only", f"{sha}^", sha],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        not_runtime = diff.returncode == 0 and all(
            predicate(path, "omnibase_infra") for path in diff.stdout.split()
        )
        merges.append(
            (
                sha,
                datetime.fromtimestamp(int(stamp), tz=UTC),
                "NON_RUNTIME" if not_runtime else "RUNTIME",
            )
        )
    return merges


def check_overdue_dev_merges(*, force: bool = False) -> tuple[BaseModel, ...] | None:
    """Run ``request_overdue_dev_merge_build`` once over the checkout's recent dev merges (OMN-26316).

    Called from the idle tick (through ``check_pin``), so it runs with no job. The merges are
    the tracked dev head's recent first-parent commits and the jobs are the agent's own
    records. It publishes nothing: one run line per run (``overdue dev merge builds ...``)
    and one warning per merge judged REQUEST, naming its request id, which is what the
    publish that follows will carry. Throttled to ``dev_merge_build_check_interval_seconds``
    (``force`` skips the throttle); never raises; returns the decisions, or ``None`` when
    throttled or not runnable.
    """
    try:
        settings = _idle_settings()
        if not _due(
            _overdue_last, settings, "dev_merge_build_check_interval_seconds", force
        ):
            return None
        for relative in _OVERDUE_MODULES:
            _load(relative)
        handler_mod = sys.modules[
            "omnibase_infra.handlers.handler_dev_merge_build_overdue"
        ]
        request_mod = sys.modules[
            "omnibase_infra.models.model_overdue_dev_merge_build_request"
        ]
        execution = _agent()
        now = datetime.now(UTC)
        window = _seconds(settings, "dev_merge_build_window_seconds")
        jobs = tuple(_overdue_jobs(Path(execution.STATE_DIR)))
        handler = handler_mod.HandlerDevMergeBuildOverdue()
        decisions: list[BaseModel] = []
        for sha, merged_at, runtime_class in _dev_merges(
            execution,
            now,
            _seconds(settings, "dev_merge_build_lookback_seconds"),
            float(_seconds(settings, "build_decision_git_diff_timeout_seconds")),
        ):
            decisions.append(
                handler.handle(
                    request_mod.ModelOverdueDevMergeBuildRequest(
                        repository="omnibase_infra",
                        merge_sha=sha,
                        merged_at=merged_at,
                        runtime_class=runtime_class,
                        now=now,
                        window_seconds=window,
                        jobs=jobs,
                    )
                )
            )
        overdue = [d for d in decisions if getattr(d, "outcome", "") == "REQUEST"]
        logger.log(
            logging.WARNING if overdue else logging.INFO,
            "overdue dev merge builds merges=%d satisfied=%d waiting=%d requests=%d",
            len(decisions),
            sum(getattr(d, "outcome", "") == "SATISFIED" for d in decisions),
            sum(getattr(d, "outcome", "") == "WAIT" for d in decisions),
            len(overdue),
        )
        for decision in overdue:
            logger.warning(
                "overdue dev merge build sha=%s due_at=%s request_id=%s",
                getattr(decision, "merge_sha", ""),
                getattr(decision, "due_at", ""),
                getattr(decision, "request_id", ""),
            )
        return tuple(decisions)
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RuntimeError,
        OSError,
        ImportError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ):
        logger.exception(
            "overdue dev merge builds skipped: the merges, jobs or contract were unread"
        )
        return None


# --- OMN-26431: one ledger row per dev merge sha decision --------------------------------

#: The agent's own decisions about a sha: the idle converge verdicts that state a result.
#: ``up_to_date``, ``already_attempted`` and the unreadable heads state none (the first two
#: have a row already or need no build; an unread ref names no sha).
_IDLE_RESULTS = {
    "frozen": ("refused", "frozen"),
    "not_routed_elsewhere": ("refused", "not_routed_elsewhere"),
    "job_active": ("deferred", "busy"),
    "recently_active": ("deferred", "idle_gate"),
    "windows_unreadable": ("deferred", "probe_schedule_unreadable"),
    "probe_window": ("deferred", "probe_window"),
}
_BLOCKER = re.compile(r"^(?P<id>\w+) run at (?P<start>\d{4}-\d\d-\d\dT\d\d:\d\dZ)$")
_BLOCKER_TIME = "%Y-%m-%dT%H:%MZ"
_DECISION_STORE = "build-decision-rows.txt"
_DECISION_STORE_MAX = 500
_CLASSIFIER = "scripts/runtime_change_classifier.py"
_CLASSIFIER_MODULE = "_omnibase_infra_runtime_change_classifier"
_PUBLISHED = "PUBLISHED"
_DETECTED = "DETECTED_NOT_PUBLISHED"
#: The refs (head, running) each idle decision read, per repository, for the rows after it.
_refs: dict[str, tuple[str | None, str | None]] = {}
#: Repositories whose ``not_routed_elsewhere`` head was read once since their verdict last changed.
_not_routed_read: set[str] = set()
_append_failures: list[int] = [0]


def _field(model: object, name: str) -> str:
    """One field of the agent's model (or enum member) as text; empty when it has none."""
    value = getattr(model, name, None)
    value = getattr(value, "value", value)
    return "" if value is None else str(value)


def _clock(inputs: object) -> datetime:
    """The decision's own clock reading (the agent's injected clock), else the wall clock."""
    moment = getattr(inputs, "now", None)
    return moment if isinstance(moment, datetime) else datetime.now(UTC)


def append_failures() -> int:
    """How many build decision rows could not be written since this process started."""
    return _append_failures[0]


def record_adoption_revert(adopted_ref: str, previous_ref: str) -> str:
    """Append the alarm row of a restart storm that returned the agent to ``previous_ref``.

    Returns ``""`` when the append took the row, else why not; never raises (OMN-26494).
    """
    try:
        settings = load_settings(_agent())
        handler = handler_module.HandlerDeployJobReading(
            settings=settings, environ=os.environ
        )
        result = handler.handle_adoption_revert(
            adopted_ref=adopted_ref, previous_ref=previous_ref, now=datetime.now(UTC)
        )
        return str(getattr(result, "detail", ""))
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RuntimeError,
        OSError,
    ) as exc:
        return f"{type(exc).__name__}: {exc}"


def _agent() -> ModuleType:
    return importlib.import_module("deploy_agent.agent")


def _decision_keys(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split() if path.is_file() else []


def _mark(path: Path, key: str) -> bool:
    """Record ``key`` in the agent's state directory; ``False`` when it was already there."""
    keys = _decision_keys(path)
    if key in keys:
        return False
    keys.append(key)
    path.write_text("\n".join(keys[-_DECISION_STORE_MAX:]) + "\n", encoding="utf-8")
    return True


def _not_runtime(execution: ModuleType) -> Callable[[str, str], bool]:
    """The repository's ``is_not_runtime_path`` predicate, loaded by path from the agent's clone."""
    if _CLASSIFIER_MODULE not in sys.modules:
        path = Path(execution.AGENT_CLONE_ROOT) / _CLASSIFIER
        spec = importlib.util.spec_from_file_location(_CLASSIFIER_MODULE, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load the runtime change classifier at {path}")
        module = sys.modules[_CLASSIFIER_MODULE] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    predicate: Callable[[str, str], bool] = sys.modules[
        _CLASSIFIER_MODULE
    ].is_not_runtime_path
    return predicate


def _changes_no_runtime_path(
    execution: ModuleType,
    repository: str,
    head: str,
    running: str | None,
    timeout: float,
) -> bool:
    """Whether every path changed between the lane's revision and ``head`` is declared not-runtime.

    The half of the trigger's classifier that lists not-runtime paths: when it lists every
    changed path, the runtime set is empty whatever else the classifier adds. An unread
    clone, ref or diff is ``False``: the sha stays a runtime sha.
    """
    if running is None:
        return False
    clone = (
        Path(execution.REPO_DIR)
        if repository == "omnibase_infra"
        else execution._sibling_clone(repository)
    )
    if clone is None:
        return False
    done = subprocess.run(
        ["git", "-C", str(clone), "diff", "--name-only", running, head],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if done.returncode != 0:
        return False
    predicate = _not_runtime(execution)
    return all(predicate(path, repository) for path in done.stdout.split())


def _probe_earliest(
    execution: ModuleType, detail: str, now: datetime
) -> datetime | None:
    """The end of the guarded probe run the converge is held for, plus the agent's margin.

    The run is named in the verdict's detail (``C15 run at <UTC minute>``); its duration is
    the probe schedule's ``max_duration_minutes``, read through the agent's own loader.
    """
    found = _BLOCKER.match(detail)
    if found is None:
        return None
    idle = importlib.import_module(f"{execution.__package__}.idle_converge")
    start = datetime.strptime(found["start"], _BLOCKER_TIME).replace(tzinfo=UTC)
    for window in idle.load_probe_windows(execution.AGENT_CLONE_ROOT):
        if window.id == found["id"]:
            return (
                start
                + timedelta(minutes=int(window.max_duration_minutes))
                + idle.PROBE_MARGIN
            )
    return None


_UNTAGGED_HEAD_MODULES = (
    *_FAILED_HEAD_MODULES,
    "nodes/node_deploy_job_reading_effect/runtime_untagged_head_gate.py",
)


def untagged_head_gate(host: ProtocolPostFailureHost) -> UntaggedHeadGate:
    """The agent's untagged-head rule for its idle builds (OMN-25942), loaded from this checkout once.

    It reads origin's dev tags through the host's idle seam and writes its one row per head
    through this module's build decision entry.
    """
    *_, rule = (_load(relative) for relative in _UNTAGGED_HEAD_MODULES)
    gate: UntaggedHeadGate = rule.UntaggedHeadGate(host, record_untagged_head)
    return gate


def idle_build_gate(host: ProtocolPostFailureHost) -> IdleBuildGate:
    """The agent's refusals for its two idle builds: failed head (OMN-26493), then untagged head (OMN-25942)."""
    *_, rule = (_load(relative) for relative in _UNTAGGED_HEAD_MODULES)
    gate: IdleBuildGate = rule.IdleBuildGate(
        failed_head_gate(host), untagged_head_gate(host)
    )
    return gate


def _emit(
    repository: str,
    sha: str,
    runtime_class: str,
    now: datetime,
    *,
    result: str = "",
    reason: str = "",
    job: str = "",
    verdict_class: str = "",
    correlation: str = "",
    earliest_start: datetime | None = None,
) -> BaseModel | None:
    """Write one decision's row once; never raises and never retries a refused append."""
    try:
        execution = _agent()
        request = decision_request_module.ModelDevMergeBuildDecisionRequest(
            repository=repository,
            trigger_sha=sha,
            runtime_class=runtime_class,
            result=result,
            reason=reason,
            job=job,
            verdict_class=verdict_class,
            correlation=correlation,
            earliest_start=earliest_start,
            decided_at=now,
        )
        key = decision_handler_module.decision_key(request)
        if not _mark(Path(execution.STATE_DIR) / _DECISION_STORE, key):
            return None
        settings = load_settings(execution)
        handler = decision_handler_module.HandlerDevMergeBuildDecision(
            settings=settings,
            append=functools.partial(
                handler_module.append_ledger_row, settings, os.environ
            ),
        )
        written: BaseModel = handler.handle(request)
        if getattr(written, "outcome", "") != "appended":
            _append_failures[0] += 1
        return written
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ):
        _append_failures[0] += 1
        logger.exception(
            "build decision row for %s %s skipped friction_type=build_decision_row_skipped",
            repository,
            sha[:12],
        )
        return None


def _idle_class(
    execution: ModuleType, repository: str, head: str, running: str | None
) -> str:
    try:
        timeout = load_settings(execution)["build_decision_git_diff_timeout_seconds"]
        if isinstance(timeout, int | float) and _changes_no_runtime_path(
            execution, repository, head, running, float(timeout)
        ):
            return "NON_RUNTIME"
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        logger.exception("build decision class unread for %s %s", repository, head[:12])
    return _DETECTED


def record_idle(
    read_head: Callable[[str], str | None], inputs: object, decision: object
) -> BaseModel | None:
    """Write the row of one idle converge verdict that states a result; never raises.

    ``read_head`` is the agent's own dev head reader; ``inputs`` and ``decision`` are the idle
    converge decision's own models. Called on every
    idle check. A verdict that converges, or that states no result, writes nothing here (the
    converge's queued or deferred row is written where it is decided). ``not_routed_elsewhere``
    reads no head, so its head is read once per verdict change.
    """
    repository = _field(inputs, "repository")
    head = _field(inputs, "head_ref") or None
    running = _field(inputs, "running_ref") or None
    now = _clock(inputs)
    _refs[repository] = (head, running)
    verdict = _field(decision, "verdict")
    if verdict != "not_routed_elsewhere":
        _not_routed_read.discard(repository)
    states = _IDLE_RESULTS.get(verdict)
    if states is None:
        return None
    if verdict == "not_routed_elsewhere" and head is None:
        if repository in _not_routed_read:
            return None
        _not_routed_read.add(repository)
        head = read_head(repository)
    if head is None:
        return None
    result, reason = states
    execution = _agent()
    earliest = None
    try:
        if verdict == "probe_window":
            earliest = _probe_earliest(execution, _field(decision, "detail"), now)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
        logger.exception("build decision earliest start unread")
    runtime_class = (
        _PUBLISHED
        if verdict == "not_routed_elsewhere"
        else _idle_class(execution, repository, head, running)
    )
    return _emit(
        repository,
        head,
        runtime_class,
        now,
        result=result,
        reason=reason,
        earliest_start=earliest,
    )


def record_deferred(repository: str, head: str, now: datetime) -> BaseModel | None:
    """A converge the load gate or the recreate window holds back; never raises."""
    _, running = _refs.get(repository, (None, None))
    return _emit(
        repository,
        head,
        _idle_class(_agent(), repository, head, running),
        now,
        result="deferred",
        reason="deploy_gate",
    )


def record_queued(
    repository: str, decision: object, cmd: object, now: datetime
) -> BaseModel | None:
    """A converge accepted as a job: ``queued`` with the job id; never raises."""
    head = _field(decision, "head_ref")
    _, running = _refs.get(repository, (None, None))
    return _emit(
        repository,
        head,
        _idle_class(_agent(), repository, head, running),
        now,
        result="queued",
        job=_field(cmd, "correlation_id"),
    )


def record_failed_head(refusal: object, now: datetime) -> BaseModel | None:
    """An idle build refused for a head whose last dev job failed (OMN-26493); never raises.

    One row per head: the key (repository, head, ``refused``, ``failed_head``) is marked in
    the agent's state directory, so the next refusal of the head writes nothing.
    """
    return _emit(
        "omnibase_infra",
        _field(refusal, "head"),
        _DETECTED,
        now,
        result="refused",
        reason="failed_head",
        job=_field(refusal, "failed_job"),
        verdict_class=_field(refusal, "verdict_class"),
    )


def record_untagged_head(head: str, now: datetime) -> BaseModel | None:
    """An idle build refused because no dev tag names the dev head yet (OMN-25942); never raises.

    One row per head: the key (repository, head, ``refused``, ``untagged_head``) is marked in
    the agent's state directory, so the next refusal of the head writes nothing.
    """
    return _emit(
        "omnibase_infra",
        head,
        _DETECTED,
        now,
        result="refused",
        reason="untagged_head",
    )


def record_command(cmd: object) -> BaseModel | None:
    """A published rebuild command the agent runs: ``queued`` with its job; never raises.

    Only a dev command from a CI requester (``gha/<repo>/...``) naming a sha writes a row. A
    converge carries no ref; its row is written where the converge is accepted.
    """
    execution = _agent()
    routing = importlib.import_module(f"{execution.__package__}.routing")
    repository: str | None = routing.requester_repository(_field(cmd, "requested_by"))
    ref = _field(cmd, "git_ref")
    if (
        repository is None
        or _field(cmd, "runtime_lane") != "dev"
        or not re.fullmatch(r"[0-9a-f]{40}", ref)
    ):
        return None
    job = _field(cmd, "correlation_id")
    return _emit(
        repository,
        ref,
        _PUBLISHED,
        datetime.now(UTC),
        result="queued",
        job=job,
        correlation=job,
    )
