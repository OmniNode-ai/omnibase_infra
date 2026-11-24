# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-only boot supervision and forward redeployment of a healthy revision.

The deploy agent runs outside the runtime it replaces. Recovery and its durable
outbox therefore work even when runtime-effects (and the SCM bus worker) cannot
boot. A snapshot contains resolved environment values: keep it private, never
log it, and never use it outside its original compose project.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import NAMESPACE_URL, uuid5

import yaml
from pydantic import BaseModel, JsonValue, RootModel

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.model_subprocess_outcome import (
    ModelSubprocessOutcome,
)
from omnibase_infra.models.handlers.model_value_holder import ModelValueHolder
from omnibase_infra.models.model_dev_boot_container import ModelDevBootContainer
from omnibase_infra.models.model_dev_boot_point import ModelDevBootPoint
from omnibase_infra.models.model_dev_boot_report_config import ModelDevBootReportConfig
from omnibase_infra.models.model_dev_boot_request import ModelDevBootRequest
from omnibase_infra.models.model_dev_boot_result import ModelDevBootResult
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_error_string,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_dev_preflight_runner import (
        ProtocolDevPreflightRunner,
    )
    from omnibase_infra.protocols.protocol_dev_roll_events import ProtocolDevRollEvents

logger = logging.getLogger(__name__)
DEV_BOOT_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
POLL_SECONDS = 5
DOCKER_READ_SECONDS = 10
# OMN-26314: the roll step applies a service's registered installs before its compose wait.
ENV_STATE_DIR: Final = "DEPLOY_AGENT_STATE_DIR"
# Run by path from the deploy-source clone, as the agent runs its own branch-install entry.
SERVICE_INSTALLS_ENTRY: Final = (
    "src/omnibase_infra/nodes/node_dev_roll_events_effect/"
    "runtime_dev_service_installs.py"
)
# OMN-25942: the post-roll work-ledger proof, run by path from the deploy-source clone.
WORK_LEDGER_CONTRACT: Final = (
    "src/omnibase_infra/contracts/work_ledger_query_client.contract.yaml"
)
ONEX_ENTRY: Final = "scripts/onex"
REPORT_SECONDS = 20
# A full container log (json-file rotation holds up to 30 MB) is read before recovery.
LOG_READ_SECONDS = 60

# OMN-24127: the kept copy of a failed container's log loses credential-shaped
# spans only. The whole-text sanitizer (sanitize_error_string) replaced an 8 MB
# real boot log with its 39-byte notice because the log held one of its sensitive
# words, so the kept log held no cause. The regexes below are the ones
# cli_delegate._redact_refusal_text uses for the same reason.
# The deploy agent runs this module against the published omnibase_infra, which
# may predate the shared helper, so the credential value shapes are held here
# (same list as util_dlq_credential_redaction.CREDENTIAL_VALUE_PATTERNS).
_REDACTION_MARKER: Final[str] = "[REDACTED]"
_CREDENTIAL_VALUE_SHAPES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"onxk_[A-Za-z0-9_-]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{35,}"),
    re.compile(
        r"(?<![A-Za-z0-9_.-])[A-Fa-f0-9]{32}\.[A-Za-z0-9_-]{16,}"
        r"(?![A-Za-z0-9_.-])"
    ),
    re.compile(r"(?i:\bbearer)\s+[A-Za-z0-9._~+/-]{16,}=*"),
)
_PEM_KEY_MATERIAL: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN[^-]*-----.*?(?:-----END[^-]*-----|$)", re.DOTALL
)
_URL_USERINFO: Final[re.Pattern[str]] = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/\s@]+@"
)
_SECRET_PAIR: Final[re.Pattern[str]] = re.compile(
    r"(?P<key>(?:pass(?:word|wd)?|secret|(?:api[_-]?|access[_-]?|auth[_-]?)?"
    r"(?:key|token))\s*=\s*)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)


def redact_credential_spans(text: str) -> str:
    """Replace only the credential-shaped spans of ``text`` and keep every other line.

    Applied in order: the shared credential value shapes (an ``sk-`` key, a JWT,
    a Bearer value, ...), PEM key material, URL userinfo (keeping scheme and
    host), and the value of a ``key=value`` pair whose key names a secret
    (keeping the key). No truncation: an incident log stays whole and only the
    credential spans shrink.
    """
    redacted = text
    for shape in _CREDENTIAL_VALUE_SHAPES:
        redacted = shape.sub(_REDACTION_MARKER, redacted)
    redacted = _PEM_KEY_MATERIAL.sub("[REDACTED key material]", redacted)
    redacted = _URL_USERINFO.sub(r"\g<scheme>[REDACTED]@", redacted)
    return _SECRET_PAIR.sub(r"\g<key>[REDACTED]", redacted)


class DevBootFailedError(RuntimeError):
    """A dev runtime failed boot, or cannot be safely replaced."""


class DevBootSettingsDriftError(DevBootFailedError):
    """A healthy dev container carries a compose config hash the clone no longer renders.

    The container was created from the compose files the deploy read when it ran
    ``up``. The deploy-source clone is a canonical clone that the canonical-clone sync
    fast-forwards while a job runs (OMN-25942: job 20cf29c6 pinned commit 925858038 at
    08:04:02Z and the sync moved the clone at 08:12:26Z and 08:18:26Z), so the
    capture after a verified roll can render other files than ``up`` did. The family is
    healthy; only the recovery point cannot be advanced.
    """

    def __init__(self, service: str, container_hash: str, rendered_hash: str) -> None:
        super().__init__(
            f"{service}: compose settings differ from the healthy container"
        )
        self.service = service
        self.container_hash = container_hash
        self.rendered_hash = rendered_hash


class DevBootRecoveryError(RuntimeError):
    """The candidate deploy failed, including when recovery succeeded."""


class DevCompanionLaneFailedError(RuntimeError):
    """A lane deployed beside the runtime failed (the gateway forwarder).

    The companion lane carries no delegation or runtime traffic, so its failure
    fails the job and is reported, but it is not by itself a reason to replace a
    runtime that is up and healthy (OMN-21645). The guard still restores the
    retained family when the runtime is unhealthy or absent.
    """


class DevOnexApiStepFailedError(DevBootFailedError):
    """The onex-api step of a dev roll stopped the job.

    onex-api is not the runtime (it is not in DEV_BOOT_SERVICES), so its
    failure fails the job and is reported, but it is not by itself a reason to
    replace a runtime that is up and healthy (OMN-24471). The guard still
    restores the retained family when the runtime is unhealthy or absent.
    """

    def __init__(self, message: str, service: str) -> None:
        super().__init__(message)
        self.service = service


def roll_expected(expected: list[str], roll_error: RuntimeError | None) -> list[str]:
    """The services the executor verifies after a roll stopped at ``roll_error``.

    A failed onex-api step is raised as itself: the step already waited on that
    service, and verifying or starting it again would let a crash-looping
    container replace the typed error with a census error, which recovers as a
    runtime failure (OMN-24471).
    """
    if isinstance(roll_error, DevOnexApiStepFailedError):
        return [service for service in expected if service != roll_error.service]
    return expected


def _roll_events_module(execution: ModuleType) -> ModuleType:
    """The dev roll events composition root, loaded by the agent's own effect loader."""
    return execution.load_dev_boot_guard(
        "nodes.node_dev_roll_events_effect.runtime_dev_roll_events"
    )


#: The canary gate's modules in import order, loaded from this checkout by path: the
#: deploy agent's environment carries no ``omnibase_infra`` package (OMN-26463).
_CANARY_MODULES: Final = (
    "enums/enum_runtime_canary_verdict.py",
    "models/model_runtime_canary_candidate.py",
    "models/model_runtime_canary_gate_decision.py",
    "models/model_runtime_canary_receipt.py",
    "models/model_runtime_canary_request.py",
    "protocols/protocol_runtime_canary_probe.py",
    "handlers/handler_runtime_canary.py",
)


def _canary_gate() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    for relative in _CANARY_MODULES:
        name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, root / relative)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load the runtime canary module: {relative}")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules["omnibase_infra.handlers.handler_runtime_canary"]


def _require_canary_receipt(command: Mapping[str, JsonValue], state_dir: Path) -> None:
    """OMN-26463: refuse a roll whose candidate has no PASS canary receipt.

    This reads a receipt file and nothing else; the health gate and this guard's boot
    supervision are as they were. The gate is off until the agent's state directory
    holds a ``canary-receipts`` directory.
    """
    _canary_gate().require_for_roll(command, state_dir)


def _load_roll_events(execution: ModuleType) -> ProtocolDevRollEvents:
    """The real events object of the running deploy agent."""
    return cast(
        "ProtocolDevRollEvents",
        _roll_events_module(execution).build_events(execution),
    )


def run_preflight(
    executor: ProtocolDevPreflightRunner,
    command: BaseModel,
    on_phase_update: Callable[[StrEnum, StrEnum], None],
) -> None:
    """Run the deploy preflight; a dev roll that fails it is announced, then the error goes on.

    The preflight runs before the boot guard is prepared, so nothing else announces a
    dev roll that fails there: the roll events publish its manifest and an incident
    naming the issuer and the failed phase (OMN-24048). Evidence only: announcing never
    raises and never replaces the preflight's own error.
    """
    try:
        executor.preflight(on_phase_update=on_phase_update)
    except Exception as error:
        execution = sys.modules["deploy_agent.executor"]
        _roll_events_module(execution).announce_preflight_failure(
            execution,
            command.model_dump(mode="json"),
            str(error),
            execution.Phase.PREFLIGHT.value,
        )
        raise


def _checked(
    argv: list[str],
    timeout: int = DOCKER_READ_SECONDS,
    *,
    execution: ModuleType,
    env: dict[str, str] | None = None,
) -> str:
    result = execution._run(argv, timeout=timeout, env=env)
    if result.returncode != 0:
        # Compose stderr can include interpolated secrets. The command's kind
        # and exit code suffice here; boot causes come from the filtered logs.
        raise DevBootFailedError(f"{argv[0]} {argv[1]} exited {result.returncode}")
    return str(
        result.stdout + result.stderr
        if argv[:2] == ["docker", "logs"]
        else result.stdout
    )


def _prove_work_ledger_answers(execution: ModuleType) -> None:
    """After a dev runtime recreate verified, prove one ledger read answers (OMN-25942).

    The command line and the bound are ``post_roll_proof`` in
    ``work_ledger_query_client.contract.yaml`` of the checkout the agent runs from;
    this module types neither. The read runs from the agent's own lab host and must
    exit 0 inside the bound with one JSON object. Anything else fails the roll, which
    ``protect`` turns into the previous image's restore.
    """
    contract = Path(execution.REPO_DIR) / WORK_LEDGER_CONTRACT
    declared = yaml.safe_load(contract.read_text(encoding="utf-8"))["post_roll_proof"]
    argv, bound = declared["argv"], declared["timeout_s"]
    if not (
        isinstance(argv, list)
        and argv
        and all(isinstance(part, str) for part in argv)
        and isinstance(bound, int)
        and bound > 0
    ):
        raise DevBootFailedError("work ledger proof declaration is malformed")
    command = [
        f"{execution.REPO_DIR}/{ONEX_ENTRY}",
        *argv,
        "--timeout",
        str(bound),
    ]
    try:
        done = execution._run(command, timeout=bound)
    except execution.subprocess.TimeoutExpired as exc:
        raise DevBootFailedError(f"work ledger: no answer within {bound}s") from exc
    except OSError as exc:
        raise DevBootFailedError(
            f"work ledger: {argv[0]} could not start: {_safe(str(exc))}"
        ) from exc
    if done.returncode != 0:
        raise DevBootFailedError(
            f"work ledger: {argv[0]} exited {done.returncode}: "
            f"{_safe(done.stderr or done.stdout)[:240]}"
        )
    try:
        answer = json.loads(done.stdout)
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise DevBootFailedError("work ledger: the answer is not one JSON object")


def _safe(text: str) -> str:
    return " ".join(
        sanitize_error_message(RuntimeError(text), max_length=2000)
        .replace("|", " ")
        .split()
    )[:2000]


def _step_failed(
    service: str, execution: ModuleType, message: str
) -> DevBootFailedError:
    """OMN-24471: a failed onex-api step stays distinguishable from a runtime step's."""
    if service == execution.ONEX_API_SERVICE:
        return DevOnexApiStepFailedError(message, service)
    return DevBootFailedError(message)


def _declares_installs_for(state_dir: Path, service: str) -> bool:
    """Whether the agent's single store declares an install for ``service`` (OMN-26314).

    Unreadable or absent reads as none: the roll then goes as it always did and the
    post-rebuild apply and the idle apply still carry the install.
    """
    try:
        records = json.loads(
            (state_dir / "branch-installs" / "records.json").read_text(encoding="utf-8")
        )
        return any(
            service in install["services"] for install in records.get("installs", [])
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def apply_installs_before_wait(
    argv: list[str],
    timeout: int,
    state_dir: Path,
    *,
    execution: ModuleType,
    env: Mapping[str, str],
) -> None:
    """Start the replacement WITHOUT the wait and put its registered installs in (OMN-26314).

    A registered install exists to make the replacement healthy, so it must be in the
    container the compose wait reads. The caller issues the unchanged ``up --wait`` next;
    the wait, its budget and the refusal on an unhealthy runtime are not touched here.
    Never a verdict: an apply that fails or cannot read back whole is logged at ERROR
    and the wait judges the runtime as it is. Dev lane only: the roll step is the dev
    roll and the apply refuses a prod or stability compose project.
    """
    service = argv[-1]
    try:
        started = execution._run(argv, timeout=timeout, env=dict(env))
    except execution.subprocess.TimeoutExpired as exc:
        raise _step_failed(
            service,
            execution,
            f"Runtime roll stopped at {service}: compose up timed out",
        ) from exc
    if started.returncode != 0:
        raise _step_failed(
            service,
            execution,
            f"Runtime roll stopped at {service}: "
            f"{_safe(started.stderr) or 'compose up failed'}",
        )
    lane = execution.EnumRuntimeLane.DEV
    apply_argv = [
        sys.executable,
        f"{execution.REPO_DIR}/{SERVICE_INSTALLS_ENTRY}",
        "--state-dir",
        str(state_dir),
        "--compose-project",
        execution.lane_config_for(lane).compose_project,
        "--service",
        service,
    ]
    apply_env = {
        **env,
        "PYTHONPATH": ":".join(
            part
            for part in (f"{execution.REPO_DIR}/src", env.get("PYTHONPATH", ""))
            if part
        ),
    }
    try:
        applied = execution._run(apply_argv, timeout=timeout, env=apply_env)
    except (OSError, execution.subprocess.TimeoutExpired):
        logger.exception(
            "branch install before the roll wait could not run for %s "
            "friction_type=branch_install_failed",
            service,
        )
        return
    for line in applied.stderr.splitlines():
        if "friction_type=" in line:
            logger.error("%s", line)
        elif line.strip():
            logger.info("%s", line)
    if applied.returncode != 0:
        logger.error(
            "branch install before the roll wait exited %s for %s: %s "
            "friction_type=branch_install_failed",
            applied.returncode,
            service,
            _safe(applied.stdout or applied.stderr)[-300:],
        )


def run_runtime_roll_step(
    argv: list[str],
    timeout: int,
    *,
    execution: ModuleType,
    extra_env: Mapping[str, str] | None = None,
) -> ModelSubprocessOutcome:
    """Wait for one dev replacement's declared health before touching its peer.

    The executor retains its compose model, environment and I/O boundary. A
    failed wait is terminal even if a later read could find the project running
    or at the expected config hash: those reads cannot prove the remaining
    services were updated or that the replacement finished booting.
    """
    service = argv[-1]
    lane = execution.EnumRuntimeLane.DEV
    if service in execution.SERVICE_HEALTH_TARGET_INDEX:
        # OMN-25106: a running container that stayed unhealthy is not replaced
        # by the compose up below, so it is restarted first.
        restart_if_wedged(
            execution._health_target_container(lane, service),
            timeout,
            execution=execution,
        )
    wait_budget = execution.runtime_health_probe_budget(lane, service)
    command = list(argv)
    index = command.index("--no-deps")
    command[index:index] = ["--wait", "--wait-timeout", str(wait_budget)]
    env = execution._compose_env(extra_env)
    state_dir = env.get(ENV_STATE_DIR, "").strip()
    if state_dir and _declares_installs_for(Path(state_dir), service):
        apply_installs_before_wait(
            argv, timeout, Path(state_dir), execution=execution, env=env
        )
    try:
        result = execution._run(
            command,
            timeout=max(
                timeout, wait_budget + execution.RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS
            ),
            env=env,
        )
    except execution.subprocess.TimeoutExpired as exc:
        raise _step_failed(
            service,
            execution,
            f"Runtime roll stopped at {service}: compose up timed out",
        ) from exc
    if result.returncode != 0:
        raise _step_failed(
            service,
            execution,
            f"Runtime roll stopped at {service}: "
            f"{_safe(result.stderr) or 'compose health wait failed'}",
        )
    if not execution.verify_containers_up(
        [service], timeout_s=execution.CONTAINER_VERIFY_TIMEOUT_SECONDS, lane=lane
    )[0]:
        raise _step_failed(
            service,
            execution,
            f"Runtime roll stopped at {service}: container did not reach running state",
        )
    return ModelSubprocessOutcome(
        returncode=int(result.returncode),
        stdout=str(result.stdout),
        stderr=str(result.stderr),
        timed_out=False,
    )


def run_compose_up_steps(
    steps: list[list[str]],
    timeout: int,
    *,
    rolling: bool,
    lane: str,
    execution: ModuleType,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[ModelSubprocessOutcome, DevBootFailedError | None]:
    """Stop a failed dev roll, retaining its verdict for container recovery.

    Other lanes keep their running-state check and subprocess timeout path.
    The executor may restore stopped containers after a dev failure, but must
    raise the retained error even when that recovery succeeds.
    """
    env = execution._compose_env(extra_env)
    result = ModelSubprocessOutcome(returncode=0, stdout="", stderr="", timed_out=False)
    for step in steps:
        if rolling and lane is execution.EnumRuntimeLane.DEV:
            try:
                result = run_runtime_roll_step(
                    step, timeout, execution=execution, extra_env=extra_env
                )
            except DevBootFailedError as exc:
                return ModelSubprocessOutcome(
                    returncode=1, stdout="", stderr=str(exc), timed_out=False
                ), exc
        else:
            process = execution._run(step, timeout=timeout, env=env)
            result = ModelSubprocessOutcome(
                returncode=int(process.returncode),
                stdout=str(process.stdout),
                stderr=str(process.stderr),
                timed_out=False,
            )
        if result.returncode != 0 or (
            rolling
            and lane is not execution.EnumRuntimeLane.DEV
            and not execution.verify_containers_up(
                step[-1:],
                timeout_s=execution.CONTAINER_VERIFY_TIMEOUT_SECONDS,
                lane=lane,
            )[0]
        ):
            break
    return result, None


def preflight_dev_images(
    lane: str, services: list[str], *, execution: ModuleType
) -> None:
    """Refuse an incompatible dev image before compose up can replace anything."""
    if lane is not execution.EnumRuntimeLane.DEV:
        return
    config = execution.lane_config_for(lane)
    for service in DEV_BOOT_SERVICES:
        if service not in services or service in config.disabled_services:
            continue
        with execution.candidate_probe_file_args(lane, service) as file_args:
            argv = [
                "docker",
                "compose",
                *file_args,
                "-p",
                config.compose_project,
                "--profile",
                "runtime",
                "run",
                "--rm",
                "--no-deps",
                "--pull",
                "never",
                "--entrypoint",
                "python",
                service,
                "-c",
                execution.IMAGE_PREFLIGHT_SCRIPT,
            ]
            result = execution._run(argv, timeout=60, env=execution._compose_env())
        if result.returncode != 0:
            # Only our own setting diagnoses are safe to forward. Other Python
            # or compose failures still refuse the image without dumping env.
            diagnoses = [
                line
                for line in result.stderr.splitlines()
                if line.startswith(
                    ("BIFROST_BASE_CONTRACT_PATH", "ONEX_ACTIVE_RUNTIME_PACKAGES")
                )
            ]
            reason = (
                _safe("; ".join(diagnoses))
                or f"candidate check exited {result.returncode}"
            )
            raise DevBootFailedError(f"{service}: {reason}")


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise DevBootFailedError("compose document requires an object")
    return value


def _json_string(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise DevBootFailedError("incident job requires a string")
    return value


def _healthy(container: ModelDevBootContainer) -> bool:
    state = container.get("State", {})
    return bool(
        state.get("Status") == "running"
        and state.get("Health", {}).get("Status") == "healthy"
    )


def _window(container: ModelDevBootContainer) -> int:
    health = container["Config"].get("Healthcheck", {})
    nanos = health.get("StartPeriod", 0) + health.get("Retries", 0) * (
        health.get("Interval", 0) + health.get("Timeout", 0)
    )
    if nanos <= 0:
        raise DevBootFailedError("dev runtime has no declared boot health window")
    return math.ceil(float(nanos) / 1_000_000_000)


def _unchanged_and_healthy(
    point: ModelDevBootPoint,
    read_containers: Callable[[], list[ModelDevBootContainer]],
    *,
    execution: ModuleType,
) -> bool:
    try:
        current = read_containers()
    except (OSError, ValueError, RuntimeError, execution.subprocess.SubprocessError):
        return False
    return (
        all(_healthy(c) for c in current)
        and {c["Id"] for c in current} == set(point.container_ids)
        and {c["Image"] for c in current} == set(point.images.values())
    )


def _all_healthy(
    read_containers: Callable[[], list[ModelDevBootContainer]], *, execution: ModuleType
) -> bool:
    try:
        return all(_healthy(c) for c in read_containers())
    except (OSError, ValueError, RuntimeError, execution.subprocess.SubprocessError):
        return False


def retire_lane_services[LaneT: BaseModel](
    config: LaneT, retired: Collection[str], phase_by_service: Mapping[str, StrEnum]
) -> LaneT:
    """``config`` with ``retired`` services disabled and their settle phases off.

    A service the policy retires by name is never recreated, so the phase that
    only builds or pins its image has nothing to deliver (OMN-24476).
    """
    current = config.model_dump(
        mode="json", include={"disabled_services", "disabled_phases"}
    )
    phases = {phase_by_service[s] for s in retired if s in phase_by_service}
    return config.model_copy(
        update={
            "disabled_services": frozenset(current["disabled_services"]).union(retired),
            "disabled_phases": frozenset(current["disabled_phases"]).union(phases),
        }
    )


def folded_dev_services(
    policy: Mapping[str, str], folded: frozenset[str], prefix: str = "DEV"
) -> set[str]:
    """Retirement uses the same contract mapping as the compose gates.

    A lane that folds nothing (any other dev instance) retires nothing. On the
    lane that folds, the services main carries with no profile of its own
    (``<prefix>_RUNTIME_MAIN_RETIRED_SERVICES``) are retired with the folded
    ones; a lane that declares none (stability) retires only the mapped ones.
    """
    if not folded:
        return set()
    mapping = json.loads(policy.get("DEV_RUNTIME_MAIN_FOLDABLE_SERVICES", "{}"))
    if (
        not isinstance(mapping, dict)
        or any(
            not isinstance(role, str)
            or not isinstance(service, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", service)
            for role, service in mapping.items()
        )
        or len(set(mapping.values())) != len(mapping)
    ):
        raise RuntimeError("invalid dev foldable service mapping")
    if missing := folded - mapping.keys():
        raise RuntimeError(f"fold policy maps no service for {sorted(missing)}")
    retired = {
        service.strip()
        for service in policy.get(f"{prefix}_RUNTIME_MAIN_RETIRED_SERVICES", "").split(
            ","
        )
        if service.strip()
    }
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", service) for service in retired):
        raise RuntimeError("invalid dev retired service list")
    return {*(mapping[profile] for profile in folded), *retired}


#: ``docker inspect`` states of a lane container that mean it is down.
DOWN_STATES = frozenset({"created", "exited", "dead"})
#: OMN-25106: consecutive failed health probes after which a running container
#: reads as wedged, which restores it like a stopped one. The dev runtime's
#: compose healthcheck probes every 30 s, so 20 is a 10 minute non-200 window;
#: docker does not count failed probes inside ``start_period``.
WEDGED_FAILING_STREAK: Final = 20
#: The state, then (when the container declares a healthcheck) its health
#: status and failing streak, as whitespace-separated fields.
PRESENCE_INSPECT_FORMAT: Final = (
    "{{.State.Status}}"
    " {{if .State.Health}}{{.State.Health.Status}} {{.State.Health.FailingStreak}}{{end}}"
)


def _is_wedged(fields: Sequence[str]) -> bool:
    """Whether inspect fields read a RUNNING container that stayed unhealthy too long.

    Only state ``running`` can be wedged: a paused or restarting container with
    a failing streak is not what a restart cures, so it reads as it did before.
    """
    return (
        len(fields) >= 3
        and fields[0].lower() == "running"
        and fields[1].lower() == "unhealthy"
        and fields[2].isdigit()
        and int(fields[2]) >= WEDGED_FAILING_STREAK
    )


def _presence_of(container: str, fields: Sequence[str]) -> tuple[str, str]:
    """``down`` or ``running`` with its detail, from one container's inspect fields."""
    state = fields[0].lower() if fields else ""
    if state in DOWN_STATES:
        return "down", f"{container} is {state}"
    if _is_wedged(fields):
        return "down", (
            f"{container} is {state} but unhealthy: {fields[2]} consecutive failed "
            f"health probes, at least {WEDGED_FAILING_STREAK} read as wedged"
        )
    return "running", f"{container} is {state}"


def restart_if_wedged(container: str, timeout: int, *, execution: ModuleType) -> bool:
    """``docker restart`` one container the presence read calls wedged (OMN-25106).

    The restore's rebuild converges with ``compose up -d``, which leaves a
    running container whose image and configuration are unchanged exactly as it
    is, and the post-deploy verify declines to recreate a runtime that still
    answers ``/health`` degraded. A wedged container therefore has to be
    restarted before the roll waits on it. A read or a restart that docker
    refuses is not an error here: the roll goes on and its own wait decides.
    """
    try:
        read = execution._run(
            ["docker", "inspect", "--format", PRESENCE_INSPECT_FORMAT, container],
            timeout=DOCKER_READ_SECONDS,
        )
        if read.returncode != 0 or not _is_wedged(read.stdout.split()):
            return False
        restarted = execution._run(["docker", "restart", container], timeout=timeout)
    except (OSError, execution.subprocess.TimeoutExpired):
        return False
    return bool(restarted.returncode == 0)


def lane_runtime_containers(
    health_targets: Sequence[tuple[str, int]], disabled_services: frozenset[str]
) -> tuple[str, ...]:
    """The containers a dev lane needs running to serve delegation, main first (OMN-22845).

    The runtime-effects container is the only consumer of the delegate-skill
    command topic, so a lane with main running and effects absent accepts every
    request and answers none. The effects container is ``<main>-effects`` on
    every lane. A lane that folds effects into main (its effects health target
    then names the main container) or retires the service expects main alone.
    """
    main = health_targets[0][0]
    effects_target = health_targets[1][0] if len(health_targets) > 1 else main
    if effects_target == main or "runtime-effects" in disabled_services:
        return (main,)
    return (main, f"{main}-effects")


def running_ref_if_healthy(
    container: str,
    repository: str,
    read_ref: Callable[[str, str], str | None],
) -> str | None:
    """``read_ref``'s answer for a healthy runtime, ``""`` for an unhealthy one (OMN-24562).

    A runtime that failed its boot is parked and still answers ``docker exec`` with
    the provenance of the image it was built from, so the stability follow read it
    as running the tag it had just built and never rebuilt it. An unhealthy runtime
    runs no revision: ``""`` names no tag and no commit that contains one. ``None``
    stays what docker could not answer.
    """
    try:
        shown = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=DOCKER_READ_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if shown.returncode == 0 and shown.stdout.strip() == "unhealthy":
        return ""
    return read_ref(container, repository)


def healthy_ref_reader(
    container: Callable[[], str], read_ref: Callable[[str, str], str | None]
) -> Callable[[str], str | None]:
    """The repository -> running ref reader of a lane's runtime, unhealthy reading ``""``.

    ``container`` resolves the lane's runtime container at READ time, never when the
    reader is built: the agent builds this reader in ``__init__`` on every lane, and a
    lane configuration that cannot resolve (a bad stability fold policy) must stop
    only the lane that reads it. A resolution failure is logged and reads ``None``,
    what docker could not answer, so that lane's follow starts no job and every other
    lane runs on.
    """

    def read(repository: str) -> str | None:
        try:
            name = container()
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception(
                "runtime container of the followed lane unresolved: %s", error
            )
            return None
        return running_ref_if_healthy(name, repository, read_ref)

    return read


def read_lane_presence(
    health_targets: Sequence[tuple[str, int]],
    disabled_services: frozenset[str],
    *,
    run: Callable[..., ModelSubprocessOutcome],
    unreadable: tuple[type[Exception], ...],
) -> tuple[str, str]:
    """``(state, detail)`` of the lane's containers: running, down, companion_down or unknown.

    Main reads on its own: absent, created, exited, dead or wedged (running
    but unhealthy for ``WEDGED_FAILING_STREAK`` probes, OMN-25106) is ``down``. With
    main running, a down container beside it is ``companion_down`` and one that
    cannot be read is ``unknown``, which never restores. ``unreadable`` is what
    ``run`` raises when docker does not answer.
    """
    states: list[tuple[str, str]] = []
    for container in lane_runtime_containers(health_targets, disabled_services):
        try:
            result = run(
                ["docker", "inspect", "--format", PRESENCE_INSPECT_FORMAT, container],
                timeout=DOCKER_READ_SECONDS,
            )
        except unreadable as exc:
            states.append(("unknown", f"docker inspect {container}: {exc}"))
            continue
        said = (result.stderr or result.stdout).strip()
        if result.returncode != 0:
            if "no such object" in said.lower() or "no such container" in said.lower():
                states.append(("down", f"no container named {container}"))
            else:
                states.append(
                    (
                        "unknown",
                        f"docker inspect {container} exited {result.returncode}: {said[:200]}",
                    )
                )
        else:
            states.append(_presence_of(container, result.stdout.split()))
    if states[0][0] != "running":
        return states[0]
    for wanted in ("down", "unknown"):
        named = [detail for state, detail in states[1:] if state == wanted]
        if named:
            return ("companion_down" if wanted == "down" else wanted), "; ".join(named)
    return "running", "; ".join(detail for _, detail in states)


def fold_compose_env(policy: Mapping[str, str], primary: bool) -> dict[str, str]:
    """Keep compose activation aligned with retirement after all env overlays."""
    selected = {}
    for key, value in policy.items():
        if key.startswith("DEV_RUNTIME_MAIN_"):
            if key.endswith(("_RUNTIME_PROFILE", "_FULL_PROFILE")):
                selected[key] = (
                    value
                    if primary
                    else ("runtime" if key.endswith("_RUNTIME_PROFILE") else "full")
                )
            elif key.endswith(
                ("_FOLDED_PROFILES", "_FOLDABLE_SERVICES", "_RETIRED_SERVICES")
            ):
                selected[key] = value
    return selected


def _folded_profiles(execution: ModuleType) -> tuple[str, ...]:
    """The roles the dev overlay on disk folds into main (OMN-21288)."""
    return tuple(sorted(execution.lane_folded_profiles(execution.EnumRuntimeLane.DEV)))


def _refuse_cross_fold_pairing(point: ModelDevBootPoint, execution: ModuleType) -> None:
    """An image is only ever restored under the overlay of its own fold generation.

    A healthy family captured before the fold runs a main without the folded
    writers' contracts; under the fold overlay their containers are retired
    too, so nothing would project. Moving the image and the overlay together is
    a rollback by revision, through a deploy of that git ref.
    """
    current = _folded_profiles(execution)
    if tuple(sorted(point.folded_profiles)) != current:
        raise DevBootFailedError(
            f"retained revision {point.revision[:12]} ran with "
            f"{len(point.folded_profiles)} folded profiles but the dev overlay "
            f"folds {len(current)}: refusing to pair its image with this "
            "overlay; roll back by revision (deploy the git ref whose overlay "
            "matches)"
        )


def _dev_lane_services(disabled_services: Collection[str]) -> tuple[str, ...]:
    """The DEV boot services a lane runs, given the services its policy disables."""
    return tuple(
        service for service in DEV_BOOT_SERVICES if service not in disabled_services
    )


class HandlerDevBootGuard:
    """Supervise only the selected DEV compose project under its caller's lock."""

    def __init__(
        self,
        state_dir: Path,
        *,
        config_hashes: Callable[[Sequence[str]], dict[str, str]],
        verify_recovery: Callable[[Callable[[StrEnum, StrEnum], None]], None],
        execution: ModuleType,
        reporting: ModelDevBootReportConfig,
        log_byte_cap: int | None = None,
    ) -> None:
        self.execution = execution
        self.reporting = reporting
        self.config_hashes = config_hashes
        self.verify_recovery = verify_recovery
        if log_byte_cap is not None and log_byte_cap <= 0:
            raise ValueError("log_byte_cap must be positive when set")
        self.log_byte_cap = log_byte_cap
        self.events: ProtocolDevRollEvents | None = None
        self._command: Mapping[str, JsonValue] | None = None
        self._prepared_at: datetime | None = None
        self.config = self.execution.lane_config_for(self.execution.EnumRuntimeLane.DEV)
        self.services = _dev_lane_services(self.config.disabled_services)
        self._retained_services = self.services
        self.root = state_dir / "dev-boot-guard" / self.config.compose_project
        self.snapshot_path = self.root / "last-healthy.json"
        self.point: ModelValueHolder[ModelDevBootPoint] = ModelValueHolder()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure handler over the host's Docker compose project."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting: retags images and recreates dev runtime containers."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelDevBootRequest) -> ModelDevBootResult:
        if request.operation == "prepare":
            self.prepare()
        elif request.operation == "wait":
            self.wait()
        elif request.operation == "preflight":
            preflight_dev_images(
                self.execution.EnumRuntimeLane.DEV,
                list(self.services),
                execution=self.execution,
            )
        else:
            deliver_reports(self.root, execution=self.execution, config=self.reporting)
        return ModelDevBootResult(operation=request.operation, status="pass")

    def wait(self, *, deployed_generation: bool = False) -> None:
        """All replicas must be Docker healthy; exhausted boots fail immediately.

        ``deployed_generation`` is the post-roll wait of a deploy: it adopts the
        service list of the generation just rolled before it reads containers.
        """
        if deployed_generation:
            # The guard is built before git_pull, from the previous
            # generation's policy, so recovery knows what that generation ran.
            # A deploy that folds a role (OMN-22930: runtime-effects into main)
            # retires a service the pre-pull list still names; re-read the
            # list of the generation just rolled and retain the previous one
            # for the recovery path.
            self._retained_services = self.services
            self.config = self.execution.lane_config_for(
                self.execution.EnumRuntimeLane.DEV
            )
            self.services = _dev_lane_services(self.config.disabled_services)
        started = self.execution._dev_boot_clock()
        containers = _dev_containers(self)
        window = max(_window(c) for c in containers)
        logger.info(
            "Dev boot window project=%s seconds=%d", self.config.compose_project, window
        )
        while True:
            failures: list[str] = []
            for container in containers:
                if _healthy(container):
                    continue
                service = container["Config"]["Labels"]["com.docker.compose.service"]
                logs = _checked(
                    ["docker", "logs", "--tail", "80", container["Id"]],
                    execution=self.execution,
                )
                for exhausted in re.finditer(r"BOOT FAILED (\d+)/(\d+):[^\n]*", logs):
                    if int(exhausted[1]) >= int(exhausted[2]):
                        raise DevBootFailedError(f"{service}: {_safe(exhausted[0])}")
                if "RETRIES EXHAUSTED:" in logs:
                    causes = [
                        line for line in logs.splitlines() if "BOOT FAILED" in line
                    ]
                    raise DevBootFailedError(
                        f"{service}: {_safe(causes[-1] if causes else 'RETRIES EXHAUSTED')}"
                    )
                state = container.get("State", {})
                if (
                    state.get("Status") in {"exited", "dead"}
                    or state.get("Health", {}).get("Status") == "unhealthy"
                ):
                    raise DevBootFailedError(
                        f"{service}: {_safe(causes_from_logs(logs))}"
                    )
                failures.append(service)
            if not failures:
                return
            remaining = window - (self.execution._dev_boot_clock() - started)
            if remaining <= 0:
                raise DevBootFailedError(
                    f"dev boot window exhausted ({window}s): {', '.join(failures)}"
                )
            time.sleep(min(POLL_SECONDS, remaining))
            containers = _dev_containers(self)

    def _capture(self) -> ModelDevBootPoint:
        containers = _dev_containers(self)
        if not all(_healthy(c) for c in containers):
            raise DevBootFailedError("cannot capture an unhealthy dev runtime")
        rendered = (
            RootModel[dict[str, JsonValue]]
            .model_validate_json(
                _checked(
                    [
                        "docker",
                        "compose",
                        *self.execution._compose_file_args(
                            self.execution.EnumRuntimeLane.DEV
                        ),
                        "-p",
                        self.config.compose_project,
                        "--profile",
                        "runtime",
                        "config",
                        "--format",
                        "json",
                    ],
                    self.execution.CONFIG_HASH_TIMEOUT_SECONDS,
                    env=self.execution._compose_env(),
                    execution=self.execution,
                )
            )
            .root
        )
        hashes = self.config_hashes(self.services)
        for container in containers:
            labels = container["Config"]["Labels"]
            service = labels["com.docker.compose.service"]
            if not hashes.get(service) or hashes[service] != labels.get(
                "com.docker.compose.config-hash"
            ):
                raise DevBootSettingsDriftError(
                    service,
                    labels.get("com.docker.compose.config-hash", ""),
                    hashes.get(service, ""),
                )
        # Compose's canonical JSON already escapes literal dollars for reuse.
        # Preserve those strings verbatim when writing the recovery config.
        compose = copy.deepcopy(rendered)
        compose["services"] = {
            service: _json_object(compose["services"])[service]
            for service in self.services
        }
        images: dict[str, str] = {}
        service_images: dict[str, str] = {}
        revisions: set[str] = set()
        for container in containers:
            labels = container["Config"]["Labels"]
            service = labels["com.docker.compose.service"]
            revision = labels.get("org.opencontainers.image.revision", "")
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise DevBootFailedError(
                    f"{service}: healthy image has no commit revision"
                )
            revisions.add(revision)
            image_id = container["Image"]
            if service in service_images and service_images[service] != image_id:
                raise DevBootFailedError(
                    f"{service}: replicas disagree on image identity"
                )
            service_images[service] = image_id
            retained = f"onex-boot-good-{self.config.compose_project}-{service}:{image_id.removeprefix('sha256:')}"
            if retained in images and images[retained] != image_id:
                raise DevBootFailedError(
                    f"{service}: replicas disagree on image identity"
                )
            images[retained] = image_id
            self.execution._tag_image(image_id, retained)
            spec = _json_object(_json_object(compose["services"])[service])
            spec["image"] = retained
            spec["pull_policy"] = "never"
            spec.pop("build", None)
            spec.pop("depends_on", None)
        if len(revisions) != 1:
            raise DevBootFailedError(
                "healthy runtime family has mixed commit revisions"
            )
        # Preserve existing data/network identities. A recovery neither creates
        # fresh volumes nor brings up the broker or database underneath them.
        for kind in ("networks", "volumes"):
            for name, resource in _json_object(compose.get(kind, {})).items():
                _json_object(compose[kind])[name] = {
                    "name": _json_object(resource)["name"],
                    "external": True,
                }
        point = ModelDevBootPoint(
            project=self.config.compose_project,
            revision=next(iter(revisions)),
            container_ids=tuple(sorted(c["Id"] for c in containers)),
            images=images,
            compose=compose,
            folded_profiles=_folded_profiles(self.execution),
        )
        self.execution._atomic_write_private_json(
            self.snapshot_path, point.model_dump(mode="json")
        )
        return point

    def prepare(self, command: Mapping[str, JsonValue] | None = None) -> None:
        """Retain the live healthy family BEFORE git updates compose settings.

        Given the roll's command (the deploy agent always gives it) this is a real
        roll with an issuer: the guard then announces the roll through the dev roll
        events and keeps each failing container's log (OMN-24013). Without one it
        behaves as it always did and announces nothing.
        """
        if command is not None:
            _require_canary_receipt(command, self.root.parents[1])
            self._command = command
            if self.events is None:
                try:
                    self.events = _load_roll_events(self.execution)
                except Exception:
                    # The events are evidence; a sender that cannot be built (no
                    # broker settings) never fails or delays a roll.
                    logger.exception(
                        "dev roll events unavailable friction_type=dev_event_publish_failed"
                    )
            if self.log_byte_cap is None:
                self.log_byte_cap = _roll_events_module(self.execution).log_byte_cap(
                    self.execution
                )
            self._prepared_at = datetime.now(UTC)
        self._retain()
        if command is not None:
            _announce(self.events, lambda events: events.manifest(command))

    def _retain(self) -> None:
        try:
            self.point.hold(self._capture())
            return
        except (
            OSError,
            ValueError,
            RuntimeError,
            self.execution.subprocess.SubprocessError,
        ):
            if not self.snapshot_path.is_file():
                raise DevBootFailedError(
                    "no last healthy dev revision; refusing runtime replacement"
                ) from None
        point = ModelDevBootPoint.model_validate_json(
            self.execution._read_private_text(self.snapshot_path)
        )
        if point.project != self.config.compose_project:
            raise DevBootFailedError("last healthy snapshot belongs to another project")
        # A snapshot of another fold generation is held, not refused here: the
        # fold change itself moves the config hash, so refusing left every
        # forward deploy across a fold with no path at all. _restore still
        # refuses the pairing, so a failed deploy records the incident and
        # names the rollback by revision instead of restoring the old image.
        self.point.hold(point)

    def _restore(self) -> None:
        if self.point.current is None:
            raise DevBootFailedError("no last healthy dev revision")
        _refuse_cross_fold_pairing(self.point.current, self.execution)
        for retained, image_id in self.point.current.images.items():
            self.execution._tag_image(image_id, retained)
        path = self.root / "recovery-compose.json"
        self.execution._atomic_write_private_json(path, self.point.current.compose)
        _checked(
            [
                "docker",
                "compose",
                "-f",
                str(path),
                "-p",
                self.point.current.project,
                "--profile",
                "runtime",
                "up",
                "-d",
                "--no-deps",
                "--no-build",
                "--pull",
                "never",
                "--force-recreate",
                *self.services,
            ],
            self.execution.VERIFY_RECREATE_TIMEOUT_SECONDS,
            execution=self.execution,
        )
        self.wait()
        self.verify_recovery(
            # Recovery has its own incident outcome. Updating the original
            # job's phase would turn its failed verdict back into in_progress.
            lambda phase, status: logger.info(
                "Dev recovery phase=%s status=%s", phase.value, status.value
            ),
        )
        # A recovery recreates the family from ``recovery-compose.json``: the
        # same retained images and settings, new container ids, and a compose
        # config hash that differs from the regular compose model (the image is
        # the retained tag). ``prepare`` can therefore never re-capture that
        # family (``_capture`` refuses a hash mismatch), so without recording
        # the new ids the snapshot keeps the ids of containers that no longer
        # exist, and the next failed job, even one that failed before touching a
        # container, reads the healthy family as changed and force-recreates it
        # again (OMN-22724, job 54914064 on 2026-11-06 took the dev runtime down
        # this way). A write failure is not a failed recovery: the family is
        # already recovered and verified, and a snapshot that keeps the old ids
        # is the previous behaviour.
        try:
            last_healthy = self.point.current
            if last_healthy is None:
                raise DevBootFailedError("no last healthy dev revision")
            with_new_ids = last_healthy.model_copy(
                update={
                    "container_ids": tuple(
                        sorted(c["Id"] for c in _dev_containers(self))
                    )
                }
            )
            self.execution._atomic_write_private_json(
                self.snapshot_path, with_new_ids.model_dump(mode="json")
            )
            self.point.hold(with_new_ids)
        except (
            OSError,
            ValueError,
            RuntimeError,
            self.execution.subprocess.SubprocessError,
        ):
            logger.warning("Cannot record recovered dev container ids", exc_info=True)

    @contextlib.contextmanager
    def protect(
        self,
        job: str,
        commit: str,
        mark_failed: Callable[[str], object],
    ) -> Iterator[None]:
        """Keep the failed verdict even after successfully recovering the lane."""
        if self.point.current is None:
            raise DevBootFailedError("no last healthy dev revision")
        try:
            yield
            _prove_work_ledger_answers(self.execution)
            _advance_point(self, job, commit)
            _announce_start(self, job, success=True)
        except Exception as exc:
            _announce_start(self, job, success=False)
            # Recovery restores the retained generation, which ran the services
            # the guard was built with, not the ones the failed candidate runs.
            self.services = self._retained_services
            boot_causes: list[str] = []
            # OMN-23431: recovery recreates the failed container, which destroys
            # Docker's probe history; keep it on the incident first.
            health_probes: list[JsonValue] = []
            log_captures: list[dict[str, JsonValue]] = []
            # The kept log text goes to the sink only; the incident file holds the
            # artifact path and the digest, not a second copy of the log.
            captured_logs: dict[str, str] = {}
            try:
                for container in _dev_containers(self):
                    reading = health_probe_reading(container)
                    health_probes.append(reading)
                    summary = health_probe_summary(reading)
                    if summary is not None:
                        boot_causes.append(summary)
                    logs = _checked(
                        ["docker", "logs", "--timestamps", container["Id"]],
                        LOG_READ_SECONDS,
                        execution=self.execution,
                    )
                    tail = "\n".join(logs.splitlines()[-80:])
                    # Some entrypoints summarize the counter as "preflight
                    # exited 1" and log the setting diagnosis separately.
                    boot_causes.extend(
                        line
                        for line in tail.splitlines()
                        if "BIFROST_BASE_CONTRACT_PATH" in line
                        or "ONEX_ACTIVE_RUNTIME_PACKAGES" in line
                    )
                    boot_causes.extend(
                        line for line in tail.splitlines() if "BOOT FAILED" in line
                    )
                    if self.log_byte_cap is not None and summary is not None:
                        encoded = logs.encode("utf-8")
                        bytes_total = len(encoded)
                        raw_text = encoded[-self.log_byte_cap :].decode(
                            "utf-8", errors="replace"
                        )
                        # Logs can hold secrets; only credential-shaped spans are
                        # replaced and every other line is kept.
                        kept_text = redact_credential_spans(raw_text)
                        kept_bytes = kept_text.encode("utf-8")
                        artifact_path = (
                            self.root / f"incident-{job}-{reading['service']}.log"
                        )
                        error: str | None = None
                        try:
                            artifact_path.parent.mkdir(parents=True, exist_ok=True)
                            artifact_path.write_text(kept_text, encoding="utf-8")
                        except OSError as write_error:
                            error = sanitize_error_string(str(write_error))
                        log_captures.append(
                            {
                                "service": reading["service"],
                                "container": reading["container"],
                                "bytes_total": bytes_total,
                                "bytes_kept": len(kept_bytes),
                                "complete": bytes_total <= self.log_byte_cap
                                and kept_text == raw_text,
                                "redacted": kept_text != raw_text,
                                "sha256": hashlib.sha256(kept_bytes).hexdigest(),
                                "artifact_path": str(artifact_path),
                                "error": error,
                            }
                        )
                        captured_logs[str(reading["service"])] = kept_text
            except (
                OSError,
                ValueError,
                RuntimeError,
                self.execution.subprocess.SubprocessError,
            ):
                logger.warning("Cannot read failed dev boot causes")
            reason = _safe(f"{exc}; {'; '.join(dict.fromkeys(boot_causes))}")
            incident: dict[str, JsonValue] = {
                "job": job,
                "commit": commit,
                "error": reason,
                "status": "failed",
                "project": self.config.compose_project,
                "at": self.execution._dev_boot_timestamp(),
                "restored_revision": self.point.current.revision,
                "recovery_verified": False,
                "ledger_pending": True,
                "exception_pending": True,
                "health_probes": health_probes,
            }
            if self.log_byte_cap is not None:
                incident["log_captures"] = cast("JsonValue", log_captures)
            path = self.root / f"incident-{job}.json"
            # Best effort reporting must never get in front of lane recovery.
            try:
                self.execution._atomic_write_private_json(path, incident)
                mark_failed(f"commit={commit}: {reason}")
            except (OSError, RuntimeError):
                logger.exception("Cannot persist failed dev deploy before recovery")
            try:
                unchanged = _unchanged_and_healthy(
                    self.point.current,
                    lambda: _dev_containers(self),
                    execution=self.execution,
                )
                kept_revision: str | None = None
                left_running = False
                if (
                    isinstance(
                        exc, (DevCompanionLaneFailedError, DevOnexApiStepFailedError)
                    )
                    and not unchanged
                    and _all_healthy(
                        lambda: _dev_containers(self), execution=self.execution
                    )
                ):
                    # OMN-21645 (the gateway companion lane) and OMN-24471
                    # (the onex-api step): neither failure is the runtime's,
                    # and both can arrive after the new runtime was replaced
                    # and verified. Rolling that healthy family back recreates
                    # the runtime a second time and ends every in-flight run
                    # with runtime_shutdown. A capture that refuses leaves
                    # kept_revision unset, so the retained family is restored
                    # as before.
                    try:
                        captured = self._capture()
                        self.point.hold(captured)
                        kept_revision = captured.revision
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        self.execution.subprocess.SubprocessError,
                    ):
                        kept_revision = None
                    # OMN-24471: onex-api is not the runtime. A runtime that is
                    # up and healthy is left running even when the family
                    # cannot become the recovery point (a partial roll reads as
                    # mixed revisions): a restore here is the recreate the
                    # onex-api failure never justified. The failure stays
                    # failed and reported; the next roll rolls the rest.
                    left_running = kept_revision is None and isinstance(
                        exc, DevOnexApiStepFailedError
                    )
                if kept_revision is not None:
                    incident["runtime_kept_revision"] = kept_revision
                elif left_running:
                    incident["runtime_left_running"] = True
                elif not unchanged:
                    self._restore()
                incident["recovery_skipped_healthy"] = unchanged
                incident["recovery_verified"] = True
            except (
                OSError,
                ValueError,
                RuntimeError,
                self.execution.subprocess.SubprocessError,
            ) as recovery:
                incident["recovery_error"] = _safe(str(recovery))
                logger.exception("Dev recovery failed for commit=%s", commit)
            try:
                self.execution._atomic_write_private_json(path, incident)
                deliver_reports(
                    self.root, execution=self.execution, config=self.reporting
                )
            except (OSError, RuntimeError):
                logger.exception("Dev deploy failure report remains pending")
            if self.events is not None and self._command is not None:
                try:
                    sunk_incident: Mapping[str, JsonValue] = (
                        {
                            **incident,
                            "log_captures": [
                                {
                                    **capture,
                                    "log": captured_logs.get(
                                        str(capture["service"]), ""
                                    ),
                                }
                                for capture in log_captures
                            ],
                        }
                        if self.log_byte_cap is not None
                        else incident
                    )
                    self.events.incident(
                        sunk_incident,
                        issuer=str(self._command["requested_by"]),
                    )
                except Exception:
                    logger.exception(
                        "Dev boot incident publish failed; recovery already ran"
                    )
            raise DevBootRecoveryError(
                f"commit={commit}: {reason}; forward redeploy revision={self.point.current.revision} "
                f"verified={incident['recovery_verified']}"
            ) from exc


def health_probe_reading(container: ModelDevBootContainer) -> dict[str, JsonValue]:
    """Docker's own record of one container's boot health, as inspect reported it."""
    state = container.get("State", {})
    health = state.get("Health", {})
    declared = container["Config"].get("Healthcheck", {})
    probes: list[JsonValue] = [
        {
            "start": probe.get("Start"),
            "end": probe.get("End"),
            "exit_code": probe.get("ExitCode"),
            "output": " ".join(
                sanitize_error_string(str(probe.get("Output", "")))
                .replace("|", " ")
                .split()
            ),
        }
        for probe in health.get("Log", [])
    ]
    return {
        "service": container["Config"]["Labels"].get("com.docker.compose.service"),
        "container": container["Id"][:12],
        "status": state.get("Status"),
        "started_at": state.get("StartedAt"),
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
        "restart_count": container.get("RestartCount"),
        "health_status": health.get("Status"),
        "failing_streak": health.get("FailingStreak"),
        "healthcheck": {
            "start_period_ns": declared.get("StartPeriod"),
            "interval_ns": declared.get("Interval"),
            "retries": declared.get("Retries"),
            "timeout_ns": declared.get("Timeout"),
        },
        "probes": probes,
    }


def health_probe_summary(reading: Mapping[str, JsonValue]) -> str | None:
    """One line naming a not-healthy container's last probe, or None when healthy."""
    if reading.get("health_status") == "healthy":
        return None
    line = (
        f"{reading.get('service')} health={reading.get('health_status')} "
        f"streak={reading.get('failing_streak')} started={reading.get('started_at')} "
        f"restarts={reading.get('restart_count')} oom={reading.get('oom_killed')}"
    )
    probes = reading.get("probes")
    last = probes[-1] if isinstance(probes, list) and probes else None
    if not isinstance(last, dict):
        return f"{line} last_probe=none"
    return (
        f"{line} last_probe={last.get('start')} exit={last.get('exit_code')} "
        f"{last.get('output')}"
    )


def causes_from_logs(logs: str) -> str:
    causes = [line for line in logs.splitlines() if "BOOT FAILED" in line]
    return (
        causes[-1]
        if causes
        else "container exited or became unhealthy before boot completed"
    )


# The receipt statuses of the bus append command that mean the ledger holds the row. A
# ``spooled`` receipt (the durable bus was down, the row waits in the local outbox) is not one.
_LEDGER_ACCEPTED: Final = frozenset({"accepted", "duplicate", "already-present"})

# One shell, so the environment the launcher gave the agent (the bus append command with its own
# quoting, and the host name) is read by the shell and never by this handler. The row, the lane
# and the request id arrive as positional parameters, never inside the script text.
_APPEND_STATUS_SCRIPT: Final = (
    'eval "append=(${ONEX_LEDGER_BUS_APPEND_COMMAND:?})"; '
    'printf "%s\\n" "$1" | "${append[@]}" --lane "$2" --host "${ONEX_LEDGER_HOST_NAME:?}" '
    '--request-id "$3" --rows-file -'
)


def _append_status_over_bus(
    execution: ModuleType, row: str, *, lane: str, request_id: str
) -> bool:
    """Append one STATUS row through the bus append command (OMN-26507).

    The row goes on the stdin of the command the launcher exports, with ``--lane``, ``--host``,
    ``--request-id`` and ``--rows-file -``; no ledger file path is passed or read. True only when
    the command exited 0 and its receipt says the ledger took the row, so a spooled or refused
    row stays pending for the next delivery.
    """
    try:
        done = execution._run(
            [
                "bash",
                "-c",
                _APPEND_STATUS_SCRIPT,
                "ledger-append",
                row,
                lane,
                request_id,
            ],
            timeout=REPORT_SECONDS,
        )
    except (OSError, execution.subprocess.SubprocessError):
        logger.warning("Dev boot STATUS delivery still pending: append command failed")
        return False
    if done.returncode != 0:
        logger.warning("Dev boot STATUS delivery still pending: append command refused")
        return False
    try:
        receipt = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False
    return isinstance(receipt, dict) and receipt.get("status") in _LEDGER_ACCEPTED


def deliver_reports(
    root: Path, *, execution: ModuleType, config: ModelDevBootReportConfig
) -> None:
    """Retry durable STATUS and exception deliveries independently of the bus."""
    if config.plugin_dir is None:
        logger.warning(
            "Dev boot reports pending: configure DEPLOY_AGENT_REPORT_PLUGIN_DIR"
        )
        return
    plugin = config.plugin_dir
    interpreter = str(config.registry_root / ".onex-dispatch-venv" / "bin" / "python")
    for path in root.glob("incident-*.json"):
        incident = (
            RootModel[dict[str, JsonValue]]
            .model_validate_json(execution._read_private_text(path))
            .root
        )
        evidence = (
            f"commit={incident['commit']} job={incident['job']} error={incident['error']} "
            f"restored_revision={incident['restored_revision']} recovery_verified={incident['recovery_verified']}"
        )
        if incident.get("recovery_error"):
            evidence += f" recovery_error={incident['recovery_error']}"
        request_id = uuid5(
            NAMESPACE_URL, f"deploy-boot:{incident['project']}:{incident['job']}"
        )
        status_row = (
            f"{incident['at']} | STATUS | lane=deploy-agent-{incident['project']} "
            f"| actor=script:deploy-agent | model=none | req={request_id} | {evidence}"
        )
        commands = {
            "exception_pending": [
                interpreter,
                str(plugin / "skills" / "queue-status" / "scripts" / "queue_status.py"),
                "--exception",
                "emit",
                "--class",
                "dev-runtime-boot-failed",
                "--key",
                _json_string(incident["job"]),
                "--source",
                "deploy-agent",
                "--evidence",
                evidence,
                "--tried",
                "forward redeploy of retained healthy dev images and settings",
                "--proposed-fix",
                "repair the failed candidate and redeploy dev",
                "--repo",
                "omnibase_infra",
                "--no-dispatch",
            ],
        }
        if incident["ledger_pending"]:
            if not config.ledger_via_bus:
                logger.warning(
                    "Dev boot STATUS pending: declare ONEX_LEDGER_WRITE_VIA=bus"
                )
            elif _append_status_over_bus(
                execution,
                status_row,
                lane=f"deploy-agent-{incident['project']}",
                request_id=str(request_id),
            ):
                incident["ledger_pending"] = False
                execution._atomic_write_private_json(path, incident)
        for pending, argv in commands.items():
            if not incident[pending]:
                continue
            try:
                result = execution._run(argv, timeout=REPORT_SECONDS)
                acknowledged = result.returncode == 0 and bool(
                    re.search(
                        r"(?m)^EXCEPTION-QUEUE (appended|repeat)\b", result.stdout
                    )
                )
                if acknowledged:
                    incident[pending] = False
                    execution._atomic_write_private_json(path, incident)
            except (OSError, execution.subprocess.SubprocessError):
                logger.warning(
                    "Dev boot incident %s delivery still pending: %s",
                    incident["job"],
                    pending,
                )


def _dev_containers(guard: HandlerDevBootGuard) -> list[ModelDevBootContainer]:
    ids = _checked(
        [
            "docker",
            "ps",
            "--all",
            "-q",
            "--no-trunc",
            "--filter",
            f"label=com.docker.compose.project={guard.config.compose_project}",
            "--filter",
            "label=com.docker.compose.oneoff=False",
        ],
        execution=guard.execution,
    ).split()
    if not ids:
        raise DevBootFailedError("dev runtime containers are absent")
    raw = (
        RootModel[list[ModelDevBootContainer]]
        .model_validate_json(
            _checked(["docker", "inspect", *ids], execution=guard.execution)
        )
        .root
    )
    containers: list[ModelDevBootContainer] = []
    for container in raw:
        labels = container["Config"]["Labels"]
        if labels.get("com.docker.compose.project") != guard.config.compose_project:
            raise DevBootFailedError("container belongs to another compose project")
        if labels.get("com.docker.compose.service") in guard.services:
            containers.append(container)
    found = {c["Config"]["Labels"]["com.docker.compose.service"] for c in containers}
    if found != set(guard.services):
        raise DevBootFailedError(
            f"dev runtime containers absent: {sorted(set(guard.services) - found)}"
        )
    return containers


def _announce(
    events: ProtocolDevRollEvents | None, call: Callable[[ProtocolDevRollEvents], None]
) -> None:
    """Run one announcement; evidence never fails or delays a roll."""
    if events is None:
        return
    try:
        call(events)
    except Exception:
        logger.exception("dev roll event failed friction_type=dev_event_publish_failed")


def _advance_point(guard: HandlerDevBootGuard, job: str, commit: str) -> None:
    """Record the verified family as the recovery point, or keep the one held.

    A roll that reached this call was replaced, verified and Docker healthy. A
    compose config hash that no longer matches (the clone moved after ``up``, see
    :class:`DevBootSettingsDriftError`) says the family cannot be re-rendered
    exactly, not that the deploy failed: rolling a verified family back to the
    previous image recreates the runtime a second time and leaves dev one image
    behind on every such job. The previous recovery point stays held and the drift
    is written beside the incidents for the next reader.
    """
    try:
        guard.point.hold(guard._capture())
    except DevBootSettingsDriftError as drift:
        logger.warning(
            "Dev recovery point not advanced job=%s commit=%s service=%s "
            "container_hash=%s rendered_hash=%s",
            job,
            commit,
            drift.service,
            drift.container_hash,
            drift.rendered_hash,
        )
        record: dict[str, JsonValue] = {
            "job": job,
            "commit": commit,
            "service": drift.service,
            "container_config_hash": drift.container_hash,
            "rendered_config_hash": drift.rendered_hash,
            "kept_recovery_revision": (
                guard.point.current.revision if guard.point.current else ""
            ),
            "at": guard.execution._dev_boot_timestamp(),
        }
        try:
            guard.execution._atomic_write_private_json(
                guard.root / f"settings-drift-{job}.json", record
            )
        except OSError:
            logger.warning("Cannot record dev settings drift", exc_info=True)


def _announce_start(guard: HandlerDevBootGuard, job: str, *, success: bool) -> None:
    if guard.events is None or guard._command is None:
        return
    try:
        raw = runtime_started_at(guard)
        now = datetime.now(UTC)
        events_module = _roll_events_module(guard.execution)
        started = events_module.docker_started_at(raw, fallback=now)
        if raw is not None:
            if guard._prepared_at is not None and started < guard._prepared_at:
                return
        elif not success:
            return
        guard.events.started(
            job, started, events_module.EnumDevStartCause.ROLL_RECREATE
        )
    except Exception:
        logger.exception(
            "Dev boot start announce failed",
            extra={"friction_type": "dev_event_publish_failed"},
        )


def runtime_started_at(guard: HandlerDevBootGuard) -> str | None:
    """The runtime container's Docker start time, or None when unreadable."""
    try:
        for container in _dev_containers(guard):
            service = container["Config"]["Labels"].get("com.docker.compose.service")
            if service != "omninode-runtime":
                continue
            started_at = container.get("State", {}).get("StartedAt")
            if isinstance(started_at, str) and started_at:
                return started_at
            return None
    except (
        OSError,
        ValueError,
        RuntimeError,
        guard.execution.subprocess.SubprocessError,
    ):
        return None
    return None
