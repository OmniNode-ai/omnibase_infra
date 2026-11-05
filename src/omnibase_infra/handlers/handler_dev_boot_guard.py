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
import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import JsonValue, RootModel

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
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)
DEV_BOOT_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
POLL_SECONDS = 5
DOCKER_READ_SECONDS = 10
REPORT_SECONDS = 20


class DevBootFailedError(RuntimeError):
    """A dev runtime failed boot, or cannot be safely replaced."""


class DevBootRecoveryError(RuntimeError):
    """The candidate deploy failed, including when recovery succeeded."""


class DevCompanionLaneFailedError(RuntimeError):
    """A lane deployed beside the runtime failed (the gateway forwarder).

    The companion lane carries no delegation or runtime traffic, so its failure
    fails the job and is reported, but it is not by itself a reason to replace a
    runtime that is up and healthy (OMN-21645). The guard still restores the
    retained family when the runtime is unhealthy or absent.
    """


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


def _safe(text: str) -> str:
    return " ".join(
        sanitize_error_message(RuntimeError(text), max_length=2000)
        .replace("|", " ")
        .split()
    )[:2000]


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
    wait_budget = execution.runtime_health_probe_budget(lane, service)
    command = list(argv)
    index = command.index("--no-deps")
    command[index:index] = ["--wait", "--wait-timeout", str(wait_budget)]
    try:
        result = execution._run(
            command,
            timeout=max(
                timeout, wait_budget + execution.RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS
            ),
            env=execution._compose_env(extra_env),
        )
    except execution.subprocess.TimeoutExpired as exc:
        raise DevBootFailedError(
            f"Runtime roll stopped at {service}: compose up timed out"
        ) from exc
    if result.returncode != 0:
        raise DevBootFailedError(
            f"Runtime roll stopped at {service}: "
            f"{_safe(result.stderr) or 'compose health wait failed'}"
        )
    if not execution.verify_containers_up(
        [service], timeout_s=execution.CONTAINER_VERIFY_TIMEOUT_SECONDS, lane=lane
    )[0]:
        raise DevBootFailedError(
            f"Runtime roll stopped at {service}: container did not reach running state"
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


def folded_dev_services(policy: Mapping[str, str], folded: frozenset[str]) -> set[str]:
    """Retirement uses the same contract mapping as the compose gates.

    A lane that folds nothing (any other dev instance) retires nothing. On the
    lane that folds, the services main carries with no profile of its own
    (``DEV_RUNTIME_MAIN_RETIRED_SERVICES``) are retired with the folded ones.
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
        for service in policy.get("DEV_RUNTIME_MAIN_RETIRED_SERVICES", "").split(",")
        if service.strip()
    }
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", service) for service in retired):
        raise RuntimeError("invalid dev retired service list")
    return {*(mapping[profile] for profile in folded), *retired}


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
    ) -> None:
        self.execution = execution
        self.reporting = reporting
        self.config_hashes = config_hashes
        self.verify_recovery = verify_recovery
        self.config = self.execution.lane_config_for(self.execution.EnumRuntimeLane.DEV)
        self.services = tuple(
            service
            for service in DEV_BOOT_SERVICES
            if service not in self.config.disabled_services
        )
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

    def _containers(self) -> list[ModelDevBootContainer]:
        ids = _checked(
            [
                "docker",
                "ps",
                "--all",
                "-q",
                "--no-trunc",
                "--filter",
                f"label=com.docker.compose.project={self.config.compose_project}",
                "--filter",
                "label=com.docker.compose.oneoff=False",
            ],
            execution=self.execution,
        ).split()
        if not ids:
            raise DevBootFailedError("dev runtime containers are absent")
        raw = (
            RootModel[list[ModelDevBootContainer]]
            .model_validate_json(
                _checked(["docker", "inspect", *ids], execution=self.execution)
            )
            .root
        )
        containers: list[ModelDevBootContainer] = []
        for container in raw:
            labels = container["Config"]["Labels"]
            if labels.get("com.docker.compose.project") != self.config.compose_project:
                raise DevBootFailedError("container belongs to another compose project")
            if labels.get("com.docker.compose.service") in self.services:
                containers.append(container)
        found = {
            c["Config"]["Labels"]["com.docker.compose.service"] for c in containers
        }
        if found != set(self.services):
            raise DevBootFailedError(
                f"dev runtime containers absent: {sorted(set(self.services) - found)}"
            )
        return containers

    def wait(self) -> None:
        """All replicas must be Docker healthy; exhausted boots fail immediately."""
        started = self.execution._dev_boot_clock()
        containers = self._containers()
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
            containers = self._containers()

    def _capture(self) -> ModelDevBootPoint:
        containers = self._containers()
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
                raise DevBootFailedError(
                    f"{service}: compose settings differ from the healthy container"
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

    def prepare(self) -> None:
        """Retain the live healthy family BEFORE git updates compose settings."""
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
            self.point.hold(self._capture())
        except Exception as exc:
            boot_causes: list[str] = []
            try:
                for container in self._containers():
                    logs = _checked(
                        ["docker", "logs", "--tail", "80", container["Id"]],
                        execution=self.execution,
                    )
                    # Some entrypoints summarize the counter as "preflight
                    # exited 1" and log the setting diagnosis separately.
                    boot_causes.extend(
                        line
                        for line in logs.splitlines()
                        if "BIFROST_BASE_CONTRACT_PATH" in line
                        or "ONEX_ACTIVE_RUNTIME_PACKAGES" in line
                    )
                    boot_causes.extend(
                        line for line in logs.splitlines() if "BOOT FAILED" in line
                    )
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
            }
            path = self.root / f"incident-{job}.json"
            # Best effort reporting must never get in front of lane recovery.
            try:
                self.execution._atomic_write_private_json(path, incident)
                mark_failed(f"commit={commit}: {reason}")
            except (OSError, RuntimeError):
                logger.exception("Cannot persist failed dev deploy before recovery")
            try:
                unchanged = _unchanged_and_healthy(
                    self.point.current, self._containers, execution=self.execution
                )
                kept_revision: str | None = None
                if (
                    isinstance(exc, DevCompanionLaneFailedError)
                    and not unchanged
                    and _all_healthy(self._containers, execution=self.execution)
                ):
                    # OMN-21645: the companion lane failed after the new
                    # runtime was replaced and verified. Rolling that healthy
                    # family back recreates the runtime a second time and ends
                    # every in-flight run with runtime_shutdown. A capture that
                    # refuses leaves kept_revision unset, so the retained
                    # family is restored as before.
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
                if kept_revision is not None:
                    incident["runtime_kept_revision"] = kept_revision
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
            raise DevBootRecoveryError(
                f"commit={commit}: {reason}; forward redeploy revision={self.point.current.revision} "
                f"verified={incident['recovery_verified']}"
            ) from exc


def causes_from_logs(logs: str) -> str:
    causes = [line for line in logs.splitlines() if "BOOT FAILED" in line]
    return (
        causes[-1]
        if causes
        else "container exited or became unhealthy before boot completed"
    )


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
            "ledger_pending": [
                interpreter,
                str(plugin / "scripts" / "onex_ledger.py"),
                "--append",
                status_row,
            ],
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
        for pending, argv in commands.items():
            if not incident[pending]:
                continue
            if pending == "ledger_pending" and not config.ledger_via_bus:
                logger.warning(
                    "Dev boot STATUS pending: declare ONEX_LEDGER_WRITE_VIA=bus"
                )
                continue
            try:
                result = execution._run(argv, timeout=REPORT_SECONDS)
                # A ledger relay being queued is NOT an acknowledgement from
                # the ledger. Keep the source incident until it is accepted.
                acknowledged = result.returncode == 0 and bool(
                    re.search(r"(?m)^OK\b", result.stdout)
                    and "queued" not in result.stdout.lower()
                    if pending == "ledger_pending"
                    else re.search(
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
