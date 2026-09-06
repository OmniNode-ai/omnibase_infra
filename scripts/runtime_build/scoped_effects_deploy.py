#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Governed image-only replacement of the existing dev effects service [OMN-17991].

This path never runs the full deploy workflow or rewrites its shared files.
Unchanged, approved container bootstrap still runs when effects is recreated.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from scoped_effects_plan import ModelEffectsDeployPlan

CommandRunner = Callable[[list[str], str | None, int], str]
_PROJECT_LABEL = "com.docker.compose.project"
_SERVICE_LABEL = "com.docker.compose.service"
_WORKDIR_LABEL = "com.docker.compose.project.working_dir"
_FILES_LABEL = "com.docker.compose.project.config_files"
_VERSION_LABEL = "com.docker.compose.version"
_CONFIG_HASH_LABEL = "com.docker.compose.config-hash"
_PROCESS_TERM_GRACE = 15
_IMAGE_CONFIG_FIELDS = (
    "Env",
    "Cmd",
    "Entrypoint",
    "User",
    "WorkingDir",
    "Healthcheck",
    "ExposedPorts",
    "Volumes",
    "StopSignal",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compose_chain_sha256(paths: Sequence[Path]) -> str:
    """Hash the ordered absolute file names and their byte hashes, without secrets."""
    rows = []
    for path in paths:
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("compose chain must contain existing absolute files")
        rows.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return hashlib.sha256(_canonical(rows)).hexdigest()


class CommandProcessUnfencedError(RuntimeError):
    """Recovery must not race an invocation whose process group may survive."""


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=_PROCESS_TERM_GRACE)
    except subprocess.TimeoutExpired:
        pass
    # The CLI can exit while a plugin remains in its process group. Always kill
    # the surviving group, not only the original direct child's PID.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise CommandProcessUnfencedError(
            "command process group could not be reaped; automatic recovery is unsafe"
        ) from exc


def _run_command(args: list[str], stdin: str | None, timeout: int) -> str:
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"{args[0]} {args[1]} could not be started") from exc
    try:
        stdout, _ = process.communicate(input=stdin, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError) as exc:
        try:
            _stop_process_group(process)
        except BaseException as fencing_error:
            raise CommandProcessUnfencedError(
                "command process group termination is unproved; recovery requires operator inspection"
            ) from fencing_error
        raise RuntimeError(
            f"{args[0]} {args[1]} interrupted or timed out; its process group was terminated"
        ) from exc
    if process.returncode:
        # Compose diagnostics can contain resolved credentials. Preserve the exit
        # status, never copy arbitrary stdout/stderr into receipts or terminal logs.
        raise RuntimeError(f"{args[0]} {args[1]} failed with exit {process.returncode}")
    return stdout


def _objects(text: str) -> list[dict[str, Any]]:
    value = json.loads(text)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RuntimeError("Docker returned an invalid object inventory")
    return value


def _compose(
    plan: ModelEffectsDeployPlan,
    override: Path | None = None,
    snapshot: Path | None = None,
) -> list[str]:
    args = [
        "docker",
        "compose",
        "--project-directory",
        str(plan.compose_working_dir),
        "-p",
        plan.compose_project,
    ]
    sources = (snapshot,) if snapshot is not None else plan.compose_files
    for path in (*sources, *((override,) if override else ())):
        args.extend(("-f", str(path)))
    return [*args, "--profile", "runtime"]


def _project(
    plan: ModelEffectsDeployPlan, runner: CommandRunner
) -> list[dict[str, Any]]:
    ids = runner(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label={_PROJECT_LABEL}={plan.compose_project}",
            "--format",
            "{{.ID}}",
        ],
        None,
        30,
    ).split()
    if not ids:
        raise RuntimeError("active compose project has no containers")
    containers = _objects(runner(["docker", "inspect", *ids], None, 30))
    if any(
        item.get("Config", {}).get("Labels", {}).get(_PROJECT_LABEL)
        != plan.compose_project
        for item in containers
    ):
        raise RuntimeError("container inventory crossed the approved compose project")
    return containers


def _service(containers: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [
        item
        for item in containers
        if item.get("Config", {}).get("Labels", {}).get(_SERVICE_LABEL) == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one existing container for service {name}"
        )
    return matches[0]


def _healthy(container: Mapping[str, Any]) -> bool:
    state = container.get("State", {})
    return (
        state.get("Running") is True
        and state.get("Status") == "running"
        and not any(state.get(key, False) for key in ("Paused", "Restarting", "Dead"))
        and state.get("Health", {}).get("Status") == "healthy"
    )


def _snapshot(containers: Sequence[dict[str, Any]], service: str) -> dict[str, object]:
    return {
        item["Id"]: {
            "image": item["Image"],
            "started_at": item["State"]["StartedAt"],
            "running": item["State"].get("Running"),
            "status": item["State"].get("Status"),
            "paused": bool(item["State"].get("Paused", False)),
            "restarting": bool(item["State"].get("Restarting", False)),
            "dead": bool(item["State"].get("Dead", False)),
            "health": item["State"].get("Health", {}).get("Status"),
            "exit_code": item["State"].get("ExitCode"),
            "mounts": sorted(
                item.get("Mounts", []), key=lambda mount: mount["Destination"]
            ),
        }
        for item in containers
        if item["Config"]["Labels"][_SERVICE_LABEL] != service
    }


def _runtime_settings(container: Mapping[str, Any]) -> dict[str, object]:
    config = container.get("Config", {})
    return {
        "config": {key: config.get(key) for key in _IMAGE_CONFIG_FIELDS},
        "host": {
            key: value
            for key, value in container.get("HostConfig", {}).items()
            if key not in {"ContainerIDFile", "Binds"}
        },
        "mounts": sorted(
            container.get("Mounts", []), key=lambda mount: mount["Destination"]
        ),
        "networks": sorted(container.get("NetworkSettings", {}).get("Networks", {})),
    }


def _assert_active_config(
    plan: ModelEffectsDeployPlan, target: Mapping[str, Any], rendered: Mapping[str, Any]
) -> None:
    """Refuse critical rendered-config drift before considering image replacement."""
    config = target["Config"]
    desired = rendered["services"][plan.service]
    _assert_mutable_resources(target.get("HostConfig", {}), desired)
    environment = dict(
        entry.split("=", 1) for entry in config.get("Env", []) if "=" in entry
    )
    for key, value in desired.get("environment", {}).items():
        if value is None:
            if key in environment:
                raise RuntimeError(
                    "active effects environment differs from rendered compose"
                )
        elif environment.get(key) != str(value):
            raise RuntimeError(
                "active effects environment differs from rendered compose"
            )
    for compose_key, inspect_key in (
        ("command", "Cmd"),
        ("entrypoint", "Entrypoint"),
        ("user", "User"),
        ("working_dir", "WorkingDir"),
    ):
        if compose_key in desired and desired[compose_key] != config.get(inspect_key):
            raise RuntimeError(
                f"active effects {compose_key} differs from rendered compose"
            )
    for compose_key, inspect_key in (
        ("privileged", "Privileged"),
        ("read_only", "ReadonlyRootfs"),
        ("init", "Init"),
    ):
        if bool(desired.get(compose_key, False)) != bool(
            target.get("HostConfig", {}).get(inspect_key, False)
        ):
            raise RuntimeError(
                f"active effects {compose_key} differs from rendered compose"
            )
    if desired.get("configs") or desired.get("secrets") or desired.get("tmpfs"):
        raise RuntimeError(
            "unsupported effects mount source requires explicit compatibility verification"
        )
    expected_mounts = []
    for volume in desired.get("volumes", []):
        kind = volume.get("type")
        source = volume.get("source")
        if kind == "volume":
            source = rendered.get("volumes", {}).get(source, {}).get("name")
        if kind not in {"bind", "volume"} or not source:
            raise RuntimeError(
                "unresolved effects volume prevents image-only verification"
            )
        expected_mounts.append(
            (kind, source, volume["target"], not volume.get("read_only", False))
        )
    actual_mounts = [
        (
            mount["Type"],
            mount.get("Name") if mount["Type"] == "volume" else mount["Source"],
            mount["Destination"],
            mount["RW"],
        )
        for mount in target.get("Mounts", [])
    ]
    if sorted(expected_mounts) != sorted(actual_mounts):
        raise RuntimeError("active effects mounts differ from rendered compose")
    desired_networks = desired.get("networks", {})
    networks = {
        rendered.get("networks", {}).get(name, {}).get("name")
        for name in desired_networks
    }
    if None in networks or networks != set(
        target.get("NetworkSettings", {}).get("Networks", {})
    ):
        raise RuntimeError("active effects networks differ from rendered compose")


def _byte_count(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise RuntimeError("unsupported memory limit representation")
    match = re.fullmatch(r"(-?[0-9]+(?:\.[0-9]+)?)\s*([kmgtp]?)(?:i?b)?", value.lower())
    if not match:
        raise RuntimeError("unsupported memory limit units")
    return int(Decimal(match[1]) * (1024 ** " kmgtp".index(match[2] or " ")))


def _assert_mutable_resources(
    host: Mapping[str, Any], desired: Mapping[str, Any]
) -> None:
    """Compose hashes are creation receipts: `docker update` does not update them.

    Normalize only update-mutable Linux resource/restart fields, following
    Compose's getDeployResources/getRestartPolicy mappings. Unsupported complex
    blkio settings refuse rather than pretending to preserve them.
    """
    resources = desired.get("deploy", {}).get("resources", {})
    limits = resources.get("limits", {})
    reservations = resources.get("reservations", {})
    memory = _byte_count(limits.get("memory") or desired.get("mem_limit", 0))
    swap = _byte_count(desired.get("memswap_limit", 0))
    if swap == 0 and memory > 0:
        swap = memory * 2  # Docker's documented unset-swap total memory allowance.
    cpus = limits.get("cpus") or desired.get("cpus", 0)
    deploy_pids = limits.get("pids", 0)
    if not isinstance(deploy_pids, int) or isinstance(deploy_pids, bool):
        raise RuntimeError("unsupported mutable PIDs representation")
    expected: dict[str, Any] = {
        "NanoCpus": int(Decimal(str(cpus)) * 1_000_000_000),
        "Memory": memory,
        "MemoryReservation": _byte_count(
            reservations.get("memory") or desired.get("mem_reservation", 0)
        ),
        "MemorySwap": swap,
        "CpusetCpus": desired.get("cpuset", ""),
        "CpusetMems": "",  # Compose has no cpuset_mems service setting.
        "KernelMemory": 0,
        "BlkioWeight": desired.get("blkio_config", {}).get("weight", 0),
        # Compose setLimits overrides the service value only for positive PIDs.
        "PidsLimit": deploy_pids if deploy_pids > 0 else desired.get("pids_limit", 0),
    }
    for field, compose_key in (
        ("CpuPeriod", "cpu_period"),
        ("CpuQuota", "cpu_quota"),
        ("CpuRealtimePeriod", "cpu_rt_period"),
        ("CpuRealtimeRuntime", "cpu_rt_runtime"),
        ("CpuShares", "cpu_shares"),
        ("CpuCount", "cpu_count"),
    ):
        value = desired.get(compose_key, 0)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"unsupported mutable CPU representation: {compose_key}")
        expected[field] = value
    expected["CpuPercent"] = int(Decimal(str(desired.get("cpu_percent", 0))) * 100)
    for field, wanted in expected.items():
        actual = host.get(field)
        if actual is None:
            actual = "" if isinstance(wanted, str) else 0
        if actual != wanted:
            raise RuntimeError(
                f"live mutable effects setting differs from rendered Compose: {field}"
            )
    swappiness = desired.get("mem_swappiness") or None
    actual_swappiness = host.get("MemorySwappiness")
    if actual_swappiness == -1:
        actual_swappiness = None  # Engine sentinel for inherited host policy.
    if actual_swappiness != swappiness or bool(host.get("OomKillDisable")) != bool(
        desired.get("oom_kill_disable", False)
    ):
        raise RuntimeError(
            "live mutable effects memory policy differs from rendered Compose"
        )
    blkio = desired.get("blkio_config", {})
    if any(value for key, value in blkio.items() if key != "weight") or any(
        host.get(key)
        for key in (
            "BlkioWeightDevice",
            "BlkioDeviceReadBps",
            "BlkioDeviceWriteBps",
            "BlkioDeviceReadIOps",
            "BlkioDeviceWriteIOps",
        )
    ):
        raise RuntimeError(
            "device-specific blkio configuration requires separate preservation proof"
        )
    restart, _, retries = str(desired.get("restart", "no")).partition(":")
    policy = desired.get("deploy", {}).get("restart_policy")
    if policy is not None:
        restart, retries = (
            policy.get("condition", "no"),
            str(policy.get("max_attempts", 0)),
        )
    restart = {"none": "no", "any": "always", "": "no"}.get(restart, restart)
    actual_policy = host.get("RestartPolicy") or {}
    if (
        actual_policy.get("Name") or "no",
        actual_policy.get("MaximumRetryCount", 0),
    ) != (restart, int(retries or 0)):
        raise RuntimeError(
            "live mutable effects restart policy differs from rendered Compose"
        )


def _preflight(
    plan: ModelEffectsDeployPlan, runner: CommandRunner
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if compose_chain_sha256(plan.compose_files) != plan.expected_compose_sha256:
        raise RuntimeError("compose chain bytes differ from the approved plan")
    containers = _project(plan, runner)
    target = _service(containers, plan.service)
    if (
        target["Id"] != plan.expected_container_id
        or target["Image"] != plan.expected_image_id
    ):
        raise RuntimeError("active effects identity differs from the approved plan")
    if not _healthy(target):
        raise RuntimeError("current effects service must already be healthy")
    labels = target["Config"]["Labels"]
    if labels.get(_WORKDIR_LABEL) != str(plan.compose_working_dir):
        raise RuntimeError(
            "active compose working directory differs from the approved plan"
        )
    if labels.get(_FILES_LABEL, "").split(",") != [
        str(path) for path in plan.compose_files
    ]:
        raise RuntimeError("active compose file chain differs from the approved plan")
    version = runner(["docker", "compose", "version", "--short"], None, 30).strip()
    if not version or version != labels.get(_VERSION_LABEL):
        raise RuntimeError(
            "Compose CLI version differs from the active container's creation version"
        )
    if _compose_service_hash(plan, runner) != labels.get(_CONFIG_HASH_LABEL):
        raise RuntimeError(
            "complete rendered effects configuration differs from its active Compose hash"
        )
    rendered = json.loads(
        runner([*_compose(plan), "config", "--format", "json"], None, 30)
    )
    if not isinstance(rendered, dict) or rendered.get("name") != plan.compose_project:
        raise RuntimeError("rendered compose project differs from the approved plan")
    desired = rendered["services"][plan.service]
    if (
        desired.get("scale", 1) != 1
        or desired.get("deploy", {}).get("replicas", 1) != 1
    ):
        raise RuntimeError("effects-only rollout requires exactly one replica")
    _assert_active_config(plan, target, _unescape_render(rendered))
    _assert_dependencies(plan, containers, rendered)
    return containers, rendered


def _unescape_render(value: Any) -> Any:
    # `compose config` escapes every dollar for safe reparsing. Compare actual
    # container values after decoding this one rendering layer; retain escaped
    # values in the private snapshot that Compose itself will parse again.
    if isinstance(value, str):
        return value.replace("$$", "$")
    if isinstance(value, dict):
        return {key: _unescape_render(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unescape_render(item) for item in value]
    return value


def _compose_service_hash(
    plan: ModelEffectsDeployPlan, runner: CommandRunner, snapshot: Path | None = None
) -> str:
    fields = runner(
        [*_compose(plan, snapshot=snapshot), "config", "--hash", plan.service], None, 30
    ).split()
    if (
        len(fields) != 2
        or fields[0] != plan.service
        or not re.fullmatch(r"[0-9a-f]{64}", fields[1])
    ):
        raise RuntimeError("Compose service hash is unavailable or unsupported")
    return fields[1]


def _assert_dependencies(
    plan: ModelEffectsDeployPlan,
    containers: Sequence[dict[str, Any]],
    rendered: Mapping[str, Any],
) -> None:
    for service, settings in (
        rendered["services"][plan.service].get("depends_on", {}).items()
    ):
        dependency = _service(containers, service)
        condition = settings.get("condition", "service_started")
        state = dependency["State"]
        if condition == "service_healthy":
            ready = _healthy(dependency)
        elif condition == "service_started":
            ready = state.get("Running") is True
        elif condition == "service_completed_successfully":
            ready = state.get("Status") == "exited" and state.get("ExitCode") == 0
        else:
            raise RuntimeError("unsupported compose dependency condition")
        if not ready:
            raise RuntimeError(
                f"existing dependency {service} does not satisfy its declared condition"
            )


def _probe(
    runner: CommandRunner,
    script: str,
    *,
    image: str | None = None,
    container: str | None = None,
) -> dict[str, Any]:
    if image is not None:
        args = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "uv",
            image,
        ]
    elif container is not None:
        args = ["docker", "exec", "-i", container, "uv"]
    else:
        raise RuntimeError("probe requires a specific image or container")
    args.extend(
        (
            "run",
            "--no-project",
            "--no-sync",
            "--python",
            "/app/.venv/bin/python",
            "python",
            "-",
        )
    )
    value = json.loads(runner(args, script, 120))
    if not isinstance(value, dict) or not isinstance(value.get("source_pins"), dict):
        raise RuntimeError("image probe returned incomplete provenance")
    for key in (
        "dependency_fingerprint",
        "shared_runtime_fingerprint",
        "market_payload_fingerprint",
    ):
        if not isinstance(value.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[key]
        ):
            raise RuntimeError("image probe returned invalid content fingerprints")
    if not isinstance(value.get("infra_source_ref"), str) or not re.fullmatch(
        r"(?:[0-9a-f]{12}|[0-9a-f]{40})", value["infra_source_ref"]
    ):
        raise RuntimeError("image probe returned invalid infra source provenance")
    if "base_image_id" not in value or (
        value["base_image_id"] is not None
        and (
            not isinstance(value["base_image_id"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["base_image_id"])
        )
    ):
        raise RuntimeError("image probe returned invalid base image provenance")
    topics = value.get("required_topics")
    if (
        not isinstance(topics, list)
        or not topics
        or any(not isinstance(topic, str) or not topic for topic in topics)
        or len(topics) != len(set(topics))
    ):
        raise RuntimeError("image probe returned an invalid required topic inventory")
    return value


def _hotpatch_preflight(
    plan: ModelEffectsDeployPlan,
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, str]:
    if "HOTPATCH_PREFLIGHT_BYPASS" in os.environ:
        raise RuntimeError("scoped effects rollout refuses HOTPATCH_PREFLIGHT_BYPASS")
    if not plan.hotpatch_ledger.is_file() or not plan.source_clones_root.is_dir():
        raise RuntimeError("required hotpatch ledger or source clone root is absent")
    refs = {"omnibase_infra": plan.infra_source_sha, **plan.source_pins}
    for repo, expected in refs.items():
        clone = plan.source_clones_root / repo
        if not clone.is_dir():
            raise RuntimeError(f"required source clone is absent: {repo}")
        source = candidate["infra_source_ref"] if repo == "omnibase_infra" else expected
        resolved = runner(
            ["git", "-C", str(clone), "rev-parse", "--verify", f"{source}^{{commit}}"],
            None,
            30,
        ).strip()
        if resolved != expected:
            raise RuntimeError(
                f"candidate source does not uniquely resolve to the approved pin: {repo}"
            )
    name = target.get("Name", "").removeprefix("/")
    if not name:
        raise RuntimeError(
            "effects container name is unavailable for the hotpatch gate"
        )
    ledger = yaml.safe_load(plan.hotpatch_ledger.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or not isinstance(ledger.get("rows"), list):
        raise RuntimeError("hotpatch ledger has no valid row inventory")
    for row in ledger["rows"]:
        if not isinstance(row, dict):
            raise RuntimeError("hotpatch ledger contains an invalid row")
        if (
            row.get("container") == name
            and row.get("status", "active") != "reconciled"
            and row.get("source_repo") not in refs
        ):
            raise RuntimeError(
                "effects hotpatch source lacks an explicit candidate pin; host HEAD fallback is forbidden"
            )
    repo_root = Path(__file__).resolve().parents[2]
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "--project",
        str(repo_root),
        "python",
        str(repo_root / "scripts" / "preflight_hotpatch_ledger.py"),
        "--container",
        name,
        "--ledger",
        str(plan.hotpatch_ledger),
        "--clones-root",
        str(plan.source_clones_root),
    ]
    for repo, ref in sorted(refs.items()):
        command.extend(("--build-ref", f"{repo}={ref}"))
    runner(command, None, 120)
    return {
        "ledger_sha256": hashlib.sha256(plan.hotpatch_ledger.read_bytes()).hexdigest(),
        "container": name,
        "infra_source_sha": plan.infra_source_sha,
    }


def _assert_topics(
    containers: Sequence[dict[str, Any]], required: list[str], runner: CommandRunner
) -> None:
    broker = _service(containers, "redpanda")
    if not _healthy(broker):
        raise RuntimeError("existing project broker is not healthy")
    raw = runner(
        [
            "docker",
            "exec",
            broker["Id"],
            "rpk",
            "topic",
            "list",
            "-X",
            "brokers=redpanda:9092",
        ],
        None,
        30,
    )
    rows = [line.split() for line in raw.splitlines() if line.strip()]
    if not rows or rows[0][:2] != ["NAME", "PARTITIONS"]:
        raise RuntimeError("broker topic metadata format is unrecognized")
    observed: set[str] = set()
    for row in rows[1:]:
        if len(row) < 3 or not row[1].isdigit() or not row[2].isdigit():
            raise RuntimeError("broker topic metadata contains an invalid row")
        observed.add(row[0])
    missing = set(required).difference(observed)
    if missing:
        raise RuntimeError(
            f"required broker topics are absent ({len(missing)}); scoped rollout cannot provision them"
        )


def _write(path: Path, payload: object, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _wait_target(
    plan: ModelEffectsDeployPlan,
    image: str,
    runner: CommandRunner,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    deadline = monotonic() + plan.health_timeout_seconds
    while True:
        target = _service(_project(plan, runner), plan.service)
        if target["Image"] != image:
            raise RuntimeError(
                "recreated effects container is not the intended immutable image"
            )
        if _healthy(target):
            return target
        if monotonic() >= deadline:
            raise RuntimeError(
                "effects health did not become healthy within the approved bound"
            )
        sleep(min(1.0, max(0.0, deadline - monotonic())))


def _recreate(
    plan: ModelEffectsDeployPlan,
    override: Path,
    runner: CommandRunner,
    snapshot: Path,
    snapshot_sha: str,
    image: str,
) -> None:
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != snapshot_sha:
        raise RuntimeError(
            "private admitted Compose snapshot changed; recreation refuses"
        )
    if json.loads(override.read_text(encoding="utf-8")) != {
        "services": {plan.service: {"image": image, "pull_policy": "never"}}
    }:
        raise RuntimeError(
            "private image override differs from the approved immutable image"
        )
    runner(
        [
            *_compose(plan, override, snapshot),
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--no-build",
            "--pull",
            "never",
            plan.service,
        ],
        None,
        plan.health_timeout_seconds,
    )


def _assert_unchanged(
    plan: ModelEffectsDeployPlan,
    before: Sequence[dict[str, Any]],
    target_before: Mapping[str, Any],
    runner: CommandRunner,
    override: Path,
    snapshot: Path,
    snapshot_sha: str,
    verified_target: Mapping[str, Any],
    rendered: Mapping[str, Any],
) -> None:
    after = _project(plan, runner)
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != snapshot_sha:
        raise RuntimeError("private admitted Compose snapshot changed during rollout")
    if _snapshot(before, plan.service) != _snapshot(after, plan.service):
        raise RuntimeError(
            "non-target container identities, images, start times or mounts changed"
        )
    target = _service(after, plan.service)
    if (
        target["Id"] != verified_target["Id"]
        or target["Image"] != verified_target["Image"]
        or not _healthy(target)
    ):
        raise RuntimeError(
            "final effects identity/image/health differs from the installed-code readback"
        )
    _assert_dependencies(plan, after, rendered)
    if _runtime_settings(target_before) != _runtime_settings(target):
        raise RuntimeError("effects runtime settings changed beyond the image")
    labels = target["Config"]["Labels"]
    if labels.get(_WORKDIR_LABEL) != str(plan.compose_working_dir) or labels.get(
        _FILES_LABEL, ""
    ).split(",") != [str(snapshot), str(override)]:
        raise RuntimeError(
            "recreated effects compose ownership differs from the scoped invocation"
        )
    if compose_chain_sha256(plan.compose_files) != plan.expected_compose_sha256:
        raise RuntimeError("shared compose files changed during the scoped rollout")


def _assert_lock(lock: Path, identity: tuple[int, int]) -> None:
    try:
        stat = lock.stat()
        owner = (lock / "pid").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("scoped deployment lock ownership was lost") from exc
    if (
        lock.is_symlink()
        or (stat.st_dev, stat.st_ino) != identity
        or owner != str(os.getpid())
    ):
        raise RuntimeError("scoped deployment lock ownership changed")


def _freeze_render(rendered: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(rendered))
    # Compose up can ensure project networks/volumes even with --no-deps.
    # External identities prohibit this scoped path from creating either.
    for category in ("networks", "volumes"):
        for key, definition in frozen.get(category, {}).items():
            if not isinstance(definition, dict) or not definition.get("name"):
                raise RuntimeError(
                    "unresolved shared network/volume identity prevents frozen rollout"
                )
            frozen[category][key] = {"name": definition["name"], "external": True}
    return frozen


def run_deploy(
    plan: ModelEffectsDeployPlan,
    *,
    execute: bool = False,
    state_root: Path | None = None,
    runner: CommandRunner = _run_command,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Preview by default; serialize, prove compatibility, replace and verify on execute."""
    state_root = (
        state_root if state_root is not None else Path.home() / ".omnibase" / "infra"
    )
    if "HOTPATCH_PREFLIGHT_BYPASS" in os.environ:
        raise RuntimeError("scoped effects rollout refuses HOTPATCH_PREFLIGHT_BYPASS")
    lock = state_root / ".deploy.lock"
    if lock.exists():
        raise RuntimeError(
            "deployment lock already exists; scoped executor never steals it"
        )
    before, rendered_before = _preflight(plan, runner)
    target_before = _service(before, plan.service)
    if not execute:
        return {
            "status": "PREVIEW",
            "compatibility_verified": False,
            "ticket_id": plan.ticket_id,
            "current_image_id": plan.expected_image_id,
            "candidate_image_id": plan.candidate_image_id,
        }

    state_root.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("deployment lock was acquired by another owner") from exc
    pid_file = lock / "pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    lock_stat = lock.stat()
    lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
    run_id = uuid4().hex
    record_dir = state_root / "scoped-effects" / plan.ticket_id / run_id
    receipt: dict[str, Any] = {
        "schema_version": "1",
        "ticket_id": plan.ticket_id,
        "reason": plan.reason,
        "status": "REFUSED",
        "rollback_verified": False,
        "record_dir": str(record_dir),
        "started_at": datetime.now(UTC).isoformat(),
        "actor": {
            "uid": os.getuid(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "argv": sys.argv,
        },
        "expected_container_id": plan.expected_container_id,
        "previous_image_id": plan.expected_image_id,
        "candidate_image_id": plan.candidate_image_id,
        "compose_sha256": plan.expected_compose_sha256,
        "source_pins": dict(plan.source_pins),
    }
    mutation_attempted = False
    observed_candidate_id: str | None = None
    live_before: dict[str, Any] | None = None
    try:
        _assert_lock(lock, lock_identity)
        record_dir.mkdir(parents=True, mode=0o700)
        _write(record_dir / "plan.json", plan.model_dump(mode="json"))
        script = (
            Path(__file__)
            .with_name("scoped_image_probe.py")
            .read_text(encoding="utf-8")
        )
        images = []
        for image in (plan.expected_image_id, plan.candidate_image_id):
            found = _objects(runner(["docker", "image", "inspect", image], None, 30))
            if len(found) != 1 or found[0].get("Id") != image:
                raise RuntimeError(
                    "local immutable image identity could not be verified"
                )
            images.append(found[0])
        if {
            key: images[0].get("Config", {}).get(key) for key in _IMAGE_CONFIG_FIELDS
        } != {
            key: images[1].get("Config", {}).get(key) for key in _IMAGE_CONFIG_FIELDS
        }:
            raise RuntimeError(
                "candidate image changes runtime configuration beyond installed code"
            )
        layers = [image.get("RootFS", {}).get("Layers") for image in images]
        if any(
            not isinstance(layer_list, list)
            or not layer_list
            or any(
                not isinstance(layer, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer)
                for layer in layer_list
            )
            for layer_list in layers
        ):
            raise RuntimeError("immutable image filesystem ancestry is unavailable")
        if len(layers[1]) <= len(layers[0]) or layers[1][: len(layers[0])] != layers[0]:
            raise RuntimeError(
                "candidate filesystem does not derive from the exact prior image"
            )
        expected_environment = dict(
            entry.split("=", 1)
            for entry in images[0].get("Config", {}).get("Env", [])
            if "=" in entry
        )
        for key, value in (
            _unescape_render(rendered_before)["services"][plan.service]
            .get("environment", {})
            .items()
        ):
            if value is None:
                expected_environment.pop(key, None)
            else:
                expected_environment[key] = str(value)
        if expected_environment != dict(
            entry.split("=", 1)
            for entry in target_before["Config"].get("Env", [])
            if "=" in entry
        ):
            raise RuntimeError(
                "active effects environment contains drift outside rendered compose/image defaults"
            )
        baseline = _probe(runner, script, image=plan.expected_image_id)
        candidate = _probe(runner, script, image=plan.candidate_image_id)
        if candidate["base_image_id"] != plan.expected_image_id:
            raise RuntimeError(
                "candidate provenance does not name the exact current base image"
            )
        if candidate["source_pins"] != plan.source_pins:
            raise RuntimeError(
                "candidate source provenance differs from approved immutable pins"
            )
        for key in (
            "dependency_fingerprint",
            "shared_runtime_fingerprint",
            "required_topics",
        ):
            if candidate[key] != baseline[key]:
                raise RuntimeError(f"candidate {key} differs from the current image")
        current, rendered_current = _preflight(plan, runner)
        if _snapshot(before, plan.service) != _snapshot(current, plan.service):
            raise RuntimeError(
                "non-target state changed during image compatibility checks"
            )
        live_before = _probe(runner, script, container=plan.expected_container_id)
        if live_before["dependency_fingerprint"] != baseline["dependency_fingerprint"]:
            raise RuntimeError(
                "live effects dependencies differ from its immutable image"
            )
        if (
            live_before["market_payload_fingerprint"]
            != baseline["market_payload_fingerprint"]
        ):
            raise RuntimeError(
                "live Market payload differs from its immutable image; unrecorded patch may be lost"
            )
        if live_before["required_topics"] != baseline["required_topics"]:
            raise RuntimeError(
                "live effects topic requirements differ from its immutable image"
            )
        _assert_topics(current, candidate["required_topics"], runner)
        retention_tag = (
            f"onex-scoped-effects-rollback:{plan.ticket_id.lower()}-{run_id}"
        )
        if runner(
            [
                "docker",
                "image",
                "ls",
                "--quiet",
                "--filter",
                f"reference={retention_tag}",
            ],
            None,
            30,
        ).strip():
            raise RuntimeError("rollback retention reference already exists")
        runner(["docker", "tag", plan.expected_image_id, retention_tag], None, 30)
        receipt["rollback_image_ref"] = retention_tag
        receipt["baseline_evidence"] = baseline
        receipt["candidate_evidence"] = candidate
        override = record_dir / "candidate.compose.json"
        _write(
            override,
            {
                "services": {
                    plan.service: {
                        "image": plan.candidate_image_id,
                        "pull_policy": "never",
                    }
                }
            },
            mode=0o400,
        )
        rollback_override = record_dir / "rollback.compose.json"
        _write(
            rollback_override,
            {
                "services": {
                    plan.service: {
                        "image": plan.expected_image_id,
                        "pull_policy": "never",
                    }
                }
            },
            mode=0o400,
        )
        # Final compare under the same exclusion lock, immediately before effects.
        current, _ = _preflight(plan, runner)
        if _snapshot(before, plan.service) != _snapshot(current, plan.service):
            raise RuntimeError("non-target state changed before recreation")
        if _runtime_settings(target_before) != _runtime_settings(
            _service(current, plan.service)
        ):
            raise RuntimeError("effects settings changed during compatibility checks")
        receipt["hotpatch_evidence"] = _hotpatch_preflight(
            plan, candidate, _service(current, plan.service), runner
        )
        current, rendered_current = _preflight(plan, runner)
        if _snapshot(before, plan.service) != _snapshot(
            current, plan.service
        ) or _runtime_settings(target_before) != _runtime_settings(
            _service(current, plan.service)
        ):
            raise RuntimeError("runtime state changed during the hotpatch preflight")
        if rendered_current != rendered_before:
            raise RuntimeError(
                "fully resolved Compose configuration changed during admission"
            )
        frozen = _freeze_render(rendered_current)
        snapshot = record_dir / "admitted.compose.json"
        _write(snapshot, frozen, mode=0o400)
        snapshot_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if (
            _compose_service_hash(plan, runner, snapshot)
            != target_before["Config"]["Labels"][_CONFIG_HASH_LABEL]
        ):
            raise RuntimeError(
                "frozen Compose snapshot does not reproduce the active effects configuration"
            )
        receipt["private_compose_snapshot_sha256"] = snapshot_sha
        _assert_lock(lock, lock_identity)
        _write(
            record_dir / "intent.json",
            {
                **receipt,
                "status": "PREPARED",
                "non_target_snapshot": _snapshot(before, plan.service),
            },
        )
        mutation_attempted = True
        _recreate(
            plan, override, runner, snapshot, snapshot_sha, plan.candidate_image_id
        )
        target = _wait_target(plan, plan.candidate_image_id, runner, sleep, monotonic)
        observed_candidate_id = target["Id"]
        live_after = _probe(runner, script, container=target["Id"])
        if (
            live_after["source_pins"] != plan.source_pins
            or live_after["dependency_fingerprint"]
            != candidate["dependency_fingerprint"]
        ):
            raise RuntimeError(
                "installed effects source/dependencies differ from the candidate proof"
            )
        if (
            live_after["market_payload_fingerprint"]
            != candidate["market_payload_fingerprint"]
        ):
            raise RuntimeError(
                "installed effects Market payload differs from the candidate proof"
            )
        if (
            live_after["shared_runtime_fingerprint"]
            != live_before["shared_runtime_fingerprint"]
            or live_after["required_topics"] != live_before["required_topics"]
        ):
            raise RuntimeError("mounted runtime material changed during replacement")
        _assert_unchanged(
            plan,
            before,
            target_before,
            runner,
            override,
            snapshot,
            snapshot_sha,
            target,
            rendered_current,
        )
        _assert_lock(lock, lock_identity)
        receipt.update(
            status="PASSED",
            container_id=target["Id"],
            installed_evidence=live_after,
            verified_at=datetime.now(UTC).isoformat(),
        )
    except BaseException as exc:  # noqa: BLE001 — transaction boundary must adjudicate interrupted/unknown failures
        receipt["error"] = (
            str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        )
        if mutation_attempted:
            try:
                # A timeout can occur after Docker accepted recreation. Only
                # restore our candidate or the still-original target; never
                # overwrite a foreign replacement or use changed shared files.
                if isinstance(exc, CommandProcessUnfencedError):
                    raise CommandProcessUnfencedError(
                        "forward process group may remain active; automatic rollback withheld"
                    )
                _assert_lock(lock, lock_identity)
                targets = [
                    item
                    for item in _project(plan, runner)
                    if item["Config"]["Labels"][_SERVICE_LABEL] == plan.service
                ]
                if len(targets) > 1:
                    raise RuntimeError(
                        "effects ownership is ambiguous; automatic rollback refuses"
                    )
                is_original = False
                is_ours = False
                if targets:
                    current_target = targets[0]
                    labels = current_target["Config"]["Labels"]
                    is_original = (
                        current_target["Id"] == plan.expected_container_id
                        and current_target["Image"] == plan.expected_image_id
                    )
                    is_ours = (
                        current_target["Image"] == plan.candidate_image_id
                        and (
                            observed_candidate_id is None
                            or current_target["Id"] == observed_candidate_id
                        )
                        and labels.get(_WORKDIR_LABEL) == str(plan.compose_working_dir)
                        and labels.get(_FILES_LABEL, "").split(",")
                        == [str(snapshot), str(override)]
                    )
                if targets and not (is_original or is_ours):
                    raise RuntimeError(
                        "effects ownership changed; automatic rollback refuses a foreign replacement"
                    )
                _recreate(
                    plan,
                    rollback_override,
                    runner,
                    snapshot,
                    snapshot_sha,
                    plan.expected_image_id,
                )
                target = _wait_target(
                    plan, plan.expected_image_id, runner, sleep, monotonic
                )
                restored = _probe(runner, script, container=target["Id"])
                if restored != live_before:
                    raise RuntimeError(
                        "restored installed evidence differs from the prior effects container"
                    )
                _assert_unchanged(
                    plan,
                    before,
                    target_before,
                    runner,
                    rollback_override,
                    snapshot,
                    snapshot_sha,
                    target,
                    rendered_current,
                )
                _assert_lock(lock, lock_identity)
                receipt.update(
                    status="FAILED_ROLLED_BACK",
                    rollback_verified=True,
                    rollback_container_id=target["Id"],
                    verified_at=datetime.now(UTC).isoformat(),
                )
            except BaseException as rollback_error:  # noqa: BLE001 — recovery failure must never become success
                receipt.update(
                    status="FAILED_UNFENCED"
                    if isinstance(rollback_error, CommandProcessUnfencedError)
                    else "FAILED_ROLLBACK",
                    rollback_error=str(rollback_error)
                    if isinstance(rollback_error, RuntimeError)
                    else type(rollback_error).__name__,
                )
    finally:
        try:
            if record_dir.is_dir():
                receipt["finished_at"] = datetime.now(UTC).isoformat()
                _write(record_dir / "receipt.json", receipt)
        finally:
            # Never remove another owner's files or recurse into deployment state.
            if receipt["status"] != "FAILED_UNFENCED":
                try:
                    _assert_lock(lock, lock_identity)
                except RuntimeError:
                    pass  # Ownership lost: never remove a foreign lock.
                else:
                    pid_file.unlink()
                    lock.rmdir()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    def interrupted(signum: int, frame: object) -> None:
        raise InterruptedError(
            "scoped rollout interrupted; rollback adjudication required"
        )

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_deploy(
            ModelEffectsDeployPlan.load(args.plan), execute=args.execute
        )
    except Exception as exc:  # noqa: BLE001 — CLI reports all unverified outcomes as refusal
        detail = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        print(json.dumps({"status": "REFUSED", "error": detail}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"PREVIEW", "PASSED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
