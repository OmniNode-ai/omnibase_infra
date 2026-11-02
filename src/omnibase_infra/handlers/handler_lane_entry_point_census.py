# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Record the command of every runtime-image container, per lane (OMN-21288, T24).

The census reads container inspection (a recorded ``docker inspect`` array, or the
daemon through read-only ``docker ps`` and ``docker inspect``), keeps the containers
whose image the policy names as the runtime image or whose entrypoint is the runtime
image's launcher (a container created from a bare image id carries no image name to
match), and records each one's entrypoint
and cmd as run. An entry point outside the policy's allowed list reads ``remove``.
The policy, the allowed list and the table families are data; this module holds
none of them. It starts, stops and changes no container.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lane_entry_point_census import (
    ModelContainerEntryPoint,
    ModelEntryPointFamily,
    ModelLaneEntryPointCensus,
    ModelLaneEntryPointCensusRequest,
    ModelLaneEntryPointCensusResult,
    ModelLaneEntryPointPolicy,
)

COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
_PYTHON = re.compile(r"python[0-9.]*")


def load_policy(path: Path) -> ModelLaneEntryPointPolicy:
    """Load and validate the entry-point policy data file."""
    raw = load_yaml_mapping_no_duplicates(
        path.read_text(encoding="utf-8"), source=path.name
    )
    return ModelLaneEntryPointPolicy.model_validate(raw)


def lane_projects(manifest_path: Path) -> dict[str, str]:
    """Map each declared compose project to the lane the manifest names it for."""
    raw = load_yaml_mapping_no_duplicates(
        manifest_path.read_text(encoding="utf-8"), source=manifest_path.name
    )
    lanes = raw.get("lanes")
    if not isinstance(lanes, dict):
        raise ValueError(f"{manifest_path.name}: no 'lanes' mapping")
    result: dict[str, str] = {}
    for lane, spec in lanes.items():
        project = spec.get("compose_project") if isinstance(spec, dict) else None
        if isinstance(project, str) and project:
            result[project] = str(lane)
    return result


def _is_launched(argv: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    """True when argv starts with the launcher prefix, its program compared by basename."""
    if not prefix or len(argv) < len(prefix):
        return False
    head = argv[0].rsplit("/", 1)[-1] == prefix[0].rsplit("/", 1)[-1]
    return head and argv[1 : len(prefix)] == prefix[1:]


def entry_point_of(argv: tuple[str, ...], launcher_prefix: tuple[str, ...]) -> str:
    """Name the command after the launcher prefix: ``python -m pkg.mod`` or the program."""
    if launcher_prefix and _is_launched(argv, launcher_prefix):
        argv = argv[len(launcher_prefix) :]
    if not argv:
        return ""
    program = argv[0].rsplit("/", 1)[-1]
    if _PYTHON.fullmatch(program) and len(argv) > 2 and argv[1] == "-m":
        return f"python -m {argv[2]}"
    return program


def _family_of(entry_point: str, families: tuple[ModelEntryPointFamily, ...]) -> str:
    for family in families:
        if re.search(family.pattern, entry_point):
            return family.label
    return entry_point


def _mapping(value: object) -> dict[str, object]:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _words(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected a list of strings, got {type(value).__name__}")


def _entry(
    inspected: dict[str, object], policy: ModelLaneEntryPointPolicy
) -> ModelContainerEntryPoint:
    config = _mapping(inspected.get("Config"))
    entrypoint = _words(config.get("Entrypoint"))
    cmd = _words(config.get("Cmd"))
    point = entry_point_of(entrypoint + cmd, policy.launcher_prefix)
    return ModelContainerEntryPoint(
        container=str(inspected.get("Name", "")).lstrip("/"),
        image=str(config.get("Image", "")),
        state=str(_mapping(inspected.get("State")).get("Status", "")),
        entrypoint=entrypoint,
        cmd=cmd,
        command=" ".join(entrypoint + cmd),
        entry_point=point,
        family=_family_of(point, policy.entry_point_families),
        verdict="allowed" if point in policy.allowed_commands else "remove",
    )


def build_census(
    inspected: list[dict[str, object]],
    policy: ModelLaneEntryPointPolicy,
    projects: dict[str, str],
) -> tuple[ModelLaneEntryPointCensus, ...]:
    """One census record per lane that holds a runtime-image container."""
    runtime = [re.compile(pattern) for pattern in policy.runtime_image_patterns]
    by_lane: dict[str, tuple[str, list[ModelContainerEntryPoint]]] = {}
    for item in inspected:
        config = _mapping(item.get("Config"))
        image = str(config.get("Image", ""))
        if not any(pattern.search(image) for pattern in runtime) and not _is_launched(
            _words(config.get("Entrypoint")), policy.launcher_prefix
        ):
            continue
        labels = _mapping(config.get("Labels"))
        project = str(labels.get(COMPOSE_PROJECT_LABEL, ""))
        lane = projects.get(project) or project or "unlabelled"
        by_lane.setdefault(lane, (project, []))[1].append(_entry(item, policy))
    records = []
    for lane in sorted(by_lane):
        project, entries = by_lane[lane]
        ordered = tuple(sorted(entries, key=lambda e: e.container))
        removed = tuple(e.container for e in ordered if e.verdict == "remove")
        records.append(
            ModelLaneEntryPointCensus(
                lane=lane,
                compose_project=project,
                container_count=len(ordered),
                allowed_count=len(ordered) - len(removed),
                remove_count=len(removed),
                remove=removed,
                containers=ordered,
            )
        )
    return tuple(records)


def render_table(censuses: tuple[ModelLaneEntryPointCensus, ...]) -> str:
    """The plan's entry-point table, regenerated from the census records."""
    blocks = []
    for census in censuses:
        rows = [
            f"Lane `{census.lane}`: {census.container_count} runtime-image containers, "
            f"{census.allowed_count} on an allowed entry point, "
            f"{census.remove_count} to remove.",
            "",
            "| Entry point | Containers | Names | Disposition |",
            "|---|---|---|---|",
        ]
        families: dict[str, list[ModelContainerEntryPoint]] = {}
        for entry in census.containers:
            families.setdefault(entry.family, []).append(entry)
        for family in sorted(families, key=lambda f: (-len(families[f]), f)):
            members = families[family]
            disposition = (
                "stays"
                if all(m.verdict == "allowed" for m in members)
                else "entry point to remove"
            )
            names = ", ".join(m.container for m in members)
            rows.append(f"| `{family}` | {len(members)} | {names} | {disposition} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _docker(
    args: list[str], docker_host: str, timeout_seconds: int | None
) -> subprocess.CompletedProcess[str]:
    base = ["docker", "--host", docker_host] if docker_host else ["docker"]
    return subprocess.run(
        [*base, *args],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def read_live_inspection(
    docker_host: str, timeout_seconds: int | None
) -> list[dict[str, object]]:
    """Inspect every container through read-only ``docker ps`` and ``docker inspect``.

    A container removed between the listing and the inspection makes ``docker inspect``
    exit non-zero while it still prints the others; those are kept, and only a call that
    prints no array at all is a refusal.
    """
    listed = _docker(["ps", "-a", "-q", "--no-trunc"], docker_host, timeout_seconds)
    if listed.returncode != 0:
        raise ValueError(
            f"docker ps exited {listed.returncode}: {listed.stderr.strip()[:300]}"
        )
    ids = listed.stdout.split()
    if not ids:
        return []
    inspected = _docker(["inspect", *ids], docker_host, timeout_seconds)
    if inspected.returncode != 0 and not inspected.stdout.lstrip().startswith("["):
        raise ValueError(
            f"docker inspect exited {inspected.returncode}: {inspected.stderr.strip()[:300]}"
        )
    parsed = json.loads(inspected.stdout)
    if not isinstance(parsed, list):
        raise ValueError("docker inspect did not return an array")
    return parsed


class HandlerLaneEntryPointCensus:
    """Record each runtime-image container's command and grade it against the policy.

    Classification: ``INFRA_HANDLER`` + ``EFFECT``. Reads a recorded inspection or the
    docker daemon (read-only) and the declared policy; changes nothing.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelLaneEntryPointCensusRequest
    ) -> ModelLaneEntryPointCensusResult:
        """Run one census; a refusal comes back as ``success=False`` with its reason."""
        try:
            policy = load_policy(Path(request.policy_path))
            projects = (
                lane_projects(Path(request.manifest_path))
                if request.manifest_path
                else {}
            )
            if request.inspect_json_path:
                inspected = json.loads(
                    Path(request.inspect_json_path).read_text(encoding="utf-8")
                )
                if not isinstance(inspected, list):
                    raise ValueError("the recorded inspection is not an array")
            else:
                inspected = read_live_inspection(
                    request.docker_host, request.docker_timeout_seconds
                )
            censuses = build_census(inspected, policy, projects)
        except (
            OSError,
            ValueError,
            yaml.YAMLError,
            subprocess.SubprocessError,
        ) as error:
            return ModelLaneEntryPointCensusResult(
                success=False, error_message=str(error)
            )
        return ModelLaneEntryPointCensusResult(
            success=True, censuses=censuses, table_markdown=render_table(censuses)
        )


__all__: list[str] = [
    "HandlerLaneEntryPointCensus",
    "build_census",
    "entry_point_of",
    "lane_projects",
    "load_policy",
    "read_live_inspection",
    "render_table",
]
