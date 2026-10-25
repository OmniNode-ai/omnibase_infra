# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane bundle supplies every compaction pack-builder setting (OMN-21099).

The session source's live mode requires transcript and ledger paths for the
compaction profile's operator_words_unrecorded factor. Both paths must name
read-only mounts whose host directories come from the operator env; the lanes
that layer the dev-lane overlay and replace those mounts must blank the paths.

The committed-entry checks read each compose file with compose's tags
constructed. The render checks run ``docker compose config`` over each lane
stack, so compose's own merge semantics decide the effective runtime-effects
service; they skip only when the docker CLI is absent. Neither touches a
running lane.

Left out on purpose: ONEX_CONTEXT_REPLAY_ROOT is the session source's pinned
replay mode only, and OMNI_HOME is read by the guidance and repo-doctrine
sources, whose factors the compaction_orchestrator profile does not select.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath
from typing import Any, Literal

import pytest

from tests.ci._compose_render_omn21099 import (
    BORROWER_STACKS,
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    REPO_ROOT,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

SESSION_SOURCE_PATHS = ("ONEX_SESSION_TRANSCRIPT_ROOT", "ONEX_LEDGER_PATH")
TRANSCRIPT_TARGET = "/etc/onex/session-transcripts"
LEDGER_TARGET = "/etc/onex/work-ledger"
# Container target -> the operator env variable naming its host directory.
SESSION_MOUNTS = {
    TRANSCRIPT_TARGET: "DEV_LANE_SESSION_TRANSCRIPT_DIR",
    LEDGER_TARGET: "DEV_LANE_WORK_LEDGER_DIR",
}
STACKS: dict[str, tuple[str, ...]] = {
    DEV_LANE_OVERLAY: DEV_LANE_STACK,
    **BORROWER_STACKS,
}

needs_compose = pytest.mark.skipif(
    not docker_compose_available(),
    reason="docker compose CLI absent; the committed-entry checks still hold",
)


@dataclass(frozen=True)
class PackBuilderSetting:
    name: str
    reader: str
    kind: Literal["value", "dir", "file"]


PACK_BUILDER_SETTINGS: tuple[PackBuilderSetting, ...] = (
    PackBuilderSetting(
        "ONEX_CONTEXT_SOURCE_TENANT_ID",
        "node_context_compaction_orchestrator compaction_runtime.tenant_id; "
        "node_context_source_ledger_effect and "
        "node_context_source_work_state_effect tenant_ids",
        "value",
    ),
    PackBuilderSetting(
        "OMNINODE_INTERNAL_DB_URL",
        "session barrier probe dsn_env; ledger and work-state sources dsn_env",
        "value",
    ),
    PackBuilderSetting(
        "ONEX_SESSION_TRANSCRIPT_ROOT",
        "node_session_context_sources_effect live mode",
        "dir",
    ),
    PackBuilderSetting(
        "ONEX_LEDGER_PATH",
        "node_session_context_sources_effect live mode",
        "file",
    ),
)


@cache
def _rendered(overlay: str) -> dict[str, Any]:
    return render_stack(STACKS[overlay])


def _overlay_runtime_effects(filename: str) -> dict[str, Any]:
    service = load_compose(filename)["services"]["runtime-effects"]
    assert isinstance(service, dict)
    return service


def _is_set(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mount_target(setting: PackBuilderSetting, value: str) -> str:
    return str(PurePosixPath(value).parent) if setting.kind == "file" else value


def _bind_mounts(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Rendered long-form volumes keyed by container target."""
    return {
        volume["target"]: volume
        for volume in service.get("volumes", [])
        if volume.get("type") == "bind"
    }


# ------------------------------------------------------ committed entries


def test_the_overlay_names_the_session_source_paths() -> None:
    environment = _overlay_runtime_effects(DEV_LANE_OVERLAY)["environment"]
    assert environment["ONEX_SESSION_TRANSCRIPT_ROOT"] == TRANSCRIPT_TARGET
    assert environment["ONEX_LEDGER_PATH"] == f"{LEDGER_TARGET}/ROLLING_WORK_LEDGER.md"


@pytest.mark.parametrize(("target", "variable"), sorted(SESSION_MOUNTS.items()))
def test_the_overlay_mounts_each_session_directory_read_only(
    target: str, variable: str
) -> None:
    volumes = _overlay_runtime_effects(DEV_LANE_OVERLAY)["volumes"]
    entries = [
        volume
        for volume in volumes
        if isinstance(volume, str) and volume.endswith(f":{target}:ro")
    ]
    assert len(entries) == 1, f"{target}: {entries}"
    assert entries[0].startswith(f"${{{variable}:-"), entries[0]


def test_borrower_overlays_are_found() -> None:
    """Positive control: an empty discovery would pass every borrower test."""
    assert BORROWER_STACKS, "no overlay layers the dev-lane overlay"


@pytest.mark.parametrize("overlay", sorted(BORROWER_STACKS))
def test_borrower_overlays_blank_the_session_source_paths(overlay: str) -> None:
    environment = _overlay_runtime_effects(overlay).get("environment", {})
    failures = [name for name in SESSION_SOURCE_PATHS if environment.get(name) != ""]
    assert not failures, f"{overlay} must explicitly blank: {failures}"


# ------------------------------------------------------ rendered stacks


@needs_compose
def test_the_bundle_gives_runtime_effects_every_pack_builder_setting() -> None:
    environment = _rendered(DEV_LANE_OVERLAY)["runtime-effects"]["environment"]
    missing = [
        setting.name
        for setting in PACK_BUILDER_SETTINGS
        if not _is_set(environment.get(setting.name))
    ]
    assert not missing, f"Missing pack-builder settings: {missing}"


@needs_compose
def test_each_path_setting_is_mounted_read_only() -> None:
    service = _rendered(DEV_LANE_OVERLAY)["runtime-effects"]
    mounts = _bind_mounts(service)
    failures: list[str] = []
    for setting in PACK_BUILDER_SETTINGS:
        if setting.kind == "value":
            continue
        value = service["environment"].get(setting.name)
        if not _is_set(value):
            failures.append(f"{setting.name}: absent or blank")
            continue
        target = _mount_target(setting, value)
        mount = mounts.get(target)
        if mount is None:
            failures.append(f"{setting.name}: {target} has no bind mount")
        elif mount.get("read_only") is not True:
            failures.append(f"{setting.name}: {target} is not read-only")
    assert not failures, f"Pack-builder path mount failures: {failures}"


@needs_compose
def test_the_session_mount_sources_come_from_the_operator_env() -> None:
    sources = {
        variable: str(REPO_ROOT / "operator-named" / variable.lower())
        for variable in SESSION_MOUNTS.values()
    }
    mounts = _bind_mounts(render_stack(DEV_LANE_STACK, sources)["runtime-effects"])
    failures = [
        f"{target}: {mounts.get(target, {}).get('source')!r}"
        for target, variable in SESSION_MOUNTS.items()
        if mounts.get(target, {}).get("source") != sources[variable]
    ]
    assert not failures, f"Session mounts not from the operator env: {failures}"


@needs_compose
@pytest.mark.parametrize("overlay", sorted(BORROWER_STACKS))
def test_borrower_stacks_render_the_session_source_paths_blank(overlay: str) -> None:
    environment = _rendered(overlay)["runtime-effects"]["environment"]
    failures = [name for name in SESSION_SOURCE_PATHS if environment.get(name)]
    assert not failures, f"{overlay} stack renders non-blank: {failures}"


@needs_compose
@pytest.mark.parametrize("overlay", sorted(STACKS))
def test_the_session_sources_reach_no_other_service(overlay: str) -> None:
    failures: list[str] = []
    for service_name, service in _rendered(overlay).items():
        if service_name == "runtime-effects":
            continue
        environment = service.get("environment") or {}
        failures += [
            f"{service_name}.{name}"
            for name in SESSION_SOURCE_PATHS
            if name in environment
        ]
        failures += [
            f"{service_name}:{target}"
            for target in _bind_mounts(service)
            if target in SESSION_MOUNTS
        ]
    assert not failures, f"{overlay}: session sources on other services: {failures}"
