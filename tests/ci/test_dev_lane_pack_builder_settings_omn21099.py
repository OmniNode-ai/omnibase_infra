# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane bundle supplies every compaction pack-builder setting (OMN-21099).

The session source's live mode requires transcript and ledger paths for the
compaction profile's operator_words_unrecorded factor. Both paths must name
read-only mounts; borrower lanes that replace those mounts must blank the paths.
These checks parse committed compose configuration without Docker or network I/O.

Left out on purpose: ONEX_CONTEXT_REPLAY_ROOT is the session source's pinned
replay mode only, and OMNI_HOME is read by the guidance and repo-doctrine
sources, whose factors the compaction_orchestrator profile does not select.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, NotRequired, TypedDict, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
SESSION_SOURCE_PATHS = ("ONEX_SESSION_TRANSCRIPT_ROOT", "ONEX_LEDGER_PATH")


def _borrower_overlays() -> tuple[str, ...]:
    """Lanes layering the dev-lane overlay, found as the render suite finds them."""
    return tuple(
        sorted(
            path.name
            for path in DOCKER_DIR.glob("docker-compose*.yml")
            if path.name != "docker-compose.dev-lane.yml"
            and "\n  runtime-effects:\n" in (text := path.read_text(encoding="utf-8"))
            and (
                "-f docker/docker-compose.dev-lane.yml" in text
                or "over docker-compose.infra.yml and docker-compose.dev-lane.yml"
                in text
            )
        )
    )


BORROWER_OVERLAYS = _borrower_overlays()


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

Volume = str | dict[str, object]


class ComposeService(TypedDict):
    environment: NotRequired[dict[str, object]]
    volumes: NotRequired[list[Volume]]


def _services(filename: str) -> dict[str, ComposeService]:
    text = (
        (DOCKER_DIR / filename)
        .read_text()
        .replace("!!merge ", "")
        .replace("!override", "")
        .replace("!reset", "")
    )
    return cast("dict[str, ComposeService]", yaml.safe_load(text)["services"])


def _runtime_effects_bundle() -> tuple[dict[str, object], list[Volume]]:
    base = _services("docker-compose.infra.yml")["runtime-effects"]
    overlay = _services("docker-compose.dev-lane.yml")["runtime-effects"]
    environment = dict(base.get("environment", {}))
    environment.update(overlay.get("environment", {}))
    volumes = base.get("volumes", []) + overlay.get("volumes", [])
    return environment, volumes


def _volume_targets(volumes: list[Volume], *, read_only: bool = False) -> set[str]:
    targets: set[str] = set()
    for volume in volumes:
        if isinstance(volume, str):
            # Parse from the right: operator-env source expressions contain ':'.
            parts = volume.rsplit(":", 2)
            target = parts[-1] if parts[-1].startswith("/") else parts[-2]
            is_read_only = volume.endswith(":ro")
        else:
            dict_target = volume["target"]
            assert isinstance(dict_target, str)
            target = dict_target
            is_read_only = volume.get("read_only") is True
        if not read_only or is_read_only:
            targets.add(target)
    return targets


@pytest.mark.unit
def test_the_bundle_gives_runtime_effects_every_pack_builder_setting() -> None:
    environment, _ = _runtime_effects_bundle()
    missing: list[str] = []
    for setting in PACK_BUILDER_SETTINGS:
        value = environment.get(setting.name)
        if not isinstance(value, str) or not value.strip():
            missing.append(setting.name)
    assert not missing, f"Missing pack-builder settings: {missing}"


@pytest.mark.unit
def test_each_path_setting_is_mounted_read_only() -> None:
    environment, volumes = _runtime_effects_bundle()
    read_only_targets = _volume_targets(volumes, read_only=True)
    failures: list[str] = []
    for setting in PACK_BUILDER_SETTINGS:
        if setting.kind == "value":
            continue
        value = environment.get(setting.name)
        if not isinstance(value, str) or not value.strip():
            failures.append(f"{setting.name}: absent or blank")
            continue
        target = str(PurePosixPath(value).parent) if setting.kind == "file" else value
        if target not in read_only_targets:
            failures.append(f"{setting.name}: {target} has no read-only mount")
    assert not failures, f"Pack-builder path mount failures: {failures}"


@pytest.mark.unit
def test_the_borrower_overlays_are_found() -> None:
    """Positive control: an empty discovery would pass the blanking test."""
    assert len(BORROWER_OVERLAYS) == 4, BORROWER_OVERLAYS


@pytest.mark.unit
@pytest.mark.parametrize("filename", BORROWER_OVERLAYS)
def test_borrower_overlays_blank_the_session_source_paths(filename: str) -> None:
    environment = _services(filename)["runtime-effects"].get("environment", {})
    failures = [name for name in SESSION_SOURCE_PATHS if environment.get(name) != ""]
    assert not failures, f"{filename} must explicitly blank: {failures}"


@pytest.mark.unit
def test_the_session_source_paths_are_only_on_runtime_effects() -> None:
    failures = [
        f"{service_name}.{name}"
        for service_name, service in _services("docker-compose.dev-lane.yml").items()
        if service_name != "runtime-effects"
        for name in SESSION_SOURCE_PATHS
        if name in service.get("environment", {})
    ]
    assert not failures, f"Session source paths on other services: {failures}"
