# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane bundle supplies the bus-backed pack-builder settings (OMN-21099).

The compaction pack builder and live session source use the context-source
tenant and OMNINODE_INTERNAL_DB_URL. Since omnimarket#4383, the live source
reads the bus-resident session_content and work_ledger_rows projections through
that DSN, not a transcript directory or markdown ledger. Those file paths and
mounts are retired; ONEX_LEDGER_PATH also redirects the hosted-minutes watcher's
ledger writes, so even a blank entry must not be shipped to any service.

Committed-entry checks read compose YAML with its tags constructed. Render
checks use ``docker compose config`` for the dev lane and each borrower stack,
skip only when the compose CLI is absent, and touch no running lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import pytest

from tests.ci._compose_render_omn21099 import (
    BORROWER_STACKS,
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

RETIRED_FILE_PATHS: tuple[str, ...] = (
    "ONEX_SESSION_TRANSCRIPT_ROOT",
    "ONEX_LEDGER_PATH",
)
RETIRED_MOUNT_TARGETS: tuple[str, ...] = (
    "/etc/onex/session-transcripts",
    "/etc/onex/work-ledger",
)
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
    kind: Literal["value"]


PACK_BUILDER_SETTINGS: tuple[PackBuilderSetting, ...] = (
    PackBuilderSetting(
        "ONEX_CONTEXT_SOURCE_TENANT_ID",
        "node_context_compaction_orchestrator compaction_runtime.tenant_id; "
        "ledger, work-state and live session sources tenant_ids",
        "value",
    ),
    PackBuilderSetting(
        "OMNINODE_INTERNAL_DB_URL",
        "session barrier probe dsn_env; ledger and work-state sources dsn_env; "
        "live session source dsn_env for session_content and work_ledger_rows "
        "(omnimarket#4383)",
        "value",
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


def test_the_overlay_has_no_retired_file_paths_or_mounts() -> None:
    service = _overlay_runtime_effects(DEV_LANE_OVERLAY)
    environment = service.get("environment") or {}
    failures = [name for name in RETIRED_FILE_PATHS if name in environment]
    failures += [
        f"{target}: {volume}"
        for volume in service.get("volumes", [])
        if isinstance(volume, str)
        for target in RETIRED_MOUNT_TARGETS
        if target in volume
    ]
    assert not failures, f"{DEV_LANE_OVERLAY} runtime-effects: {failures}"


def test_retired_checks_find_borrower_overlays() -> None:
    """Positive control: an empty discovery would pass every borrower test."""
    assert BORROWER_STACKS, "no overlay layers the dev-lane overlay"


@pytest.mark.parametrize("overlay", sorted(BORROWER_STACKS))
def test_borrower_overlays_have_no_retired_file_paths(overlay: str) -> None:
    environment = _overlay_runtime_effects(overlay).get("environment") or {}
    failures = [name for name in RETIRED_FILE_PATHS if name in environment]
    assert not failures, f"{overlay} runtime-effects: {failures}"


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
@pytest.mark.parametrize("overlay", sorted(STACKS))
def test_rendered_stacks_have_no_retired_file_paths_or_bind_mounts(
    overlay: str,
) -> None:
    failures: list[str] = []
    for service_name, service in _rendered(overlay).items():
        environment = service.get("environment") or {}
        failures += [
            f"{service_name}.{name}"
            for name in RETIRED_FILE_PATHS
            if name in environment
        ]
        failures += [
            f"{service_name}:{volume['target']}"
            for volume in service.get("volumes", [])
            if volume.get("type") == "bind"
            and volume.get("target") in RETIRED_MOUNT_TARGETS
        ]
    assert not failures, f"{overlay}: retired file paths or bind mounts: {failures}"
