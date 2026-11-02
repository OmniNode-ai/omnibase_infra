# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovery preserves one hook-events owner across the fold and rollback."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
    ENV_RUNTIME_NODE_SELECTION,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.integration

NODE = "projection_claude_hook_events"
OTHER = "node_other_omn19513"


def _write_contract(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    path = directory / "contract.yaml"
    path.write_text(
        dedent(f"""\
            name: "{name}"
            node_type: "effect"
            contract_version:
              major: 1
              minor: 0
              patch: 0
            node_version: "1.0.0"
            description: "OMN-19513 partition fixture"
        """),
        encoding="utf-8",
    )
    return path


def test_lane_environment_gives_the_projection_exactly_one_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "local")
    manifest = discover_contracts_from_paths(
        [_write_contract(tmp_path, NODE), _write_contract(tmp_path, OTHER)]
    )
    assert manifest.total_errors == 0
    assert {contract.name for contract in manifest.contracts} == {NODE, OTHER}

    services = dev_stack_services()
    assert "projection-claude-hook-events-writer" not in services
    main_env = {
        key: value
        for key, value in services["omninode-runtime"]["environment"].items()
        if value != "None"
    }
    main_env["ONEX_RUNTIME_FOLDED_PROFILES"] = render_env(load_contract())[
        "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
    ]
    main = filter_manifest_for_runtime_profile(manifest, "main", environ=main_env)
    assert not main.manifest.errors
    assert {c.name for c in main.manifest.contracts} == {NODE, OTHER}
    assert ENV_RUNTIME_NODE_EXCLUSION not in main_env

    # Rollback retains the inert base carrier and its node selection. Restoring
    # the exclusion hands the node back to it without a second owner.
    base = load_compose(
        Path(__file__).resolve().parents[3] / "docker/docker-compose.infra.yml"
    )
    writer = base["services"]["projection-claude-hook-events-writer"]
    assert writer["profiles"] == ["profile-carrier-optin"]
    writer_env = writer["environment"]
    rollback_main_env = {**main_env, ENV_RUNTIME_NODE_EXCLUSION: NODE}
    rollback_main = filter_manifest_for_runtime_profile(
        manifest, "main", environ=rollback_main_env
    )
    rollback_writer = filter_manifest_for_runtime_profile(
        manifest, writer_env["RUNTIME_PROFILE"], environ=writer_env
    )
    assert not rollback_main.manifest.errors
    assert not rollback_writer.manifest.errors
    assert {c.name for c in rollback_main.manifest.contracts} == {OTHER}
    assert {c.name for c in rollback_writer.manifest.contracts} == {NODE}
    assert writer_env[ENV_RUNTIME_NODE_SELECTION] == NODE
    assert (
        writer_env["KAFKA_INSTANCE_ID"]
        == main_env["KAFKA_INSTANCE_ID"]
        == "runtime-main"
    )
