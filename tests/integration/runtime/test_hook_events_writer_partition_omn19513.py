# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: discovery, the lane compose environment and the profile filter agree.

The dev lane hands ``projection_claude_hook_events`` to its own writer container.
This drives the real contract discovery path and the real profile filter with the
environment the compose files actually render, so a rename on either side (the
compose variable value, the contract name, the filter) fails here and not in a
running lane where the projection silently has no owner or two.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import yaml

from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
    ENV_RUNTIME_NODE_SELECTION,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.integration

DOCKER = Path(__file__).resolve().parents[3] / "docker"
NODE = "projection_claude_hook_events"
OTHER = "node_other_omn19513"


def _service_env(compose: str, service: str) -> dict[str, str]:
    text = (DOCKER / compose).read_text(encoding="utf-8")
    services: dict[str, Any] = yaml.safe_load(text.replace("!override", ""))["services"]
    environment = services[service]["environment"]
    return {str(key): str(value) for key, value in environment.items()}


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

    main_env = _service_env("docker-compose.dev-lane.yml", "omninode-runtime")
    writer_env = _service_env(
        "docker-compose.infra.yml", "projection-claude-hook-events-writer"
    )

    main = filter_manifest_for_runtime_profile(
        manifest, main_env.get("RUNTIME_PROFILE", "main"), environ=main_env
    )
    writer = filter_manifest_for_runtime_profile(
        manifest, writer_env["RUNTIME_PROFILE"], environ=writer_env
    )

    assert {c.name for c in main.manifest.contracts} == {OTHER}
    assert {c.name for c in writer.manifest.contracts} == {NODE}
    assert main_env[ENV_RUNTIME_NODE_EXCLUSION] == NODE
    assert writer_env[ENV_RUNTIME_NODE_SELECTION] == NODE
    # The consumer group id carries KAFKA_INSTANCE_ID; committed offsets only
    # carry over when the writer shares the main runtime's instance id.
    assert writer_env["KAFKA_INSTANCE_ID"] == "runtime-main"
