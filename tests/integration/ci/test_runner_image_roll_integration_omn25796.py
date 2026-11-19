# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-25796 runner image roll node.

Loads the real contract from disk, resolves its handler through the declared
module path, and plans a roll for a declared secondary pool from the repository's
real fleet config, Compose file and runner image lock.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "src/omnibase_infra/nodes/node_runner_image_roll/contract.yaml"
FLEET = ROOT / "config/runner_fleet.yaml"


def test_contract_declares_importable_handler_and_topic() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    handler_cls = getattr(importlib.import_module(entry["module"]), entry["name"])
    assert callable(handler_cls)
    assert contract["event_bus"]["subscribe_topics"] == [
        "onex.cmd.platform.runner-image-roll.v1"
    ]


async def test_dry_run_plans_every_runner_of_a_declared_secondary_pool(
    tmp_path: Path,
) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    handler = getattr(importlib.import_module(entry["module"]), entry["name"])()
    fleet = load_runner_fleet_config(FLEET)
    host, pool = next(
        (host, pool)
        for host in fleet.hosts
        if host.host != fleet.runner_host
        for pool in host.pools
        if "deploy" not in pool.classes
    )
    token = tmp_path / "token"
    token.touch(mode=0o600)
    token.write_text("fixture-token\n")

    result = await handler.handle(
        ModelRunnerImageRollRequest(
            host=host.host,
            prefix=pool.runner_name_prefix,
            repo_root=ROOT,
            fleet_config=FLEET,
            token_path=token,
            dry_run=True,
        )
    )

    assert result.success, result.error_message
    assert result.planned == tuple(
        f"{pool.runner_name_prefix}-{i}" for i in range(1, pool.expected_count + 1)
    )
