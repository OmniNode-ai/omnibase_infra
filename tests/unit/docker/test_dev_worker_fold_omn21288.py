# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Primary dev worker retirement, including the real node ownership path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.unit.docker.test_dev_stack_runtime_only_omn21288 import dev_stack_services

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_dev_main_attaches_every_worker_node() -> None:
    discovered = discover_contracts_from_paths(
        sorted((ROOT / "src/omnibase_infra/nodes").glob("*/contract.yaml"))
    )
    assert not discovered.errors
    env = {
        "ONEX_RUNTIME_LANE": dev_stack_services()["omninode-runtime"]["environment"][
            "ONEX_RUNTIME_LANE"
        ]
    }
    main = filter_manifest_for_runtime_profile(discovered, "main", environ=env)
    workers = filter_manifest_for_runtime_profile(discovered, "workers", environ=env)
    assert workers.manifest.contracts, "A fold without worker nodes proves nothing"
    rendered = render_env(load_contract())
    env["ONEX_RUNTIME_FOLDED_PROFILES"] = rendered["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]
    folded = filter_manifest_for_runtime_profile(discovered, "main", environ=env)
    tenant = filter_manifest_for_runtime_profile(
        discovered,
        "tenant-projection",
        environ={"ONEX_RUNTIME_LANE": env["ONEX_RUNTIME_LANE"]},
    )
    expected = {
        c.name
        for c in (
            *main.manifest.contracts,
            *workers.manifest.contracts,
            *tenant.manifest.contracts,
        )
    }
    attached = {c.name for c in folded.manifest.contracts}
    assert not folded.manifest.errors
    assert set(folded.folded_profiles) == {"workers", "tenant-projection"}
    assert folded.folded_contracts
    assert attached == expected
    folded_by_name = {c.name: c for c in folded.manifest.contracts}
    for contract in workers.manifest.contracts:
        # Preserve the contract itself, including subscriptions and handlers,
        # when main owns the worker role.
        assert folded_by_name[contract.name] == contract


def test_dev_stack_has_no_worker_and_preserves_main_identity() -> None:
    stack = dev_stack_services()
    assert "runtime-worker" not in stack
    env = stack["omninode-runtime"]["environment"]
    assert env["RUNTIME_PROFILE"] == "main"
    assert env["KAFKA_INSTANCE_ID"] == "runtime-main"
    assert env["ONEX_RUNTIME_FOLDED_PROFILES"] == (
        "${DEV_RUNTIME_MAIN_FOLDED_PROFILES:?runtime policy contract must set "
        "DEV_RUNTIME_MAIN_FOLDED_PROFILES}"
    )
    assert env["OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS"] == "redpanda:9092"


def test_shared_resolver_adds_the_fold_only_to_primary_dev() -> None:
    script = ROOT / "scripts/runtime_build/compose_files.sh"
    profiles = load_contract().profiles
    effects_folded_lanes = [
        profile
        for name, profile in profiles.items()
        if name != "dev" and profile.processes["main"].folded_profiles
    ]
    assert len(effects_folded_lanes) == 1
    for profile, folded in (
        (profiles["dev"], True),
        (effects_folded_lanes[0], False),
        (profiles["judge"], False),
        (profiles["prod"], False),
    ):
        project = profile.compose_project
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"\nresolve_compose_file_args files "$2" "$3"\n'
                'printf "%s\\n" "${files[@]}"',
                "worker-fold",
                str(script),
                str(ROOT),
                project,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert ("docker-compose.dev-fold.yml" in result.stdout) is folded
        if folded:
            assert "docker-compose.dev-tenant-fold.yml" in result.stdout
