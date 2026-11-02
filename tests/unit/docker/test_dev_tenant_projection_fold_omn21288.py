# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The primary dev runtime carries tenant projections without a second kernel."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.validation.model_stack_shape_data import (
    ModelStackShapeData,
)
from omnibase_core.validation.checker_stack_shape import check_compose
from omnibase_core.validation.validator_stack_shape import ValidatorStackShape
from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "docker/docker-compose.dev-tenant-fold.yml"
STACK_SHAPE_DATA = ROOT / "config/validation/stack_shape.yaml"
DEV_COMPOSE_FILES = (
    ROOT / "docker/docker-compose.infra.yml",
    ROOT / "docker/docker-compose.dev-lane.yml",
    OVERLAY,
)
FOLDED_KEY = "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
SERVICE = "tenant-projection-writer"


def _env(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return dict(str(e).split("=", 1) for e in env if "=" in str(e))
    return {str(k): str(v) for k, v in env.items()}


def _folded_only(profiles: object) -> bool:
    return (
        isinstance(profiles, list)
        and bool(profiles)
        and all(str(p).startswith("folded-") for p in profiles)
    )


def dev_stack_services() -> dict[str, dict]:
    """The dev stack: the base, shared dev overlay and primary-dev fold overlay merged.

    A service gated only under a folded-role profile starts under no profile a dev
    bring-up requests: its role runs inside main.
    """
    merged = dict(load_compose(DEV_COMPOSE_FILES[0]).get("services") or {})
    for path in DEV_COMPOSE_FILES[1:]:
        for name, overlay in (load_compose(path).get("services") or {}).items():
            if overlay is None:  # compose ``!reset null`` removes a service
                merged.pop(name, None)
                continue
            base = dict(merged.get(name) or {})
            env = {**_env(base), **_env(overlay)}
            base.update(overlay)
            if env:
                base["environment"] = env
            merged[name] = base
    return {
        name: service
        for name, service in merged.items()
        if not _folded_only(service.get("profiles"))
    }


def test_policy_folds_tenant_projections_on_primary_dev_only() -> None:
    policy = load_contract()
    assert (
        "tenant-projection" in policy.profiles["dev"].processes["main"].folded_profiles
    )
    assert all(
        "tenant-projection" not in profile.processes["main"].folded_profiles
        for name, profile in policy.profiles.items()
        if name != "dev"
    )
    rendered = render_env(policy)
    assert "tenant-projection" in rendered[FOLDED_KEY].split(",")
    assert (ROOT / "docker/runtime-policy.env").read_text() == format_dotenv(rendered)


def test_overlay_gates_the_carrier_under_its_folded_profile_only() -> None:
    document = yaml.compose(OVERLAY.read_text())
    assert isinstance(document, yaml.MappingNode)
    services = next(value for key, value in document.value if key.value == "services")
    carrier = next(value for key, value in services.value if key.value == SERVICE)
    assert isinstance(carrier, yaml.MappingNode)
    profiles = next(value for key, value in carrier.value if key.value == "profiles")
    assert profiles.tag == "!override"
    assert [item.value for item in profiles.value] == ["folded-tenant-projection"]
    assert SERVICE not in dev_stack_services()


def test_overlay_loads_with_the_plain_safe_loader_readers_use() -> None:
    document = yaml.safe_load(OVERLAY.read_text().replace("!override", ""))
    assert isinstance(document["services"][SERVICE], dict)


def test_main_keeps_its_identity_bindings_and_the_carriers_state() -> None:
    main = dev_stack_services()["omninode-runtime"]
    assert main["environment"]["RUNTIME_PROFILE"] == "main"
    assert main["environment"]["ONEX_RUNTIME_FOLDED_PROFILES"] == (
        f"${{{FOLDED_KEY}:?runtime policy contract must set {FOLDED_KEY}}}"
    )
    base = load_compose(ROOT / "docker/docker-compose.infra.yml")["services"]
    for key in ("ONEX_TENANT_DB_URL", "ONEX_DATABASE_TOPOLOGY_PROFILE"):
        assert main["environment"][key] == base[SERVICE]["environment"][key]
    volumes = load_compose(OVERLAY)["services"]["omninode-runtime"]["volumes"]
    assert "tenant_projection_data:/app/tenant-projection-data" in volumes
    assert "tenant_projection_logs:/app/tenant-projection-logs:ro" in volumes
    assert "runtime_data:/app/data" in base["omninode-runtime"]["volumes"]


@pytest.mark.parametrize(
    "definition",
    [
        {"image": "example/runtime:tag"},
        {
            "image": "example/runtime:tag",
            "environment": {"RUNTIME_PROFILE": "tenant-projection"},
        },
    ],
)
def test_a_restored_carrier_is_refused_even_without_a_build(definition: dict) -> None:
    """The stack-shape allowlist refuses the carrier in a compose file that does not baseline it."""
    data = yaml.safe_load(STACK_SHAPE_DATA.read_text(encoding="utf-8"))
    allowlist = ModelStackShapeData(
        runtime_services=frozenset(data["runtime"]["services"]),
        max_replicas=int(data["runtime"]["max_replicas"]),
        dependencies=frozenset(data["dependencies"]),
        one_shots=frozenset(n for n in data["one_shots"] if "*" not in n),
        lanes={},
        baseline=(),
    )
    assert SERVICE not in allowlist.runtime_services | allowlist.dependencies
    assert SERVICE not in allowlist.one_shots
    rules = {r.rule_id: r for r in ValidatorStackShape(root=ROOT).contract.rules}
    refused = check_compose(
        allowlist,
        "docker/docker-compose.restored.yml",
        {SERVICE: definition},
        ["dev"],
        rules,
    )
    assert [issue.message for issue in refused if SERVICE in issue.message]


@pytest.mark.parametrize(
    "project",
    [
        render_env(load_contract())[key]
        for key in ("DEV_COMPOSE_PROJECT", "STABILITY_TEST_COMPOSE_PROJECT")
    ],
)
def test_compose_resolver_scopes_the_fold_to_primary_dev(project: str) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ROOT}/scripts/runtime_build/compose_files.sh"\n'
            f"resolve_compose_file_args args /DEPLOY {project}\n"
            'printf "%s\\n" "${args[@]}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("docker-compose.dev-tenant-fold.yml" in result.stdout) == (
        project == render_env(load_contract())["DEV_COMPOSE_PROJECT"]
    )


def test_fold_overlay_always_triggers_the_stack_gate() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "stack-shape"
    ]
    assert len(hooks) == 1
    assert re.search(hooks[0]["files"], str(OVERLAY.relative_to(ROOT)))
