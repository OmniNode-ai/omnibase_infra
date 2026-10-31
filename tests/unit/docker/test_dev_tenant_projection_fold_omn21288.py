# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The primary dev runtime carries tenant projections without a second kernel."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)
from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_stack_runtime_only_omn21288 import (
    SEPARATE_SERVICES,
    dev_stack_services,
    find_out_of_runtime_workloads,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "docker/docker-compose.dev-tenant-fold.yml"
FOLDED_KEY = "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
SERVICE = "tenant-projection-writer"


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
    assert SERVICE not in SEPARATE_SERVICES
    assert SERVICE in find_out_of_runtime_workloads({SERVICE: definition})


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
        if hook["id"] == "dev-stack-runtime-only"
    ]
    assert len(hooks) == 1
    assert re.search(hooks[0]["files"], str(OVERLAY.relative_to(ROOT)))


# ---------------------------------------------------------------------------
# OMN-21288: the five projection writers that dispatch in-process
# ---------------------------------------------------------------------------
# omnimarket#4492 routes these contracts to handler_runtime_writer classes
# (onex_runtime_inprocess_dispatch), so main runs them as nodes once it owns
# their profiles. Runtime profile -> the dev-lane service that ran it alone.
RUNTIME_WRITERS = {
    "projection-writer-live-events": "projection-live-events-writer",
    "projection-writer-registration": "projection-registration-writer",
    "projection-writer-savings": "projection-savings-writer",
    "projection-writer-tenant-credentials": "projection-tenant-credentials-writer",
    "projection-writer-tenant-registry": "projection-tenant-registry-writer",
}
# The contract names those profiles own, as omnimarket declares them.
RUNTIME_WRITER_CONTRACTS = {
    "projection-writer-live-events": "projection_live_events",
    "projection-writer-registration": "projection_registration",
    "projection-writer-savings": "projection_savings",
    "projection-writer-tenant-credentials": "projection_tenant_credentials",
    "projection-writer-tenant-registry": "projection_tenant_registry",
}
HANDLER = ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"


def _overlay_gates() -> dict[str, str]:
    """Service -> the single folded profile the fold overlay gates it under."""
    document = yaml.compose(OVERLAY.read_text())
    assert isinstance(document, yaml.MappingNode)
    services = next(value for key, value in document.value if key.value == "services")
    gates: dict[str, str] = {}
    for key, service in services.value:
        assert isinstance(service, yaml.MappingNode)
        for field, value in service.value:
            if field.value == "profiles":
                assert value.tag == "!override", key.value
                items = [item.value for item in value.value]
                assert len(items) == 1 and items[0].startswith("folded-"), key.value
                gates[key.value] = items[0].removeprefix("folded-")
    return gates


def _handler_profile_map() -> dict[str, str]:
    body = re.search(
        r"local -A folded_profile_services=\((.*?)\n    \)", HANDLER.read_text(), re.S
    )
    assert body
    return dict(re.findall(r"\[([\w-]+)\]=([\w-]+)", body.group(1)))


def test_policy_folds_the_runtime_writers_on_primary_dev_only() -> None:
    policy = load_contract()
    folded = policy.profiles["dev"].processes["main"].folded_profiles
    assert set(RUNTIME_WRITERS) <= set(folded)
    for name, profile in policy.profiles.items():
        if name == "dev":
            continue
        for process in profile.processes.values():
            assert not set(RUNTIME_WRITERS) & set(process.folded_profiles), name
    rendered = render_env(policy)[FOLDED_KEY].split(",")
    assert set(RUNTIME_WRITERS) <= set(rendered)


def test_overlay_gates_each_folded_profile_and_nothing_else() -> None:
    gates = _overlay_gates()
    folded = set(render_env(load_contract())[FOLDED_KEY].split(","))
    assert set(gates.values()) == folded
    for profile, service in RUNTIME_WRITERS.items():
        assert gates[service] == profile
        assert service not in dev_stack_services()
    # The two writers that still run their own consume loop stay in the stack.
    for service in (
        "projection-delegation-writer",
        "projection-claude-hook-events-writer",
    ):
        assert service not in gates
        assert service in dev_stack_services()


def test_deploy_handler_retires_exactly_the_overlay_gated_services() -> None:
    """The handler's explicit map and the overlay's gates name the same pairs."""
    handler = _handler_profile_map()
    gates = _overlay_gates()
    for service, profile in gates.items():
        assert handler.get(profile) == service, profile


@pytest.mark.parametrize("profile", sorted(RUNTIME_WRITER_CONTRACTS))
def test_folded_writer_contract_attaches_to_main_only_when_folded(profile: str) -> None:
    """Main owns the writer's contract under the committed fold, and not without it."""
    contract = {
        "name": RUNTIME_WRITER_CONTRACTS[profile],
        "runtime_profiles": [profile],
    }
    folded = render_env(load_contract())[FOLDED_KEY]
    assert runtime_profile_owns_contract(
        contract, "main", environ={"ONEX_RUNTIME_FOLDED_PROFILES": folded}
    )
    assert not runtime_profile_owns_contract(contract, "main", environ={})
