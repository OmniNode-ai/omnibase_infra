# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22032 (T19): the stability lane takes the roles dev folds into its main runtime.

The lane is found through its rendered policy prefix (``STABILITY_TEST_``) and its
compose project, never by spelling its id here. The stability overlay gates each
folded service from the runtime policy, so a bring-up that requests ``runtime`` or
``full`` starts none of them and the roles run inside ``omninode-runtime``.
"""

from __future__ import annotations

import re
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
BASE = ROOT / "docker" / "docker-compose.infra.yml"
FOLDED_KEY = "STABILITY_TEST_RUNTIME_MAIN_FOLDED_PROFILES"
PROJECTION_ROLES = {
    "tenant-projection": "tenant-projection-writer",
    "projection-writer-live-events": "projection-live-events-writer",
    "projection-writer-registration": "projection-registration-writer",
    "projection-writer-savings": "projection-savings-writer",
    "projection-writer-tenant-credentials": "projection-tenant-credentials-writer",
    "projection-writer-tenant-registry": "projection-tenant-registry-writer",
    "projection-writer-delegation": "projection-delegation-writer",
    "projection-api": "projection-api",
}
# One-shots the runtime lists as service_completed_successfully: they run and exit.
EXITED_ONE_SHOTS = {
    "forward-migration",
    "intelligence-migration",
    "redpanda-partition-cap",
}
# Dependencies and the migration gate the runtime waits for, plus the runtime and
# its one worker: what the lane keeps running once the projection roles are folded.
REMAINING_CONTAINERS = {
    "keycloak",
    "migration-gate",
    "omninode-runtime",
    "postgres",
    "redpanda",
    "runtime-worker",
    "valkey",
}


def _lane_project() -> str:
    return render_env(load_contract())["STABILITY_TEST_COMPOSE_PROJECT"]


def _lane_overlay() -> Path:
    project = _lane_project()
    for path in sorted((ROOT / "docker").glob("docker-compose.*.yml")):
        match = re.search(r"(?m)^name:\s*(\S+)\s*$", path.read_text(encoding="utf-8"))
        if match and match.group(1) == project:
            return path
    raise AssertionError(f"no compose overlay names project {project!r}")


def _running_services(policy: dict[str, str], profile: str = "runtime") -> set[str]:
    """Services a ``--profile <profile>`` bring-up starts and keeps running.

    The base and the lane overlay are merged the way compose merges them for
    ``profiles``: a service the overlay lists replaces the base's list (every
    overlay entry that matters here carries ``!override`` or is new). The
    one-shots the runtime waits on to complete exit and are not counted; the
    migration gate stays alive for its healthcheck and is counted.
    """
    merged: dict[str, dict] = {
        name: dict(service or {})
        for name, service in (load_compose(BASE).get("services") or {}).items()
    }
    for name, overlay in (load_compose(_lane_overlay()).get("services") or {}).items():
        service = merged.setdefault(name, {})
        service.update({k: v for k, v in (overlay or {}).items() if k != "environment"})
    running: set[str] = set()
    for name, service in merged.items():
        profiles = [
            re.sub(
                r"\$\{([^:}]+):-([^}]+)\}",
                lambda m: policy.get(m[1], m[2]),
                str(p),
            )
            for p in service.get("profiles", [])
        ]
        if profiles and profile not in profiles:
            continue
        if name in EXITED_ONE_SHOTS:
            continue
        running.add(name)
    return running


def test_stability_policy_folds_the_dev_projection_roles_in_dev_order() -> None:
    policy = load_contract()
    dev = policy.profiles["dev"].processes["main"]
    stability = next(
        profile
        for profile in policy.profiles.values()
        if profile.compose_project == _lane_project()
    ).processes["main"]
    dev_roles = tuple(role for role in dev.folded_profiles if role != "effects")
    assert stability.folded_profiles == ("effects", *dev_roles)
    assert set(PROJECTION_ROLES) <= set(stability.folded_profiles)
    assert stability.foldable_services == PROJECTION_ROLES
    assert stability.foldable_services == {
        role: service
        for role, service in dev.foldable_services.items()
        if role in PROJECTION_ROLES
    }
    rendered = render_env(policy)
    assert rendered[FOLDED_KEY].split(",") == list(stability.folded_profiles)
    assert (ROOT / "docker" / "runtime-policy.env").read_text(
        encoding="utf-8"
    ) == format_dotenv(rendered)


def test_each_folded_service_renders_a_folded_profile_pair() -> None:
    rendered = render_env(load_contract())
    for service in PROJECTION_ROLES.values():
        key = f"STABILITY_TEST_RUNTIME_MAIN_{service.upper().replace('-', '_')}"
        for active in ("runtime", "full"):
            value = rendered[f"{key}_{active.upper()}_PROFILE"]
            assert re.fullmatch(rf"folded-[a-z-]+-{active}", value), (key, value)


def test_a_runtime_bring_up_starts_none_of_the_folded_services() -> None:
    policy = render_env(load_contract())
    for profile in ("runtime", "full"):
        running = _running_services(policy, profile)
        assert not running & set(PROJECTION_ROLES.values()), profile
        assert "omninode-runtime" in running, profile


def test_the_lane_keeps_the_runtime_its_worker_and_its_dependencies() -> None:
    assert _running_services(render_env(load_contract())) == REMAINING_CONTAINERS


def test_without_the_policy_the_folded_services_start_so_the_gate_is_what_stops_them() -> (
    None
):
    """Positive control: with no rendered policy the same files start every folded service."""
    unfolded = _running_services({})
    assert set(PROJECTION_ROLES.values()) <= unfolded
    assert len(unfolded) == len(REMAINING_CONTAINERS) + len(PROJECTION_ROLES)


def test_main_carries_the_tenant_carriers_state_and_reads_the_fold() -> None:
    main = load_compose(_lane_overlay())["services"]["omninode-runtime"]
    assert main["environment"]["ONEX_RUNTIME_FOLDED_PROFILES"] == (
        f"${{{FOLDED_KEY}:?runtime policy contract must set {FOLDED_KEY}}}"
    )
    volumes = main["volumes"]
    assert (
        "stability_test_tenant_projection_data:/app/tenant-projection-data" in volumes
    )
    assert (
        "stability_test_tenant_projection_logs:/app/tenant-projection-logs:ro"
        in volumes
    )
    # The carrier's volumes stay declared: retirement removes no volume.
    declared = load_compose(_lane_overlay())["volumes"]
    assert "stability_test_tenant_projection_data" in declared
    assert "stability_test_tenant_projection_logs" in declared


def test_projection_api_profile_list_replaces_the_base_list() -> None:
    # The tag matters as well as the value: a plain list merges with the base
    # service's runtime/full profiles and would keep the container up.
    document = yaml.compose(_lane_overlay().read_text(encoding="utf-8"))
    assert isinstance(document, yaml.MappingNode)
    services = next(v for k, v in document.value if k.value == "services")
    assert isinstance(services, yaml.MappingNode)
    for name in ("projection-api", "tenant-projection-writer"):
        service = next(v for k, v in services.value if k.value == name)
        profiles = next(v for k, v in service.value if k.value == "profiles")
        assert profiles.tag == "!override", name


def test_the_stack_shape_check_still_judges_every_folded_service_in_the_overlay() -> (
    None
):
    """The services stay declared (gated, not deleted), so the baseline counts hold."""
    data = yaml.safe_load(
        (ROOT / "config/validation/stack_shape.yaml").read_text(encoding="utf-8")
    )
    allowlist = ModelStackShapeData(
        runtime_services=frozenset(data["runtime"]["services"]),
        max_replicas=int(data["runtime"]["max_replicas"]),
        dependencies=frozenset(data["dependencies"]),
        one_shots=frozenset(n for n in data["one_shots"] if "*" not in n),
        lanes={},
        baseline=(),
    )
    rules = {r.rule_id: r for r in ValidatorStackShape(root=ROOT).contract.rules}
    overlay = _lane_overlay()
    declared = load_compose(overlay)["services"]
    refused = check_compose(
        allowlist, str(overlay.relative_to(ROOT)), declared, ["stability"], rules
    )
    refused_names = {
        service
        for service in PROJECTION_ROLES.values()
        if any(service in issue.message for issue in refused)
    }
    # With an empty baseline every folded service is refused by name; the real
    # baseline admits them only while they are declared, which is why they are
    # gated and not removed from this file.
    assert refused_names == set(PROJECTION_ROLES.values())
