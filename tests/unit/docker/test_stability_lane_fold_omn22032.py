# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22032 (plan task T19): the stability lane takes the fold dev already holds.

The lane's main runtime folds the same role list dev folds, and every application
service the roles replace is retired by a policy-rendered profile on its compose
entry. The lane is found through its rendered policy prefix, never by spelling
its id here. Each check runs over the real policy contract and the real compose
overlay, and carries a failing control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_stability_lane_effects_folded_omn21288 import _lane_overlay

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "STABILITY_TEST_RUNTIME_MAIN"
RETIRED = {
    "runtime-effects",
    "runtime-worker",
    "projection-api",
    "tenant-projection-writer",
    "projection-live-events-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
}


def _lane_name(policy) -> str:
    project = render_env(policy)["STABILITY_TEST_COMPOSE_PROJECT"]
    return next(n for n, p in policy.profiles.items() if p.compose_project == project)


def _lane_main():
    policy = load_contract()
    return policy.profiles[_lane_name(policy)].processes["main"]


def _profiles_node(service: str) -> yaml.Node:
    document = yaml.compose(_lane_overlay().read_text(encoding="utf-8"))
    top = next(v for k, v in document.value if k.value == "services")
    node = next(v for k, v in top.value if k.value == service)
    return next(v for k, v in node.value if k.value == "profiles")


def _rendered_profiles(service: str) -> list[str]:
    """The profiles of ``service`` after ``${VAR:-default}`` takes the rendered env."""
    env = render_env(load_contract())
    node = _profiles_node(service)
    out = []
    for item in node.value:
        value = item.value
        assert value.startswith("${") and ":-" in value, value
        name, default = value[2:-1].split(":-", 1)
        out.append(env.get(name, default))
    return out


def test_policy_folds_the_dev_list_and_hands_over_the_consumer_groups() -> None:
    policy = load_contract()
    dev_list = set(policy.profiles["dev"].processes["main"].folded_profiles)
    main = _lane_main()
    assert set(main.folded_profiles) == dev_list
    assert {main.foldable_services[r] for r in main.folded_profiles} == RETIRED
    for role, instance in (
        ("effects", "runtime-effects"),
        ("workers", "runtime-worker"),
    ):
        handover = main.fold_handovers[role]
        assert handover.from_instance_id == instance
        assert handover.handover_rule == "last_ran"
    assert {"workflow.dispatch", "contract.update", "runtime.worker"} <= set(
        main.capabilities
    )
    rendered = render_env(policy)
    assert json.loads(rendered[f"{PREFIX}_FOLD_HANDOVERS"]).keys() == {
        "effects",
        "workers",
    }
    assert (ROOT / "docker" / "runtime-policy.env").read_text() == format_dotenv(
        rendered
    )


def test_every_retired_service_starts_under_no_profile_a_bring_up_requests() -> None:
    requested = {"runtime", "full"}
    for service in sorted(RETIRED - {"runtime-effects"}):
        assert not requested & set(_rendered_profiles(service)), service
    # runtime-effects keeps its literal opt-in profile (OMN-21288 step 1).
    node = _profiles_node("runtime-effects")
    assert [i.value for i in node.value] == ["folded-effects"]


def test_unfolded_policy_restores_each_service_to_every_bring_up() -> None:
    """Failing control: with no fold the same keys render the original profiles."""
    policy = load_contract()
    stability = _lane_name(policy)
    main = (
        policy.profiles[stability]
        .processes["main"]
        .model_copy(update={"folded_profiles": (), "fold_handovers": {}})
    )
    unfolded = policy.model_copy(
        update={
            "profiles": {
                **policy.profiles,
                stability: policy.profiles[stability].model_copy(
                    update={
                        "processes": {
                            **policy.profiles[stability].processes,
                            "main": main,
                        }
                    }
                ),
            }
        }
    )
    rendered = render_env(unfolded)
    for key in (k for k in rendered if k.startswith(PREFIX) and k.endswith("_PROFILE")):
        assert not rendered[key].startswith("folded-"), key


def test_main_mounts_the_volumes_of_the_retired_carriers_read_only() -> None:
    volumes = load_compose(_lane_overlay())["services"]["omninode-runtime"]["volumes"]
    assert (
        "stability_test_tenant_projection_data:/app/tenant-projection-data" in volumes
    )
    assert (
        "stability_test_tenant_projection_logs:/app/tenant-projection-logs:ro"
        in volumes
    )
    assert "worker_logs:/app/worker-logs:ro" in volumes


def _started_by_runtime_bring_up(env: dict[str, str]) -> list[str]:
    """Long-running services a ``runtime`` bring-up starts, base plus overlay.

    An overlay ``profiles`` list replaces the base list (the overlay spells
    ``!override`` where the base declares one); a one-shot (``restart: no``) is
    not a container that stays up.
    """
    base = load_compose(ROOT / "docker" / "docker-compose.infra.yml")["services"]
    overlay = load_compose(_lane_overlay())["services"]
    started = []
    for name in sorted(set(base) | set(overlay)):
        base_service, service = base.get(name) or {}, overlay.get(name) or {}
        profiles = service.get("profiles", base_service.get("profiles")) or []
        resolved = [
            env.get(p[2:-1].split(":-")[0], p[2:-1].split(":-")[-1])
            if p.startswith("${")
            else p
            for p in profiles
        ]
        restart = service.get("restart", base_service.get("restart"))
        if (not resolved or "runtime" in resolved) and str(restart) != "no":
            started.append(name)
    return started


def test_a_runtime_bring_up_starts_the_runtime_and_its_dependencies_only() -> None:
    """The count behind the ticket, read statically from the committed files."""
    started = set(_started_by_runtime_bring_up(render_env(load_contract())))
    assert not RETIRED & started
    assert started == {
        "omninode-runtime",
        "postgres",
        "redpanda",
        "valkey",
        "keycloak",
    }, sorted(started)


def test_the_count_control_reads_every_retired_service_before_the_fold() -> None:
    """Failing control: the same reader finds all of them under the pre-fold env."""
    env = {k: v.replace("folded-", "") for k, v in render_env(load_contract()).items()}
    env = {
        k: ("runtime" if k.endswith("_RUNTIME_PROFILE") else v) for k, v in env.items()
    }
    started = set(_started_by_runtime_bring_up(env))
    assert (RETIRED - {"runtime-effects"}) <= started
