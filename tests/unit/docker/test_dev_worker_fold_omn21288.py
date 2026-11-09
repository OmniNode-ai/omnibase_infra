# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288 (plan task T9): primary dev's worker role runs inside the one runtime.

The retired ``runtime-worker`` service resolved a set of secret names, declared
three capabilities, held a handover-eligible consumer group and kept its logs on
a named volume. This file pins each of those to the folded main runtime, pins the
removal of the separate service from the dev stack, and carries a failing control
for each check, over the real policy contract, the real dev compose files and the
real installer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
OVERLAY = DOCKER / "docker-compose.dev-tenant-fold.yml"
DEV_LANE = DOCKER / "docker-compose.dev-lane.yml"
SERVICE = "runtime-worker"
ROLE = "workers"
RETIRED_KEY = "DEV_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON"
MAIN_STORE_KEY = "DEV_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON"
PROFILE_KEY = "DEV_RUNTIME_MAIN_RUNTIME_WORKER"


def _sources(config: dict) -> dict[str, dict]:
    return {m["logical_name"]: m["source"] for m in config["mappings"]}


def _missing_names(retired: dict, main: dict) -> list[str]:
    """Every name the retired service resolved that the runtime does not, by name."""
    return sorted(
        name
        for name, source in _sources(retired).items()
        if _sources(main).get(name) != source
    )


def _missing_capabilities(retired: str, main: str) -> list[str]:
    held = set(main.split(","))
    return sorted(c for c in retired.split(",") if c not in held)


# -- the policy half: the fold is declared, handed over and retired ----------


def test_policy_folds_the_worker_into_dev_main_and_hands_over_the_groups() -> None:
    policy = load_contract()
    process = policy.profiles["dev"].processes["main"]
    assert ROLE in process.folded_profiles
    assert process.foldable_services[ROLE] == SERVICE
    handover = process.fold_handovers[ROLE]
    assert handover.from_instance_id == "runtime-worker"
    # Queued workflow events are neither skipped nor run twice: resume from the
    # commits of the side that last ran, never a reset.
    assert handover.handover_rule == "last_ran"
    rendered = render_env(policy)
    assert ROLE in rendered["DEV_RUNTIME_MAIN_FOLDED_PROFILES"].split(",")
    assert (
        json.loads(rendered["DEV_RUNTIME_MAIN_FOLD_HANDOVERS"])[ROLE][
            "from_instance_id"
        ]
        == "runtime-worker"
    )
    assert (DOCKER / "runtime-policy.env").read_text() == format_dotenv(rendered)


def test_effects_stay_folded_beside_the_worker() -> None:
    """Positive control: the T8 fold this one stacks on is still in the policy."""
    process = load_contract().profiles["dev"].processes["main"]
    assert {"effects", ROLE} <= set(process.folded_profiles)


def test_worker_is_gated_by_the_policy_not_by_a_literal() -> None:
    document = yaml.compose(DEV_LANE.read_text())
    top = next(v for k, v in document.value if k.value == "services")
    worker = next(v for k, v in top.value if k.value == SERVICE)
    profiles = next(v for k, v in worker.value if k.value == "profiles")
    assert profiles.tag == "!override"  # a plain list would merge with runtime/full
    assert [item.value for item in profiles.value] == [
        f"${{{PROFILE_KEY}_RUNTIME_PROFILE:-runtime}}",
        f"${{{PROFILE_KEY}_FULL_PROFILE:-full}}",
    ]
    rendered = render_env(load_contract())
    assert rendered[f"{PROFILE_KEY}_RUNTIME_PROFILE"] == f"folded-{ROLE}-runtime"
    assert rendered[f"{PROFILE_KEY}_FULL_PROFILE"] == f"folded-{ROLE}-full"


def test_unfolded_policy_restores_the_worker_to_every_bring_up() -> None:
    """Rollback of the fold in policy: the same keys render the original profiles."""
    policy = load_contract()
    process = policy.profiles["dev"].processes["main"]
    kept = tuple(p for p in process.folded_profiles if p != ROLE)
    handovers = {k: v for k, v in process.fold_handovers.items() if k != ROLE}
    rolled_back = process.model_copy(
        update={"folded_profiles": kept, "fold_handovers": handovers}
    )
    dev = policy.profiles["dev"].model_copy(
        update={"processes": {**policy.profiles["dev"].processes, "main": rolled_back}}
    )
    rendered = render_env(
        policy.model_copy(update={"profiles": {**policy.profiles, "dev": dev}})
    )
    assert rendered[f"{PROFILE_KEY}_RUNTIME_PROFILE"] == "runtime"
    assert rendered[f"{PROFILE_KEY}_FULL_PROFILE"] == "full"


# -- what the retired service carried now lives in main -----------------------


def test_runtime_resolves_every_name_the_retired_worker_declared() -> None:
    rendered = render_env(load_contract())
    retired = json.loads(rendered[RETIRED_KEY])
    main = json.loads(rendered[MAIN_STORE_KEY])
    assert _sources(retired), "positive control: the retired service declares names"
    assert _missing_names(retired, main) == []


def test_comparison_names_a_secret_main_does_not_resolve() -> None:
    """FAILING CONTROL: the comparison goes red, by name, on a dropped entry."""
    rendered = render_env(load_contract())
    retired = json.loads(rendered[RETIRED_KEY])
    main = json.loads(rendered[MAIN_STORE_KEY])
    dropped = sorted(_sources(main))[0]
    short = {
        **main,
        "mappings": [m for m in main["mappings"] if m["logical_name"] != dropped],
    }
    assert _missing_names(retired, short) == [dropped]


def test_main_declares_every_capability_the_worker_declared() -> None:
    rendered = render_env(load_contract())
    retired = rendered["DEV_RUNTIME_WORKER_CAPABILITIES"]
    main = rendered["DEV_RUNTIME_MAIN_CAPABILITIES"]
    assert "workflow.dispatch" in retired, "positive control"
    assert _missing_capabilities(retired, main) == []


def test_capability_comparison_names_a_capability_main_lacks() -> None:
    """FAILING CONTROL: dropping one worker capability from main is reported by name."""
    rendered = render_env(load_contract())
    retired = rendered["DEV_RUNTIME_WORKER_CAPABILITIES"]
    short = ",".join(
        c
        for c in rendered["DEV_RUNTIME_MAIN_CAPABILITIES"].split(",")
        if c != "workflow.dispatch"
    )
    assert _missing_capabilities(retired, short) == ["workflow.dispatch"]


def test_main_attaches_every_worker_node_with_its_contract_unchanged() -> None:
    discovered = discover_contracts_from_paths(
        sorted((ROOT / "src/omnibase_infra/nodes").glob("*/contract.yaml"))
    )
    assert not discovered.errors
    env = {
        "ONEX_RUNTIME_LANE": dev_stack_services()["omninode-runtime"]["environment"][
            "ONEX_RUNTIME_LANE"
        ]
    }
    unfolded = filter_manifest_for_runtime_profile(discovered, "main", environ=env)
    workers = filter_manifest_for_runtime_profile(discovered, ROLE, environ=env)
    assert workers.manifest.contracts, "A fold without worker nodes proves nothing"
    env["ONEX_RUNTIME_FOLDED_PROFILES"] = render_env(load_contract())[
        "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
    ]
    folded = filter_manifest_for_runtime_profile(discovered, "main", environ=env)
    assert not folded.manifest.errors
    assert ROLE in set(folded.folded_profiles)
    attached = {c.name: c for c in folded.manifest.contracts}
    unfolded_names = {c.name for c in unfolded.manifest.contracts}
    for contract in workers.manifest.contracts:
        assert contract.name in attached, contract.name
        # The contract itself, subscriptions and handlers included, is main's.
        assert attached[contract.name] == contract
    assert unfolded_names - {c.name for c in workers.manifest.contracts} <= set(
        attached
    )


def test_main_keeps_the_retired_workers_logs() -> None:
    main = load_compose(OVERLAY)["services"]["omninode-runtime"]
    assert "worker_logs:/app/worker-logs:ro" in main["volumes"]
    # The volume is declared where the base stack declares it; nothing removes it.
    volumes = load_compose(DOCKER / "docker-compose.infra.yml")["volumes"]
    assert "worker_logs" in volumes


def test_main_keeps_its_own_identity_after_the_fold() -> None:
    env = dev_stack_services()["omninode-runtime"]["environment"]
    assert env["RUNTIME_PROFILE"] == "main"
    assert env["KAFKA_INSTANCE_ID"] == "runtime-main"


# -- the dev stack check ------------------------------------------------------


def test_dev_stack_has_no_worker_service() -> None:
    assert SERVICE not in dev_stack_services()
    assert "omninode-runtime" in dev_stack_services()  # positive control


def test_dev_stack_check_refuses_a_worker_service_that_stays_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILING CONTROL: with the service's gate removed it is in the dev stack."""
    from tests.unit.docker import test_dev_tenant_projection_fold_omn21288 as shape

    text = DEV_LANE.read_text()
    gate = re.search(
        rf"  {SERVICE}:\n(?:    .*\n|\n)*?    profiles: !override\n(?:      - .*\n){{2}}",
        text,
    )
    assert gate, "the gate this control removes must be present"
    ungated = tmp_path / "docker-compose.dev-lane.yml"
    ungated.write_text(
        text[: gate.start()]
        + f"  {SERVICE}:\n    build: *dev_lane_composed_runtime_build\n"
        + text[gate.end() :]
    )
    monkeypatch.setattr(
        shape,
        "DEV_COMPOSE_FILES",
        (shape.DEV_COMPOSE_FILES[0], ungated, shape.DEV_COMPOSE_FILES[2]),
    )
    assert SERVICE in dev_stack_services()


def test_primary_dev_installer_derives_the_worker_from_the_policy() -> None:
    refresh = (ROOT / "scripts/runtime_build/refresh_dev_lane.sh").read_text()
    line = next(
        line
        for line in refresh.splitlines()
        if line.startswith("readonly CORE_SERVICES=")
    )
    # The retired container is named in the installer only behind the policy's fold.
    assert f"== *,{ROLE},*" in line
    assert "== *,effects,*" in line


def _core_services(folded: str) -> list[str]:
    """Run the installer's own CORE_SERVICES line under a given fold policy."""
    import subprocess

    refresh = (ROOT / "scripts/runtime_build/refresh_dev_lane.sh").read_text()
    line = next(
        line
        for line in refresh.splitlines()
        if line.startswith("readonly CORE_SERVICES=")
    )
    result = subprocess.run(
        ["bash", "-c", f'{line}\nprintf "%s\\n" "${{CORE_SERVICES[@]}}"'],
        check=True,
        capture_output=True,
        text=True,
        env={"DEV_RUNTIME_MAIN_FOLDED_PROFILES": folded, "PATH": "/usr/bin:/bin"},
    )
    return result.stdout.split()


def test_installer_core_services_follow_the_fold() -> None:
    folded = _core_services(
        render_env(load_contract())["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]
    )
    assert folded == ["omninode-runtime", "projection-api"]


def test_installer_core_services_return_the_worker_when_unfolded() -> None:
    """FAILING CONTROL: a policy that does not fold the worker keeps its service."""
    assert _core_services("tenant-projection,effects") == [
        "omninode-runtime",
        SERVICE,
        "projection-api",
    ]
