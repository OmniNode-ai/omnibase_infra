# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19494 (LO4): every main-profile compose runtime names its lane.

A node contract may declare ``runtime_lanes`` (OMN-19408). The auto-wiring
ownership filter attaches it only on a runtime whose ``ONEX_RUNTIME_LANE`` is in
that scope, and a runtime that declares no lane gets a discovery error per
lane-scoped contract, which the health monitor reports as a DEGRADED
``discovery_errors`` dimension. Every lane-scoped contract in omnimarket today
is owned by profile ``main`` (the filter in
``runtime/auto_wiring/profile_ownership.py`` gives a contract with no
``runtime_profiles`` to ``main`` and nothing to any other profile), so the
runtimes that need a lane are the ``RUNTIME_PROFILE: main`` ones.

This module renders each lane's compose stack the way the lane is deployed
(the base file layered with its overlay, in order) and asserts that every
main-profile service in it carries the lane the lane's own table row names.
It is a render-time check: it reads YAML and resolves compose interpolation, it
starts no container and touches no running lane.

Why the layering matters and not only the file: an ``environment`` key survives
a layer that does not repeat it. The pre-PR slot overlay layers over the dev
lane, which declares ``compose-dev``; the sim-202 overlay layers over the
dogfood file. A layer that forgot its own line would name itself as the lane
beneath it, and a slot would key its health verdict onto the dev lane's lab
lane-health row (OMN-19144).

``OMN-19144``'s ``test_omn19144_dev_lane_runtime_lane_identity`` keeps the
one-speaker rule for the dev lane (only ``omninode-runtime`` declares it); this
module holds the same shape for every other lane.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"

LANE_VARIABLE = "ONEX_RUNTIME_LANE"
PROFILE_VARIABLE = "RUNTIME_PROFILE"
MAIN_PROFILE = "main"

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|:\?|\?|-)([^{}]*))?\}")


class _Stack:
    """One lane's deployed compose stack and the lane its main runtime must name.

    ``files`` are layered in order, exactly as the lane's deploy path layers
    them. ``lane`` is the literal each main-profile service must resolve to, or
    ``env`` is the invoking environment the lane's interpolation reads.
    """

    def __init__(
        self,
        *,
        files: tuple[str, ...],
        lane: str,
        main_services: frozenset[str],
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.files = files
        self.lane = lane
        self.main_services = main_services
        self.env: Mapping[str, str] = env or {}


_MAIN = frozenset({"omninode-runtime"})

#: Compose project -> stack. The layering is the one
#: ``scripts/runtime_build/compose_files.sh`` and each file's own header give.
#: The lane values are the ids the runtime lane registry accepts today and the
#: ids ``knowledge-base-internal beta/plans/2026-09-26-runtime-lane-overlays-plan.md``
#: section 3.4 names for each lane.
STACKS: dict[str, _Stack] = {
    "dev": _Stack(
        files=("docker-compose.infra.yml", "docker-compose.dev-lane.yml"),
        lane="compose-dev",
        main_services=_MAIN,
    ),
    "dev-105": _Stack(
        files=(
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-105.yml",
        ),
        lane="compose-dev-105",
        main_services=_MAIN,
    ),
    "dev-200": _Stack(
        files=(
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-200.yml",
        ),
        lane="compose-dev-200",
        main_services=_MAIN,
    ),
    "dev-202": _Stack(
        files=(
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-202.yml",
        ),
        lane="compose-dev-202",
        main_services=_MAIN,
    ),
    "stability-test": _Stack(
        files=("docker-compose.infra.yml", "docker-compose.stability-test.yml"),
        lane="stability-test",
        main_services=_MAIN,
    ),
    "judge": _Stack(
        files=("docker-compose.infra.yml", "docker-compose.judge.yml"),
        lane="judge",
        main_services=_MAIN,
    ),
    "lakshman": _Stack(
        files=("docker-compose.infra.yml", "docker-compose.lakshman.yml"),
        lane="lakshman",
        main_services=_MAIN,
    ),
    "dogfood": _Stack(
        files=("docker-compose.dogfood.yml",),
        lane="dogfood",
        main_services=_MAIN,
    ),
    "sim-202": _Stack(
        files=("docker-compose.dogfood.yml", "docker-compose.sim-202.yml"),
        lane="sim-202",
        main_services=_MAIN,
    ),
    "prepr-1": _Stack(
        files=(
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.prepr.yml",
        ),
        lane="prepr-1",
        main_services=_MAIN,
        env={"ONEX_PREPR_SLOT": "1"},
    ),
    "prepr-2": _Stack(
        files=(
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.prepr.yml",
        ),
        lane="prepr-2",
        main_services=_MAIN,
        env={"ONEX_PREPR_SLOT": "2"},
    ),
}

#: Compose files that are not a deployed lane stack of their own. A file in
#: this set must define no service that runs the ``main`` profile, and the
#: reason it needs no lane is what the value says.
NOT_A_LANE_STACK: dict[str, str] = {
    "docker-compose.prod.yml": (
        "the retired .201 prod compose project: production is the AWS onex-prod "
        "namespace (OMN-18320) and OMN-19494's own scope drops onex-prod until "
        "staging works; a line here would also be a runtime-affecting path, "
        "whose merge fires the dev lane's rebuild trigger, and a prod change "
        "takes the prod promotion gate (CLAUDE.md rules 2a and 12)"
    ),
    "docker-compose.e2e.yml": (
        "an ephemeral CI stack whose runtime sets no RUNTIME_PROFILE, so it "
        "resolves to the `default` profile (runtime_profile.py), which owns no "
        "main-profile lane-scoped contract"
    ),
    "docker-compose.ci-bus.yml": "a broker, not a runtime (plan section 3.4)",
    "docker-compose.gateway.yml": "runs onex-gateway-forwarder, not the kernel",
    "docker-compose.gateway-attach-test-lane.yml": (
        "runs an attach proof script, not the kernel"
    ),
}


def _scalar(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _load(compose_file: str) -> dict[str, Any]:
    text = (DOCKER_DIR / compose_file).read_text(encoding="utf-8")
    # `!override` is a compose-only tag SafeLoader refuses; merge semantics are
    # applied by `_merged_services`, which is what the tag changes.
    document = yaml.safe_load(text.replace("!override", ""))
    assert isinstance(document, dict), f"{compose_file} did not parse to a mapping"
    return document


def _service_environment(service: object) -> dict[str, str]:
    if not isinstance(service, dict):
        return {}
    environment = service.get("environment")
    if isinstance(environment, dict):
        return {str(key): _scalar(value) for key, value in environment.items()}
    if isinstance(environment, list):
        pairs: dict[str, str] = {}
        for item in environment:
            name, separator, value = _scalar(item).partition("=")
            pairs[name] = value if separator else ""
        return pairs
    return {}


def _merged_main_environments(stack: _Stack) -> dict[str, dict[str, str]]:
    """Merge each service's ``environment`` across the stack, later file wins.

    Compose merges an ``environment`` mapping by key across layered files, so a
    key a later file does not repeat keeps the earlier file's value. Returns
    only services whose merged ``RUNTIME_PROFILE`` is ``main``.
    """
    merged: dict[str, dict[str, str]] = {}
    for compose_file in stack.files:
        services = _load(compose_file).get("services") or {}
        for name, service in services.items():
            merged.setdefault(str(name), {}).update(_service_environment(service))
    return {
        name: env
        for name, env in merged.items()
        if env.get(PROFILE_VARIABLE) == MAIN_PROFILE
    }


def _interpolate(value: str, env: Mapping[str, str]) -> str:
    """Resolve compose interpolation with exactly the caller's environment."""

    def replace(match: re.Match[str]) -> str:
        name, operator, argument = match.group(1), match.group(2), match.group(3)
        supplied = env.get(name)
        if operator in (":?", "?"):
            assert supplied, f"${{{name}}} is required and the render supplied none"
            return supplied
        if operator in (":-", "-"):
            return supplied if supplied else (argument or "")
        return supplied or ""

    return _VAR_RE.sub(replace, value)


def _stack_ids() -> list[str]:
    return sorted(STACKS)


@pytest.mark.parametrize("stack_id", _stack_ids())
def test_every_main_profile_runtime_of_the_stack_is_the_expected_set(
    stack_id: str,
) -> None:
    """Positive control: the render finds the main runtimes before absence proves anything."""
    stack = STACKS[stack_id]
    found = frozenset(_merged_main_environments(stack))

    assert found == stack.main_services, (
        f"stack {stack_id} ({' + '.join(stack.files)}) renders main-profile services "
        f"{sorted(found)}, expected {sorted(stack.main_services)}. A new main-profile "
        f"runtime needs its own {LANE_VARIABLE} line and an entry in STACKS."
    )


@pytest.mark.parametrize("stack_id", _stack_ids())
def test_every_main_profile_runtime_declares_its_lane(stack_id: str) -> None:
    stack = STACKS[stack_id]

    for name, environment in sorted(_merged_main_environments(stack).items()):
        raw = environment.get(LANE_VARIABLE)
        assert raw is not None, (
            f"stack {stack_id}: main-profile service {name} declares no "
            f"{LANE_VARIABLE}, so every lane-scoped node contract fails closed on "
            "it and the runtime reads DEGRADED (OMN-19494)."
        )
        resolved = _interpolate(raw, stack.env)
        assert resolved == stack.lane, (
            f"stack {stack_id}: {name} renders {LANE_VARIABLE}={resolved!r} "
            f"(source {raw!r}), expected {stack.lane!r}. An environment key survives a "
            "layer that does not repeat it, so a missing line names the lane "
            "underneath."
        )


@pytest.mark.parametrize("stack_id", _stack_ids())
def test_the_rendered_lane_is_a_registered_lane(stack_id: str) -> None:
    lane = STACKS[stack_id].lane
    assert lane in REGISTERED_RUNTIME_LANES, (
        f"stack {stack_id} expects lane {lane!r}, which is outside "
        f"{sorted(REGISTERED_RUNTIME_LANES)}; the ownership filter treats an "
        "unregistered lane as no lane and every lane-scoped contract fails closed "
        "(OMN-19408)."
    )


@pytest.mark.parametrize("stack_id", _stack_ids())
def test_a_lane_file_declares_its_lane_as_a_literal_not_a_shell_value(
    stack_id: str,
) -> None:
    """Only the prepr slot id interpolates, and only the slot token it names."""
    stack = STACKS[stack_id]
    own_file = stack.files[-1]
    services = _load(own_file).get("services") or {}
    declared = {
        name: _service_environment(service).get(LANE_VARIABLE)
        for name, service in services.items()
    }
    for name, raw in declared.items():
        if raw is None:
            continue
        if "${" in raw:
            assert own_file == "docker-compose.prepr.yml", (
                f"{own_file} service {name} declares {LANE_VARIABLE}={raw!r}; an "
                "exported shell value must not be able to rename a lane."
            )
            assert re.fullmatch(r"prepr-\$\{ONEX_PREPR_SLOT\}", raw), raw


def test_the_base_file_names_no_lane() -> None:
    """The base is every lane's floor and names none, with or without a default.

    A literal there would be inherited by any layer that forgot its own line,
    and an empty `${ONEX_RUNTIME_LANE:-}` pass-through is the ambient-override
    form `test_compose_no_silent_fallbacks` bans in this file. The lane
    declaration is each lane's own file; the base's render-time requirement is
    plan task LO5.
    """
    services = _load("docker-compose.infra.yml").get("services") or {}
    for name, service in services.items():
        assert LANE_VARIABLE not in _service_environment(service), (
            f"docker-compose.infra.yml service {name} declares {LANE_VARIABLE}; the "
            "base names no lane (OMN-19494)."
        )


def test_the_prepr_overlay_does_not_inherit_the_dev_lane_declaration() -> None:
    """The sharp edge: a slot layered over the dev lane must not read `compose-dev`."""
    for slot, lane in (("1", "prepr-1"), ("2", "prepr-2")):
        stack = STACKS[f"prepr-{slot}"]
        environment = _merged_main_environments(stack)["omninode-runtime"]
        assert _interpolate(environment[LANE_VARIABLE], stack.env) == lane
        assert lane != "compose-dev"


def test_sim_202_does_not_inherit_the_dogfood_declaration() -> None:
    stack = STACKS["sim-202"]
    environment = _merged_main_environments(stack)["omninode-runtime"]
    assert environment[LANE_VARIABLE] == "sim-202"


def test_every_compose_file_is_a_lane_stack_member_or_runs_no_main_runtime() -> None:
    """A new compose file with a main-profile runtime fails here until it is placed."""
    in_a_stack = {name for stack in STACKS.values() for name in stack.files}
    # The overlays that only exist inside a stack have no RUNTIME_PROFILE of
    # their own, so the stack renders above are what covers them.
    for compose_path in sorted(DOCKER_DIR.glob("docker-compose*.yml")):
        name = compose_path.name
        if name in in_a_stack:
            continue
        services = _load(name).get("services") or {}
        main_services = sorted(
            service_name
            for service_name, service in services.items()
            if _service_environment(service).get(PROFILE_VARIABLE) == MAIN_PROFILE
        )
        assert not main_services, (
            f"{name} defines main-profile service(s) {main_services} but is in no "
            "lane stack in STACKS, so nothing proves it declares its lane "
            "(OMN-19494). Add its stack, or give it no RUNTIME_PROFILE."
        )


def test_the_unclassified_files_have_a_recorded_reason() -> None:
    for name, reason in NOT_A_LANE_STACK.items():
        assert (DOCKER_DIR / name).exists(), f"{name} is gone; drop its entry"
        assert reason
    e2e_services = _load("docker-compose.e2e.yml").get("services") or {}
    assert PROFILE_VARIABLE not in _service_environment(e2e_services["runtime"]), (
        "the e2e runtime now sets RUNTIME_PROFILE; if it is `main` it needs its own "
        "stack entry and a lane"
    )


def test_the_render_has_positive_controls() -> None:
    """The zero in every absence above is only evidence if the render finds rows."""
    dev = _merged_main_environments(STACKS["dev"])["omninode-runtime"]
    assert dev[LANE_VARIABLE] == "compose-dev"
    assert len(dev) > 10, "the dev main runtime's environment did not merge"
    assert _interpolate("prepr-${ONEX_PREPR_SLOT:?x}", {"ONEX_PREPR_SLOT": "2"}) == (
        "prepr-2"
    )
    with pytest.raises(AssertionError):
        _interpolate("prepr-${ONEX_PREPR_SLOT:?x}", {})
