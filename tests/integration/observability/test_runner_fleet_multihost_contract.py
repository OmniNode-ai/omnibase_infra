# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The host inventory and the compose files that realise it must agree (OMN-17477).

The declared inventory and the compose files are two halves of one statement,
written in different files by different changes, and nothing joined them until
this test. The failures that opens are all silent:

* a host declared with a prefix whose compose file defines services under some
  other name registers runners the fleet's own counting cannot see;
* a host declared ``arm64`` whose runners register the amd64 architecture label
  hands architecture-assuming jobs to the wrong CPU, which surfaces as a flaky
  test rather than a placement error;
* a declared count that does not match the number of service blocks is a fleet
  that reports healthy while short, because the router's degraded floor is
  computed from the declared number and the probe counts the real one.

Integration rather than unit because it crosses a file boundary the unit tests
deliberately do not: the inventory model knows nothing about compose, and the
compose test knows nothing about the inventory.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_host import (
    ModelRunnerFleetHost,
)

REPO_ROOT = Path(__file__).parents[3]
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"
PRIMARY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

# GitHub's own architecture label, per architecture. A runner declares both this
# and the fleet's `arch-<TARGETARCH>` spelling, so a workflow can pin either
# vocabulary and get the same machine.
GITHUB_ARCH_LABEL = {"amd64": "x64", "arm64": "arm64"}


def _compose_for(prefix: str, *, on_primary: bool = False) -> Path:
    # OMN-19895: a pool on the primary host lives in the primary compose file;
    # a pool anywhere else brings its own file named for its prefix, the same
    # rule deploy-runners.sh --host/--pool resolves.
    if (
        on_primary
        or prefix == load_runner_fleet_config(FLEET_CONFIG).runner_name_prefix
    ):
        return PRIMARY_COMPOSE
    return REPO_ROOT / "docker" / f"docker-compose.runners-{prefix}.yml"


def _declared_units() -> Iterator[tuple[ModelRunnerFleetHost, str, int, Path]]:
    """(host, prefix, expected_count, compose) for every host row and pool."""
    config = load_runner_fleet_config(FLEET_CONFIG)
    for host in config.hosts:
        on_primary = host.host == config.runner_host
        yield (
            host,
            host.runner_name_prefix,
            host.expected_count,
            _compose_for(host.runner_name_prefix),
        )
        for pool in host.pools:
            yield (
                host,
                pool.runner_name_prefix,
                pool.expected_count,
                _compose_for(pool.runner_name_prefix, on_primary=on_primary),
            )


def _services(path: Path) -> dict[str, dict[str, Any]]:
    loaded = cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))
    return cast("dict[str, dict[str, Any]]", loaded["services"])


@pytest.mark.integration
def test_every_declared_host_has_a_compose_file_defining_its_services() -> None:
    config = load_runner_fleet_config(FLEET_CONFIG)
    assert config.hosts, "the inventory must declare at least the primary host"

    for host, prefix, expected_count, compose in _declared_units():
        assert compose.is_file(), (
            f"host {host.host} declares prefix {prefix!r} but "
            f"{compose.relative_to(REPO_ROOT)} does not exist"
        )
        named = [
            name
            for name in _services(compose)
            if re.fullmatch(rf"{re.escape(prefix)}-\d+", name)
        ]
        assert len(named) >= expected_count, (
            f"host {host.host} declares expected_count={expected_count} for "
            f"{prefix} but {compose.relative_to(REPO_ROOT)} defines only "
            f"{len(named)} service(s) matching {prefix}-<N>: {sorted(named)}"
        )


@pytest.mark.integration
def test_every_runner_registers_the_architecture_label_of_its_host() -> None:
    """A runner's arch label must match the host it is defined on.

    The label is how a workflow pins a CPU. A wrong one is worse than a missing
    one: missing means a job cannot be pinned, wrong means it is pinned to the
    other architecture and the job is placed where it will not work.
    """
    for host, prefix, _count, compose in _declared_units():
        for name, definition in _services(compose).items():
            if not re.fullmatch(rf"{re.escape(prefix)}-\d+", name):
                continue
            labels = str(definition.get("environment", {}).get("RUNNER_LABELS", ""))
            if not labels:
                continue
            assert host.arch_label in labels.split(","), (
                f"{name} in {compose.relative_to(REPO_ROOT)} is on a "
                f"{host.arch.value} host but its RUNNER_LABELS do not carry "
                f"{host.arch_label!r}: {labels!r}"
            )
            assert GITHUB_ARCH_LABEL[host.arch.value] in labels.split(","), (
                f"{name} must also carry GitHub's own architecture label "
                f"{GITHUB_ARCH_LABEL[host.arch.value]!r}, so a workflow pinning "
                f"either vocabulary reaches the same machine: {labels!r}"
            )


@pytest.mark.integration
def test_no_runner_carries_an_architecture_label_it_is_not_on() -> None:
    """The negative half: nothing claims the OTHER architecture.

    Asserting only that the right label is present would pass a runner
    declaring BOTH, which is strictly worse than declaring neither -- it
    matches every arch-pinned job in the org.
    """
    config = load_runner_fleet_config(FLEET_CONFIG)
    all_arch_labels = {host.arch_label for host in config.hosts}

    for host in config.hosts:
        compose = _compose_for(host.runner_name_prefix)
        wrong = all_arch_labels - {host.arch_label}
        for name, definition in _services(compose).items():
            labels = set(
                str(definition.get("environment", {}).get("RUNNER_LABELS", "")).split(
                    ","
                )
            )
            if not labels & all_arch_labels:
                continue
            overlap = labels & wrong
            assert not overlap, (
                f"{name} in {compose.relative_to(REPO_ROOT)} carries "
                f"{sorted(overlap)}, which is not the architecture of any host "
                f"it is defined on ({host.arch.value})"
            )


PROOF_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "arm64-verify-runner-proof.yml"


@pytest.mark.integration
def test_every_arm64_verify_host_has_a_leg_in_the_proof_workflow() -> None:
    """A declared arm64 verify host must be proven individually, not as a class.

    One leg per host, each pinned to the class label AND that host's label, so a
    second host cannot sit broken behind a green check. Since OMN-19894 the legs
    are data (ARM64_VERIFY_PROOF_LEGS_JSON), not a literal host matrix, and the
    workflow's legs job holds them to the inventory at run time through
    .github/actions/resolve-arm64-proof-legs. This asserts the wiring, and that the
    script's reading of the inventory is the compose files' own.
    """
    import importlib.util

    action = REPO_ROOT / ".github" / "actions" / "resolve-arm64-proof-legs"
    spec = importlib.util.spec_from_file_location(
        "resolve_arm64_proof_legs", action / "resolve_arm64_proof_legs.py"
    )
    assert spec is not None and spec.loader is not None
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)

    config = load_runner_fleet_config(FLEET_CONFIG)
    workflow = yaml.safe_load(PROOF_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    proof = jobs["arm64-verify-proof"]
    assert proof["needs"] == "legs"
    assert proof["strategy"]["matrix"]["leg"] == (
        "${{ fromJSON(needs.legs.outputs.legs) }}"
    )
    assert proof["runs-on"] == "${{ matrix.leg.runs_on }}"
    step = next(step for step in jobs["legs"]["steps"] if step.get("id") == "legs")
    assert step["uses"] == "./.github/actions/resolve-arm64-proof-legs"
    assert step["with"] == {"legs-json": "${{ vars.ARM64_VERIFY_PROOF_LEGS_JSON }}"}

    compose_labels: set[str] = set()
    for host in config.hosts:
        if host.arch.value != "arm64" or "verify" not in host.classes:
            continue
        compose = _compose_for(host.runner_name_prefix)
        for definition in _services(compose).values():
            labels = str(
                definition.get("environment", {}).get("RUNNER_LABELS", "")
            ).split(",")
            compose_labels.update(x for x in labels if x.startswith("host-"))
    assert compose_labels, "the inventory declares no arm64 verify host"
    assert resolver.declared_host_labels() == compose_labels
