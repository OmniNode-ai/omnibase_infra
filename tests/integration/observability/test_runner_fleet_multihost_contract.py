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
from pathlib import Path

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

REPO_ROOT = Path(__file__).parents[3]
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"
PRIMARY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

# GitHub's own architecture label, per architecture. A runner declares both this
# and the fleet's `arch-<TARGETARCH>` spelling, so a workflow can pin either
# vocabulary and get the same machine.
GITHUB_ARCH_LABEL = {"amd64": "x64", "arm64": "arm64"}


def _compose_for(prefix: str) -> Path:
    if prefix == load_runner_fleet_config(FLEET_CONFIG).runner_name_prefix:
        return PRIMARY_COMPOSE
    return REPO_ROOT / "docker" / f"docker-compose.runners-{prefix}.yml"


def _services(path: Path) -> dict[str, dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["services"]


@pytest.mark.integration
def test_every_declared_host_has_a_compose_file_defining_its_services() -> None:
    config = load_runner_fleet_config(FLEET_CONFIG)
    assert config.hosts, "the inventory must declare at least the primary host"

    for host in config.hosts:
        compose = _compose_for(host.runner_name_prefix)
        assert compose.is_file(), (
            f"host {host.host} declares prefix {host.runner_name_prefix!r} but "
            f"{compose.relative_to(REPO_ROOT)} does not exist"
        )
        named = [
            name
            for name in _services(compose)
            if re.fullmatch(rf"{re.escape(host.runner_name_prefix)}-\d+", name)
        ]
        assert len(named) >= host.expected_count, (
            f"host {host.host} declares expected_count={host.expected_count} but "
            f"{compose.relative_to(REPO_ROOT)} defines only {len(named)} service(s) "
            f"matching {host.runner_name_prefix}-<N>: {sorted(named)}"
        )


@pytest.mark.integration
def test_every_runner_registers_the_architecture_label_of_its_host() -> None:
    """A runner's arch label must match the host it is defined on.

    The label is how a workflow pins a CPU. A wrong one is worse than a missing
    one: missing means a job cannot be pinned, wrong means it is pinned to the
    other architecture and the job is placed where it will not work.
    """
    config = load_runner_fleet_config(FLEET_CONFIG)

    for host in config.hosts:
        compose = _compose_for(host.runner_name_prefix)
        for name, definition in _services(compose).items():
            if not re.fullmatch(rf"{re.escape(host.runner_name_prefix)}-\d+", name):
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

    The proof job pins the class label AND a host label, one matrix leg per
    host. Without the host label a single leg would be satisfied by whichever
    arm64 host happened to be idle, so a second host could sit broken behind a
    green check indefinitely -- undocumented-but-running in its other
    direction, and the reading would be worse than no check because it names a
    host it did not touch.

    This asserts the correspondence rather than leaving it to be remembered
    when the next host is added.
    """
    config = load_runner_fleet_config(FLEET_CONFIG)
    workflow = yaml.safe_load(PROOF_WORKFLOW.read_text(encoding="utf-8"))
    legs = set(workflow["jobs"]["arm64-verify-proof"]["strategy"]["matrix"]["host"])

    declared = {
        f"host-{host.host.split('.')[0].split('-')[-1]}"
        for host in config.hosts
        if host.arch.value == "arm64" and "verify" in host.classes
    }
    assert declared, "the inventory declares no arm64 verify host"

    compose_labels = set()
    for host in config.hosts:
        if host.arch.value != "arm64" or "verify" not in host.classes:
            continue
        compose = _compose_for(host.runner_name_prefix)
        for definition in _services(compose).values():
            labels = str(
                definition.get("environment", {}).get("RUNNER_LABELS", "")
            ).split(",")
            compose_labels.update(x for x in labels if x.startswith("host-"))

    missing = compose_labels - legs
    assert not missing, (
        f"{sorted(missing)} name arm64 verify hosts whose runners register that "
        f"label, but .github/workflows/arm64-verify-runner-proof.yml proves only "
        f"{sorted(legs)}; add a matrix leg per host"
    )
    stale = legs - compose_labels
    assert not stale, (
        f"{sorted(stale)} are proof legs for hosts no compose file registers; "
        "a leg for an absent host blocks on a machine that will never answer"
    )
