# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the read-only verify runner's mount surface (OMN-18408).

`omninode-verify-runner-1` exists to take the scheduled VERIFY/CRON probes off
the single `omnibase-deploy` slot they were sharing with the release-train
DEPLOY jobs. The split is only worth having if the new runner is genuinely
narrower than the one it relieves: if it carried the same write surface it
would just be a second machine able to mutate the compose lanes, which is
exactly what CLAUDE.md rule 2a/12 says must stay singular.

So the load-bearing claim is an ABSENCE -- no `${DEPLOY_RUNNER_OMNI_HOME}`
bind, no OMNI_HOME env, no git `safe.directory` wiring for the private clone
tree -- and an absence cannot be proven by a test that only looks for one.
Every absence below is therefore paired with a positive control asserting the
same key IS present on `omninode-deploy-runner`. Without the control, renaming
a service or breaking the parse would report every absence as satisfied.

The parity half matters too, and is asserted in the same place: the verify
runner must keep the docker socket, the `host.docker.internal` host-gateway
alias and the operator env file, because the five moved jobs read container
revision labels and lane health endpoints off the host daemon through exactly
those. A "hardening" change that dropped one would turn every moved probe into
a false red that looks like a lane outage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

# OMN-18602 grew the class from one member to three. Every assertion in this
# file is parametrised over ALL of them rather than pinned to runner-1: they
# are interchangeable by construction, so a property that holds for one and not
# its peers is a defect that placement alone decides whether you ever see.
VERIFY_SERVICES = (
    "omninode-verify-runner-1",
    "omninode-verify-runner-2",
    "omninode-verify-runner-3",
)
# The runner this class was split off from. Positive control for every absence.
CONTROL_SERVICE = "omninode-deploy-runner"

# OMN-17477 added `arch-amd64`. `host-201` already scopes this runner to a host;
# the arch label scopes it to a CPU, and the two are deliberately separate
# vocabularies -- the arm64 verify runner on the other lab host carries
# `host-101` and `arch-arm64`, so a job that must pin a HOST and a job that must
# pin an ARCHITECTURE do not have to overload one label to mean both.
EXPECTED_LABELS = "self-hosted,omnibase-verify,host-201,linux,x64,arch-amd64"
DOCKER_SOCKET = "/var/run/docker.sock"
HOST_GATEWAY_ALIAS = "host.docker.internal:host-gateway"
OPERATOR_ENV_TARGET = "/run/omnibase-operator.env"
OMNI_HOME_VAR = "DEPLOY_RUNNER_OMNI_HOME"


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _service(name: str) -> dict[str, Any]:
    services = _compose()["services"]
    assert name in services, f"{name} is not defined in {COMPOSE_FILE}"
    service = services[name]
    assert isinstance(service, dict)
    return service


def _volumes(name: str) -> list[str]:
    return [str(v) for v in _service(name).get("volumes", [])]


def _environment(name: str) -> dict[str, str]:
    env = _service(name).get("environment", {})
    assert isinstance(env, dict), f"{name} uses list-form environment; update this test"
    return {str(k): str(v) for k, v in env.items()}


# ---------------------------------------------------------------------------
# The service exists at all, with the label that routes the moved jobs to it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_service_is_declared(verify_service: str) -> None:
    service = _service(verify_service)
    assert service.get("container_name") == verify_service


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_carries_the_verify_label_set(verify_service: str) -> None:
    assert _environment(verify_service)["RUNNER_LABELS"] == EXPECTED_LABELS


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_is_host_scoped_as_well_as_class_scoped(
    verify_service: str,
) -> None:
    """`omnibase-verify` alone stopped being a unique address on 2026-09-16.

    A second verify-class runner came online on another lab host that day
    carrying the same class label. The five jobs routed here probe THIS host's
    docker daemon and lane ports, so they require `host-201` too -- and this
    runner is the only thing that can supply it. Dropping it here would not
    break scheduling; it would let those jobs run somewhere they observe
    nothing, which reads as a lane outage.
    """
    labels = _environment(verify_service)["RUNNER_LABELS"].split(",")
    assert "host-201" in labels
    assert "omnibase-verify" in labels
    # Control: the deploy runner on the same host is deliberately NOT
    # host-scoped -- its label is already unique to one container, so a
    # host-201 there would be decoration. If this ever changes, the pairing
    # above needs rethinking rather than copying.
    assert "host-201" not in _environment(CONTROL_SERVICE)["RUNNER_LABELS"].split(",")


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_registers_under_its_own_name(verify_service: str) -> None:
    assert _environment(verify_service)["RUNNER_NAME"] == verify_service


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_registers_org_scoped_in_the_deploy_group(
    verify_service: str,
) -> None:
    """Group membership grants REPOSITORY visibility; the LABEL routes jobs.

    Reusing the existing `omnibase-deploy` group (OMN-18386's org-scoped
    registration path) is what gives this runner visibility of omnibase_infra
    without standing up a second org runner group. It does not make the runner
    eligible for deploy jobs -- `runs-on` matches labels, and this runner does
    not carry `omnibase-deploy`.
    """
    env = _environment(verify_service)
    assert env["GITHUB_ORG_URL"] == "https://github.com/OmniNode-ai"
    assert env["RUNNER_GROUP"] == _environment(CONTROL_SERVICE)["RUNNER_GROUP"]
    assert "omnibase-deploy" not in env["RUNNER_LABELS"].split(",")


# ---------------------------------------------------------------------------
# Absences, each with its positive control on omninode-deploy-runner.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_has_no_private_omni_home_bind(verify_service: str) -> None:
    assert not [v for v in _volumes(verify_service) if OMNI_HOME_VAR in v]
    # Positive control: the deploy runner does have it, so the parse works.
    assert [v for v in _volumes(CONTROL_SERVICE) if OMNI_HOME_VAR in v]


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_exports_no_omni_home(verify_service: str) -> None:
    assert "OMNI_HOME" not in _environment(verify_service)
    assert "OMNI_HOME" in _environment(CONTROL_SERVICE)


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_carries_no_private_clone_safe_directory_wiring(
    verify_service: str,
) -> None:
    """`safe.directory` entries name the private clones this runner cannot see.

    Carrying them would be harmless today and misleading tomorrow: the next
    reader would take them as evidence the clone tree is mounted here.
    """
    env = _environment(verify_service)
    assert not [k for k in env if k.startswith("GIT_CONFIG_")]
    assert [k for k in _environment(CONTROL_SERVICE) if k.startswith("GIT_CONFIG_")]


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_has_its_own_registration_credential_volume(
    verify_service: str,
) -> None:
    """A shared creds volume would hand this runner the deploy runner's cached
    registration, which is keyed to the `omnibase-deploy` label set."""
    verify_creds = [v for v in _volumes(verify_service) if "/.runner-creds" in v]
    control_creds = [v for v in _volumes(CONTROL_SERVICE) if "/.runner-creds" in v]
    assert len(verify_creds) == 1
    assert len(control_creds) == 1
    assert verify_creds[0] != control_creds[0]

    named = _compose()["volumes"]
    source = verify_creds[0].split(":", 1)[0]
    assert source in named, f"{source} is not declared under top-level volumes"


# ---------------------------------------------------------------------------
# Parity, in the other direction: what it MUST keep to do its job.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_keeps_the_docker_socket(verify_service: str) -> None:
    assert [v for v in _volumes(verify_service) if v.startswith(f"{DOCKER_SOCKET}:")]


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_keeps_the_host_gateway_alias(verify_service: str) -> None:
    assert HOST_GATEWAY_ALIAS in _service(verify_service).get("extra_hosts", [])
    assert _environment(verify_service)["LANE_PROBE_HOST"] == "host.docker.internal"


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_mounts_the_operator_env_read_only(verify_service: str) -> None:
    """Identical treatment to the deploy runner's (OMN-14983): the host file is
    bind-mounted read-only at a neutral path, and the root phase copies it into
    the runner-owned creds volume, because the `runner` job user cannot read a
    0600 file owned by the host operator uid once the entrypoint drops."""
    mounts = [v for v in _volumes(verify_service) if OPERATOR_ENV_TARGET in v]
    assert len(mounts) == 1
    assert mounts[0].endswith(":ro")
    assert "DEPLOY_RUNNER_OPERATOR_ENV_FILE" in mounts[0]
    assert (
        _environment(verify_service)["OMNIBASE_OPERATOR_ENV_FILE"]
        == _environment(CONTROL_SERVICE)["OMNIBASE_OPERATOR_ENV_FILE"]
    )


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_never_exposes_the_raw_operator_mount_to_job_steps(
    verify_service: str,
) -> None:
    """The scripts are pointed at the root-phase COPY, never the raw mount."""
    assert _environment(verify_service)["OMNIBASE_OPERATOR_ENV_FILE"] != (
        OPERATOR_ENV_TARGET
    )


# ---------------------------------------------------------------------------
# Class-level properties (OMN-18602). These are the ones that only exist once
# the class has more than one member, and they are the ones a copy-paste of a
# service block gets wrong.
# ---------------------------------------------------------------------------


def test_the_class_has_exactly_the_declared_members() -> None:
    """The count is a measurement, not a preference, so drift must be loud.

    Three is read off an Erlang-C curve against a measured offered load of 0.83
    erlangs with a 25-minute mean service time: N=1 predicts a 122-minute mean
    wait and N=3 predicts 0.7. The N=1 figure is a retrodiction -- the job this
    was sized for really did queue a median 29.1 minutes and a max of 101.3 on
    one runner -- which is what makes the rest of the curve usable.

    Adding or removing a member without revisiting that sizing is exactly the
    change this test exists to stop. It is not a refusal to grow the class; it
    is a refusal to grow it silently.
    """
    services = _compose()["services"]
    declared = {name for name in services if name.startswith("omninode-verify-runner-")}
    assert declared == set(VERIFY_SERVICES)


def test_every_class_member_has_a_distinct_credential_volume() -> None:
    """Pairwise, not merely distinct from the deploy runner's.

    The per-member test above compares each member against CONTROL_SERVICE, and
    all three would pass that while sharing one volume with each other. A
    shared registration cache names ONE runner, so the members would
    re-register over each other on recreate and the class would collapse to one
    online runner -- with the compose file still showing three.
    """
    sources = []
    for name in VERIFY_SERVICES:
        creds = [v for v in _volumes(name) if "/.runner-creds" in v]
        assert len(creds) == 1, f"{name} declares {len(creds)} creds mounts"
        sources.append(creds[0].split(":", 1)[0])
    assert len(set(sources)) == len(sources), f"shared creds volume: {sources}"

    named = _compose()["volumes"]
    for source in sources:
        assert source in named, f"{source} is not declared under top-level volumes"


def test_class_members_are_interchangeable_on_every_routing_field() -> None:
    """Placement must not depend on WHICH member GitHub picks.

    `runs-on` matches labels, and GitHub picks any idle runner carrying them.
    If one member's label set, runner group or org URL differed, a job would
    behave differently depending on placement -- a class that is a class only
    when you are lucky. The fields that may differ are exactly the identity
    ones, which the per-member tests above already pin.
    """
    identity = {"RUNNER_NAME"}
    envs = {name: _environment(name) for name in VERIFY_SERVICES}
    first = VERIFY_SERVICES[0]
    for name in VERIFY_SERVICES[1:]:
        shared_a = {k: v for k, v in envs[first].items() if k not in identity}
        shared_b = {k: v for k, v in envs[name].items() if k not in identity}
        assert shared_a == shared_b, (
            f"{name} diverges from {first}: {shared_a} vs {shared_b}"
        )


def test_no_class_member_carries_the_deploy_label() -> None:
    """The positive control for the whole split.

    `omnibase-deploy` must stay a single physical runner because it is the only
    serialisation guard that spans repositories -- a `concurrency` group is
    scoped to one repository and the sibling convergence guard is called from
    omnimarket. A verify member that picked up the deploy label would silently
    become a second machine able to run a release-train deploy.
    """
    for name in VERIFY_SERVICES:
        labels = _environment(name)["RUNNER_LABELS"].split(",")
        assert "omnibase-deploy" not in labels, f"{name} carries the deploy label"
    # Control: the label exists and is spelled this way on the runner that
    # legitimately carries it, so the assertion above is not vacuous.
    assert "omnibase-deploy" in _environment(CONTROL_SERVICE)["RUNNER_LABELS"].split(
        ","
    )
