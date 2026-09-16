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

VERIFY_SERVICE = "omninode-verify-runner-1"
# The runner this one was split off from. Positive control for every absence.
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


def test_verify_runner_service_is_declared() -> None:
    service = _service(VERIFY_SERVICE)
    assert service.get("container_name") == VERIFY_SERVICE


def test_verify_runner_carries_the_verify_label_set() -> None:
    assert _environment(VERIFY_SERVICE)["RUNNER_LABELS"] == EXPECTED_LABELS


def test_verify_runner_is_host_scoped_as_well_as_class_scoped() -> None:
    """`omnibase-verify` alone stopped being a unique address on 2026-09-16.

    A second verify-class runner came online on another lab host that day
    carrying the same class label. The five jobs routed here probe THIS host's
    docker daemon and lane ports, so they require `host-201` too -- and this
    runner is the only thing that can supply it. Dropping it here would not
    break scheduling; it would let those jobs run somewhere they observe
    nothing, which reads as a lane outage.
    """
    labels = _environment(VERIFY_SERVICE)["RUNNER_LABELS"].split(",")
    assert "host-201" in labels
    assert "omnibase-verify" in labels
    # Control: the deploy runner on the same host is deliberately NOT
    # host-scoped -- its label is already unique to one container, so a
    # host-201 there would be decoration. If this ever changes, the pairing
    # above needs rethinking rather than copying.
    assert "host-201" not in _environment(CONTROL_SERVICE)["RUNNER_LABELS"].split(",")


def test_verify_runner_registers_under_its_own_name() -> None:
    assert _environment(VERIFY_SERVICE)["RUNNER_NAME"] == VERIFY_SERVICE


def test_verify_runner_registers_org_scoped_in_the_deploy_group() -> None:
    """Group membership grants REPOSITORY visibility; the LABEL routes jobs.

    Reusing the existing `omnibase-deploy` group (OMN-18386's org-scoped
    registration path) is what gives this runner visibility of omnibase_infra
    without standing up a second org runner group. It does not make the runner
    eligible for deploy jobs -- `runs-on` matches labels, and this runner does
    not carry `omnibase-deploy`.
    """
    env = _environment(VERIFY_SERVICE)
    assert env["GITHUB_ORG_URL"] == "https://github.com/OmniNode-ai"
    assert env["RUNNER_GROUP"] == _environment(CONTROL_SERVICE)["RUNNER_GROUP"]
    assert "omnibase-deploy" not in env["RUNNER_LABELS"].split(",")


# ---------------------------------------------------------------------------
# Absences, each with its positive control on omninode-deploy-runner.
# ---------------------------------------------------------------------------


def test_verify_runner_has_no_private_omni_home_bind() -> None:
    assert not [v for v in _volumes(VERIFY_SERVICE) if OMNI_HOME_VAR in v]
    # Positive control: the deploy runner does have it, so the parse works.
    assert [v for v in _volumes(CONTROL_SERVICE) if OMNI_HOME_VAR in v]


def test_verify_runner_exports_no_omni_home() -> None:
    assert "OMNI_HOME" not in _environment(VERIFY_SERVICE)
    assert "OMNI_HOME" in _environment(CONTROL_SERVICE)


def test_verify_runner_carries_no_private_clone_safe_directory_wiring() -> None:
    """`safe.directory` entries name the private clones this runner cannot see.

    Carrying them would be harmless today and misleading tomorrow: the next
    reader would take them as evidence the clone tree is mounted here.
    """
    env = _environment(VERIFY_SERVICE)
    assert not [k for k in env if k.startswith("GIT_CONFIG_")]
    assert [k for k in _environment(CONTROL_SERVICE) if k.startswith("GIT_CONFIG_")]


def test_verify_runner_has_its_own_registration_credential_volume() -> None:
    """A shared creds volume would hand this runner the deploy runner's cached
    registration, which is keyed to the `omnibase-deploy` label set."""
    verify_creds = [v for v in _volumes(VERIFY_SERVICE) if "/.runner-creds" in v]
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


def test_verify_runner_keeps_the_docker_socket() -> None:
    assert [v for v in _volumes(VERIFY_SERVICE) if v.startswith(f"{DOCKER_SOCKET}:")]


def test_verify_runner_keeps_the_host_gateway_alias() -> None:
    assert HOST_GATEWAY_ALIAS in _service(VERIFY_SERVICE).get("extra_hosts", [])
    assert _environment(VERIFY_SERVICE)["LANE_PROBE_HOST"] == "host.docker.internal"


def test_verify_runner_mounts_the_operator_env_read_only() -> None:
    """Identical treatment to the deploy runner's (OMN-14983): the host file is
    bind-mounted read-only at a neutral path, and the root phase copies it into
    the runner-owned creds volume, because the `runner` job user cannot read a
    0600 file owned by the host operator uid once the entrypoint drops."""
    mounts = [v for v in _volumes(VERIFY_SERVICE) if OPERATOR_ENV_TARGET in v]
    assert len(mounts) == 1
    assert mounts[0].endswith(":ro")
    assert "DEPLOY_RUNNER_OPERATOR_ENV_FILE" in mounts[0]
    assert (
        _environment(VERIFY_SERVICE)["OMNIBASE_OPERATOR_ENV_FILE"]
        == _environment(CONTROL_SERVICE)["OMNIBASE_OPERATOR_ENV_FILE"]
    )


def test_verify_runner_never_exposes_the_raw_operator_mount_to_job_steps() -> None:
    """The scripts are pointed at the root-phase COPY, never the raw mount."""
    assert _environment(VERIFY_SERVICE)["OMNIBASE_OPERATOR_ENV_FILE"] != (
        OPERATOR_ENV_TARGET
    )
