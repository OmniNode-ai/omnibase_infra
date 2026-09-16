# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the credential-free customer-plane runner pair (OMN-18392).

`omninode_infra`'s customer-plane reachability probe measures which internal
surfaces refuse a TCP connect *from a customer's vantage*, and its first step
asserts that the runner holds no platform credential. That assertion is the
whole value of the probe: a runner carrying a kubeconfig, a cloud role or the
host docker socket is an operator session, and its refusals prove nothing.

The probe was pinned to a GitHub-hosted runner for exactly that identity. The
2026-09-14 operator ruling that private repositories never run on hosted
runners left it dark, and the lab fleet could not take it: every one of the 88
`omninode-runner-N` containers mounts the lab credentials directory and exports
`KUBECONFIG` (OMN-18188), which that step correctly refuses. Clearing the
variable while the file stayed mounted would be a green gate asserting nothing.

So the pair below holds nothing. These assertions are the falsifier for that
claim, and they are written against the *resolved* compose config rather than
a regex, because a regex cannot tell a live per-service value from a comment.

Each test that asserts an ABSENCE is paired with a positive control asserting
the same key IS present on a general-pool fleet runner. An empty result is not
evidence of absence: without the control, a renamed service or a broken parse
would report every absence as satisfied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

CUSTOMER_PLANE_SERVICES = (
    "omninode-customer-plane-runner-1",
    "omninode-customer-plane-runner-2",
)
# The general pool, used only as the positive control for every absence below.
CONTROL_SERVICE = "omninode-runner-1"

EXPECTED_LABELS = "self-hosted,omnibase-customer-plane,linux,x64"
LAB_CREDENTIALS_TARGET = "/home/runner/.lab-credentials"
DOCKER_SOCKET = "/var/run/docker.sock"

# The eight variables the probe's "Confirm this runner holds no platform
# credential" step refuses, transcribed from that step. A variable added there
# and not here would be unpinned, which is why the list is spelled out rather
# than imported across the repository boundary.
PROBE_FORBIDDEN_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "KUBECONFIG",
    "KAFKA_SASL_PASSWORD",
    "POSTGRES_PASSWORD",
)


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _service(name: str) -> dict[str, Any]:
    services = _compose()["services"]
    assert name in services, f"{name} is not defined in {COMPOSE_FILE}"
    return services[name]


def _volumes(service: dict[str, Any]) -> list[str]:
    return [v for v in service.get("volumes", []) if isinstance(v, str)]


# --- the pair exists and is targetable --------------------------------------


def test_both_customer_plane_services_are_defined() -> None:
    services = _compose()["services"]
    missing = [n for n in CUSTOMER_PLANE_SERVICES if n not in services]
    assert not missing, f"customer-plane runner services missing: {missing}"


def test_the_label_set_is_the_customer_plane_family_and_not_the_general_pool() -> None:
    """The label is what routes a job here, and it must route nothing else.

    Carrying `omnibase-ci` as well would let any general CI job land on a
    runner with no kubeconfig -- the OMN-18188 failure mode in reverse.
    """
    for name in CUSTOMER_PLANE_SERVICES:
        labels = _service(name)["environment"]["RUNNER_LABELS"]
        assert labels == EXPECTED_LABELS, f"{name} labels are {labels!r}"
        assert "omnibase-ci" not in labels.split(","), (
            f"{name} carries the general-pool label; a general CI job could "
            "land on a runner that holds no lab credential and fail closed."
        )


def test_the_pair_is_not_counted_as_part_of_the_general_fleet() -> None:
    """`omninode-runner-N` is the prefix every fleet-wide assertion filters on
    (tests/ci/test_runner_kubeconfig_mount.py), so a name that collided with it
    would drag these two into the fleet's own kubeconfig-mount requirement.
    """
    for name in CUSTOMER_PLANE_SERVICES:
        assert not name.startswith("omninode-runner-"), name


# --- the absences, each with its positive control ---------------------------


def test_no_probe_forbidden_env_var_is_set_on_the_pair() -> None:
    for name in CUSTOMER_PLANE_SERVICES:
        env = _service(name).get("environment", {})
        present = [v for v in PROBE_FORBIDDEN_ENV if env.get(v) is not None]
        assert not present, (
            f"{name} sets {present}, which the customer-plane probe's own "
            "credential-absence step refuses. The job would fail closed, "
            "correctly."
        )


def test_positive_control_the_general_pool_does_set_kubeconfig() -> None:
    """Without this, the absence above is satisfied by any parsing regression."""
    env = _service(CONTROL_SERVICE).get("environment", {})
    assert env.get("KUBECONFIG"), (
        f"{CONTROL_SERVICE} has no KUBECONFIG -- either OMN-18188's fleet-wide "
        "mount regressed or this control is reading the wrong service, and "
        "every absence assertion in this module is vacuous until it is fixed."
    )


def test_the_pair_mounts_no_lab_credential_directory() -> None:
    for name in CUSTOMER_PLANE_SERVICES:
        offending = [v for v in _volumes(_service(name)) if LAB_CREDENTIALS_TARGET in v]
        assert not offending, (
            f"{name} mounts the lab credentials directory: {offending}. The "
            "file on disk is what makes clearing KUBECONFIG a lie."
        )


def test_positive_control_the_general_pool_does_mount_lab_credentials() -> None:
    mounts = [
        v for v in _volumes(_service(CONTROL_SERVICE)) if LAB_CREDENTIALS_TARGET in v
    ]
    assert len(mounts) == 1, (
        f"{CONTROL_SERVICE} has {len(mounts)} lab-credential mounts, expected 1"
    )


def test_the_pair_mounts_no_docker_socket() -> None:
    """The host daemon is the largest platform capability a runner can hold.

    A customer has no route to the lab's docker socket, and a job that does is
    not measuring a customer's reachability. The runner image tolerates the
    absence: the entrypoint's `_fix_docker_socket_gid` returns early when the
    socket is not a socket, and the disk-admission self-pause logs a skip when
    the docker CLI cannot reach a daemon.
    """
    for name in CUSTOMER_PLANE_SERVICES:
        offending = [v for v in _volumes(_service(name)) if DOCKER_SOCKET in v]
        assert not offending, f"{name} mounts the docker socket: {offending}"


def test_positive_control_the_general_pool_does_mount_the_docker_socket() -> None:
    mounts = [v for v in _volumes(_service(CONTROL_SERVICE)) if DOCKER_SOCKET in v]
    assert mounts, f"{CONTROL_SERVICE} has no docker socket mount; control broken"


def test_the_pair_joins_no_docker_group() -> None:
    """`group_add: 984` exists solely to reach the socket above."""
    for name in CUSTOMER_PLANE_SERVICES:
        assert "group_add" not in _service(name), (
            f"{name} declares group_add with no socket to reach."
        )


# --- registration wiring ----------------------------------------------------


def test_each_has_its_own_credential_volume() -> None:
    """The runner's registration cache key is labels + org URL, so a shared
    volume would let one of these adopt another runner's registration.
    """
    compose = _compose()
    seen = set()
    for name in CUSTOMER_PLANE_SERVICES:
        creds = [v for v in _volumes(_service(name)) if v.endswith("/.runner-creds")]
        assert len(creds) == 1, f"{name} has {creds} credential volumes"
        volume_name = creds[0].split(":")[0]
        assert volume_name not in seen, f"{name} shares a creds volume: {volume_name}"
        seen.add(volume_name)
        assert volume_name in compose["volumes"], (
            f"{volume_name} is mounted but not declared under top-level volumes:"
        )


def test_the_pair_registers_at_org_scope_in_a_group_that_reaches_every_repo() -> None:
    """Group membership grants repository access; the label decides routing.

    `omnibase-ci` is `visibility: all`, so the probe's repository can see these
    runners without a new group or a repository-access edit. An empty group
    would make the registration repository-scoped, which `config.sh` requires
    a repository URL for and which no repository here provides.
    """
    for name in CUSTOMER_PLANE_SERVICES:
        env = _service(name)["environment"]
        assert env["GITHUB_ORG_URL"] == "https://github.com/OmniNode-ai"
        assert env["RUNNER_GROUP"] == "omnibase-ci"
