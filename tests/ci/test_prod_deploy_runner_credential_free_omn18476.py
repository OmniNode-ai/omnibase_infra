# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the credential-isolated production-deploy runner (OMN-18476).

`omninode_infra`'s `deploy-onex-prod.yml::promote-to-prod` is the production
promotion itself. It holds the onex-prod cluster-admin kubeconfig and an ECR
session for the length of a deploy, up to sixty minutes.

OMN-18452 measured the two facts that put it here. It cannot run where it is:
no private-repository job pinned to a GitHub-hosted label has started anywhere
in the organisation since 2026-09-02, and this one has not been dispatched
since 2026-08-30, so nobody has found out. And it must not run on the general
pool: all sixty `omninode-runner-N` containers bind-mount the lab credentials
directory and the host docker socket, so the prod credential would sit beside
lab credentials on a shared, persistent container, reachable through a socket
that is effective root on the host.

Operator ruling 2026-09-16, verbatim "OK" to the recommendation "the dedicated
runner class, not the fleet move". This module is the mechanical half of it.

The shape is deliberately the `omnibase-customer-plane` pair's (OMN-18392),
which exists for the same reason. Constants that must agree with that pair --
the mount targets and the forbidden-variable list -- are IMPORTED from its
module rather than re-spelled, so the two cannot drift into disagreeing about
what "holds nothing" means.

Every assertion of an ABSENCE is paired with a positive control asserting the
same key IS present on a general-pool runner. An empty result is not evidence
of absence: without the control, a renamed service or a broken parse reports
every absence as satisfied.

WHAT THIS DOES NOT CLAIM. The class is a persistent container, not a one-shot
runner. `docker/runners/entrypoint.sh` registers once and then supervises a
long-lived `run.sh`; there is no `--ephemeral` path to use. The isolation here
is about what the container HOLDS, not about how long it lives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.test_customer_plane_runner_credential_free_omn18392 import (
    DOCKER_SOCKET,
    LAB_CREDENTIALS_TARGET,
    PROBE_FORBIDDEN_ENV,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"

SERVICE = "omninode-prod-deploy-runner-1"
CONTROL_SERVICE = "omninode-runner-1"
EXPECTED_LABELS = "self-hosted,omnibase-prod-deploy,linux,x64,arch-amd64"
CLASS_LABEL = "omnibase-prod-deploy"


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


# --- the class exists and routes only what it is meant to -------------------


def test_the_prod_deploy_runner_service_is_defined() -> None:
    assert SERVICE in _compose()["services"]


def test_the_label_set_is_the_prod_deploy_family_and_not_the_general_pool() -> None:
    """The label is what routes the deploy here, and it must route nothing else.

    Carrying `omnibase-ci` as well would let any ordinary CI job land on this
    container. That is not a correctness problem for the CI job, which would
    simply find no kubeconfig -- it is a problem for this one, whose whole
    value is that the set of things that ever execute here is one workflow.
    """
    labels = _service(SERVICE)["environment"]["RUNNER_LABELS"]
    assert labels == EXPECTED_LABELS, f"{SERVICE} labels are {labels!r}"
    assert "omnibase-ci" not in labels.split(","), (
        f"{SERVICE} carries the general-pool label; an ordinary CI job could "
        "then be scheduled on the container that holds the prod credential."
    )
    assert "omnibase-customer-plane" not in labels.split(","), (
        f"{SERVICE} carries the customer-plane label; that class exists to "
        "prove it holds NO credential, and this one holds the largest."
    )


def test_the_class_label_is_not_carried_by_any_other_service() -> None:
    """Positive control on the routing claim above.

    The label is the entire access-control boundary for this container. If a
    second service carried it, a prod deploy could be scheduled somewhere this
    module has never inspected, and every absence asserted below would be true
    of a container the job never lands on.
    """
    carriers = [
        name
        for name, definition in _compose()["services"].items()
        if isinstance(definition, dict)
        and CLASS_LABEL
        in str(definition.get("environment", {}).get("RUNNER_LABELS", "")).split(",")
    ]
    assert carriers == [SERVICE], (
        f"{CLASS_LABEL} is carried by {carriers}, expected exactly [{SERVICE!r}]"
    )


def test_the_service_is_not_counted_as_part_of_the_general_fleet() -> None:
    """`omninode-runner-N` is the prefix every fleet-wide assertion filters on,
    and the one the monitor's remediation targets match. A colliding name would
    drag this container into the fleet's own kubeconfig-mount requirement and
    into auto-bounce.
    """
    assert not SERVICE.startswith("omninode-runner-")


# --- the absences, each with its positive control ---------------------------


def test_the_runner_sets_no_platform_credential_env_var() -> None:
    env = _service(SERVICE).get("environment", {})
    present = [v for v in PROBE_FORBIDDEN_ENV if env.get(v) is not None]
    assert not present, (
        f"{SERVICE} sets {present}. The deploy job establishes its own AWS "
        "session by OIDC and fetches its own kubeconfig over SSM; a "
        "pre-existing credential here is one the job never asked for and "
        "one that outlives it."
    )


def test_positive_control_the_general_pool_does_set_kubeconfig() -> None:
    """Without this, the absence above is satisfied by any parsing regression."""
    env = _service(CONTROL_SERVICE).get("environment", {})
    assert env.get("KUBECONFIG"), (
        f"{CONTROL_SERVICE} has no KUBECONFIG -- either the fleet-wide mount "
        "regressed or this control reads the wrong service, and every absence "
        "assertion in this module is vacuous until it is fixed."
    )


def test_the_runner_mounts_no_lab_credential_directory() -> None:
    offending = [v for v in _volumes(_service(SERVICE)) if LAB_CREDENTIALS_TARGET in v]
    assert not offending, (
        f"{SERVICE} mounts the lab credentials directory: {offending}. This is "
        "the co-residency the operator declined the fleet move to avoid."
    )


def test_positive_control_the_general_pool_does_mount_lab_credentials() -> None:
    mounts = [
        v for v in _volumes(_service(CONTROL_SERVICE)) if LAB_CREDENTIALS_TARGET in v
    ]
    assert len(mounts) == 1, (
        f"{CONTROL_SERVICE} has {len(mounts)} lab-credential mounts, expected 1"
    )


def test_the_runner_mounts_no_docker_socket() -> None:
    """The host daemon is the largest platform capability a runner can hold.

    A container with both the socket and the prod kubeconfig is a single object
    from which the whole lab and production are reachable. The runner image
    tolerates the absence: the entrypoint's docker-GID fixup returns early when
    the socket is not a socket, and the disk-admission self-pause logs a skip
    when the docker CLI reaches no daemon.
    """
    offending = [v for v in _volumes(_service(SERVICE)) if DOCKER_SOCKET in v]
    assert not offending, f"{SERVICE} mounts the docker socket: {offending}"


def test_positive_control_the_general_pool_does_mount_the_docker_socket() -> None:
    mounts = [v for v in _volumes(_service(CONTROL_SERVICE)) if DOCKER_SOCKET in v]
    assert mounts, f"{CONTROL_SERVICE} has no docker socket mount; control broken"


def test_the_runner_joins_no_docker_group() -> None:
    """`group_add` exists solely to reach the socket above."""
    assert "group_add" not in _service(SERVICE), (
        f"{SERVICE} declares group_add with no socket to reach."
    )


def test_positive_control_the_general_pool_does_join_the_docker_group() -> None:
    assert "group_add" in _service(CONTROL_SERVICE), (
        f"{CONTROL_SERVICE} declares no group_add; control broken"
    )


# --- registration wiring ----------------------------------------------------


def test_it_has_its_own_credential_volume_declared_at_top_level() -> None:
    """The registration cache key is labels plus organisation URL, so a shared
    volume would let this container adopt another runner's registration and,
    with it, another runner's label set.
    """
    compose = _compose()
    creds = [v for v in _volumes(_service(SERVICE)) if v.endswith("/.runner-creds")]
    assert len(creds) == 1, f"{SERVICE} has {creds} credential volumes"
    volume_name = creds[0].split(":")[0]
    assert volume_name in compose["volumes"], (
        f"{volume_name} is mounted but not declared under top-level volumes:"
    )
    others = [
        name
        for name, definition in compose["services"].items()
        if name != SERVICE
        and isinstance(definition, dict)
        and any(
            v.split(":")[0] == volume_name
            for v in _volumes(definition)
            if isinstance(v, str)
        )
    ]
    assert not others, f"{volume_name} is shared with {others}"


def test_it_registers_at_org_scope_in_a_group_that_reaches_every_repo() -> None:
    """Group membership grants repository access; the label decides routing."""
    env = _service(SERVICE)["environment"]
    assert env["GITHUB_ORG_URL"] == "https://github.com/OmniNode-ai"
    assert env["RUNNER_GROUP"] == "omnibase-ci"


# --- the image carries trivy, so the job needs no package manager -----------


def test_the_runner_image_installs_trivy() -> None:
    """The one thing the fleet image refused about this job.

    `promote-to-prod` installed Trivy with `sudo apt-get`, `wget`, `sudo gpg`
    and `sudo tee`. The runner image grants the runner user a sudoers entry for
    exactly one `rm -rf` and ships no wget, so that step is the single reason
    the job could not move off a hosted runner. Installing it at build time
    removes the step rather than rewriting it.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "TRIVY_VERSION" in dockerfile, (
        "the runner image declares no TRIVY_VERSION build arg; the deploy job "
        "then has no scanner and no way to install one without a package "
        "manager it does not have"
    )
    assert "trivy --version" in dockerfile, (
        "the Trivy install does not verify the binary executes, so a bad "
        "download would surface as a failed production deploy rather than a "
        "failed image build"
    )


def test_the_trivy_version_is_pinned_in_the_image_lock() -> None:
    """A floating scanner version changes what a production deploy enforces."""
    import json

    lock = json.loads(
        (REPO_ROOT / "docker" / "runners" / "runner-image.lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock.get("trivy_version"), (
        "runner-image.lock.json records no trivy_version; the image identity "
        "would not change when the scanner does"
    )
