# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the lab k3s kubeconfig wiring for the runner fleet (OMN-18188).

The runner containers have kubectl but, before this ticket, no kubeconfig --
`kubectl` silently fell back to `localhost:8080` and every lab-cluster workflow
step failed closed at preflight, with no signal pointing at the missing mount.

Coverage:
- Every one of the 88 `omninode-runner-N` fleet services (config-resolved, not
  regexed) carries the read-only lab kubeconfig bind mount and the matching
  `KUBECONFIG` env var pointing at the SAME in-container path.
- `omninode-deploy-runner` -- a separate, standalone service definition that
  does not use `&runner-base`/`&runner-env` -- is deliberately NOT in scope
  and is asserted to have neither.
- The mounted kubeconfig is never rsynced from this repo: it must not appear
  in deploy-runners.sh's SYNC_PATHS, because it is a host-generated secret
  file, not a build artifact -- an rsync entry would let a repo-side deploy
  silently overwrite (or attempt to create) a file that must originate solely
  on the runner host.

Note on the compose file's own structure (relevant to why this needed 88
edits, not one): `volumes:` is a YAML *list*, and YAML merge keys (`<<:`) only
merge *mappings* -- a service's own `volumes:` key entirely replaces, rather
than extends, `&runner-base`'s. The OMN-16363 disk-admission mount already
established the precedent of duplicating a fleet-wide volume into every
service's own `volumes:` list; this ticket's kubeconfig mount follows the
same shape. `environment:`, by contrast, IS a mapping, so `KUBECONFIG` set
once on `&runner-env` correctly reaches every service through the nested
`!!merge <<: *runner-env` in each service's `environment:` block -- verified
below via the real resolved config, not asserted by construction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"

EXPECTED_MOUNT_TARGET = "/home/runner/.kube/lab-ci-reader.kubeconfig"
EXPECTED_MOUNT_SOURCE_EXPR = (
    "${RUNNER_LAB_KUBECONFIG_HOST_PATH:-"
    "/home/jonah/.omnibase/runners/kubeconfig/lab-ci-reader.kubeconfig}"
)
EXPECTED_KUBECONFIG_ENV = "/home/runner/.kube/lab-ci-reader.kubeconfig"
FLEET_SERVICE_COUNT = 88


def _load_compose() -> dict[str, Any]:
    """Parse the compose file with real YAML semantics (anchors + merge keys),
    never regex -- a regex over this file cannot tell a live, effective
    per-service value from a comment or the (functionally inert) anchor copy.
    """
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _fleet_services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    services = compose["services"]
    fleet = {
        name: svc
        for name, svc in services.items()
        if name.startswith("omninode-runner-")
    }
    assert len(fleet) == FLEET_SERVICE_COUNT, (
        f"expected {FLEET_SERVICE_COUNT} omninode-runner-N services, found "
        f"{len(fleet)}: {sorted(fleet)}"
    )
    return fleet


def _volume_entries(service: dict[str, Any]) -> list[str]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    return [v for v in volumes if isinstance(v, str)]


# --- every fleet service resolves the mount + env ---------------------------


def test_every_fleet_service_has_the_kubeconfig_env_var() -> None:
    compose = _load_compose()
    fleet = _fleet_services(compose)
    missing = [
        name
        for name, svc in fleet.items()
        if svc.get("environment", {}).get("KUBECONFIG") != EXPECTED_KUBECONFIG_ENV
    ]
    assert not missing, (
        f"services missing KUBECONFIG={EXPECTED_KUBECONFIG_ENV!r}: {missing}"
    )


def test_every_fleet_service_has_the_readonly_kubeconfig_mount() -> None:
    compose = _load_compose()
    fleet = _fleet_services(compose)
    missing = []
    for name, svc in fleet.items():
        entries = _volume_entries(svc)
        matches = [e for e in entries if EXPECTED_MOUNT_TARGET in e]
        if len(matches) != 1:
            missing.append((name, matches))
    assert not missing, (
        f"services without exactly one kubeconfig mount entry: {missing}"
    )


def test_the_mount_is_read_only() -> None:
    """`:ro` is load-bearing here -- no runner job may mutate the lab
    cluster's credential file.
    """
    compose = _load_compose()
    fleet = _fleet_services(compose)
    not_ro = []
    for name, svc in fleet.items():
        entries = _volume_entries(svc)
        match = next(e for e in entries if EXPECTED_MOUNT_TARGET in e)
        if not match.endswith(":ro"):
            not_ro.append((name, match))
    assert not not_ro, f"kubeconfig mount is not read-only for: {not_ro}"


def test_the_mount_source_uses_the_documented_default() -> None:
    """The host path is override-able via RUNNER_LAB_KUBECONFIG_HOST_PATH with
    a `:-` (soft) default, not `:?` (fail-fast) -- confirmed against
    docker-compose.runners.yml's own "CONSOLIDATED REQUIRED ENV VARS" comment
    in deploy-runners.sh, which enumerates only the `:?`-guarded vars. A
    default means no new var needs registering there.
    """
    compose = _load_compose()
    fleet = _fleet_services(compose)
    for name, svc in fleet.items():
        entries = _volume_entries(svc)
        match = next(e for e in entries if EXPECTED_MOUNT_TARGET in e)
        assert match.startswith(EXPECTED_MOUNT_SOURCE_EXPR + ":"), (
            f"{name}: unexpected mount source expression: {match!r}"
        )


def test_the_required_env_vars_comment_does_not_need_a_new_entry() -> None:
    """Positive control for the claim above: the three vars actually listed
    under CONSOLIDATED REQUIRED ENV VARS are all `:?`-guarded in compose, and
    RUNNER_LAB_KUBECONFIG_HOST_PATH (a `:-` default) is correctly absent --
    proving the comment block's own criterion (fail-fast guard) rather than
    just asserting the new var's name is missing, which a checker with no
    real criterion would also satisfy.
    """
    script_text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
    start = script_text.index("CONSOLIDATED REQUIRED ENV VARS")
    end = script_text.index("set -euo pipefail")
    block = script_text[start:end]
    for required_var in (
        "DEPLOY_RUNNER_OMNI_HOME",
        "DEPLOY_RUNNER_OPERATOR_ENV_FILE",
    ):
        assert required_var in block
        assert f"{required_var}:?" in compose_text
    assert "RUNNER_LAB_KUBECONFIG_HOST_PATH" not in block
    assert "RUNNER_LAB_KUBECONFIG_HOST_PATH:?" not in compose_text


# --- omninode-deploy-runner is deliberately out of scope --------------------


def test_the_deploy_runner_service_is_not_part_of_the_fleet_count() -> None:
    compose = _load_compose()
    assert "omninode-deploy-runner" in compose["services"]
    assert "omninode-deploy-runner" not in _fleet_services(compose)


def test_the_deploy_runner_service_gets_neither_the_mount_nor_the_env() -> None:
    """omninode-deploy-runner does not use `&runner-base`/`&runner-env` -- it
    is a standalone service definition with its own OMNI_HOME clone mount and
    a narrower purpose. OMN-18188 is scoped to the shared fleet only.
    """
    compose = _load_compose()
    svc = compose["services"]["omninode-deploy-runner"]
    assert svc.get("environment", {}).get("KUBECONFIG") is None
    entries = _volume_entries(svc)
    assert not any(EXPECTED_MOUNT_TARGET in e for e in entries)


# --- the host-generated secret is never rsynced -----------------------------


def _parse_sync_paths() -> list[str]:
    spec_path = SCRIPTS_CI / "check_runner_host_artifact_freshness.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_runner_host_artifact_freshness", spec_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module.parse_sync_paths(DEPLOY_SCRIPT.read_text(encoding="utf-8"))


def test_the_kubeconfig_is_not_in_sync_paths() -> None:
    """A host-generated secret must never be rsynced from this repo -- an
    entry here would mean a repo-side deploy could silently overwrite the
    runner host's real kubeconfig with whatever (nonexistent) file lives at
    that path in the repo checkout.
    """
    sync_paths = _parse_sync_paths()
    offending = [
        p for p in sync_paths if "kubeconfig" in p.lower() or "kube" in p.lower()
    ]
    assert not offending, f"SYNC_PATHS must not reference the kubeconfig: {offending}"


def test_positive_control_sync_paths_parser_actually_finds_entries() -> None:
    """Zero rows is not evidence of absence: prove the parser used above
    returns real, known entries before trusting the "not present" result.
    """
    sync_paths = _parse_sync_paths()
    assert "docker/docker-compose.runners.yml" in sync_paths
    assert "docker/runners/entrypoint.sh" in sync_paths


# --- the compose file still parses cleanly end to end -----------------------


def test_docker_compose_config_resolves_without_error_for_the_fleet(
    tmp_path: Path,
) -> None:
    """Full-fidelity check via `docker compose config`, not just PyYAML: this
    is the same interpolation/merge engine that runs `up -d` on the host, and
    it proves RUNNER_LAB_KUBECONFIG_HOST_PATH's soft default does not turn
    into a fail-fast interpolation error for anyone running a plain
    `docker compose -f docker-compose.runners.yml config` with none of the
    optional vars exported.
    """
    import shutil

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available in this environment")
    fake_omni_home = tmp_path / "fake-omni-home"
    fake_operator_env = tmp_path / "fake-operator.env"
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "RUNNER_TOKEN": "dummy",
            "DEPLOY_RUNNER_OMNI_HOME": str(fake_omni_home),
            "DEPLOY_RUNNER_OPERATOR_ENV_FILE": str(fake_operator_env),
        },
    )
    assert result.returncode == 0, (
        f"docker compose config failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
