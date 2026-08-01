# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""scripts/deploy-gateway.sh -- the OMN-15521 sanctioned .201 gateway deploy path.

Before this script, the `omninode-gateway` compose project (the standalone
operator-edge forwarder at /opt/omninode/gateway) had NO repo-resident deploy
path: it was stood up by hand-copying files into a root-owned directory, ran
an image with an empty `org.opencontainers.image.revision` label, and had no
recorded rollback target. These tests drive the REAL script (scripts/deploy-
gateway.sh) via subprocess -- exactly the convention
tests/scripts/test_deploy_runtime_promotion_class.py already uses for
deploy-runtime.sh -- with `docker`/`sudo` replaced by inspectable fakes so the
full --execute code path runs without a live Docker daemon, systemd, or root.

Coverage maps 1:1 to the OMN-15521 falsifiable acceptance criteria:
  AC1 -- test_help_documents_runbook,
         test_print_compose_cmd_targets_repo_resident_compose_project,
         test_dry_run_default_performs_no_mutation
  AC2 -- test_build_command_stamps_oci_provenance_build_args,
         test_build_command_stamps_sibling_ref_build_args,
         test_execute_deploy_produces_non_empty_image_labels
  AC3 -- test_verify_deployment_red_before_files_absent,
         test_verify_deployment_green_after_files_present
  AC4 -- test_sync_host_files_red_before_stale_copy_diverges,
         test_sync_host_files_green_after_diff_is_empty
  AC6 -- test_execute_deploy_writes_registry_with_rollback_target,
         test_second_deploy_records_previous_digest_as_first_deploys_active,
         test_rollback_target_derived_from_running_container_not_env_file,
         test_previous_image_retagged_for_retention_before_build,
         test_rollback_target_not_recorded_if_previous_image_missing
  (AC5 -- the OMN-12912 restart/redelivery receipt -- is proven separately by
  scripts/gateway_restart_safety_proof.sh +
  tests/scripts/test_gateway_restart_safety_proof.py; the receipt itself lands
  on OMN-12912, not here, per that ticket's own filing instruction.)

Remediation round (OMN-15521, 2026-08-01): a prior version of this script (a)
derived the AC6 rollback target from gateway.env's GATEWAY_IMAGE= line
instead of the container's actual running image, with no existence check --
this produced a registry.json rollback_command that pointed at an already
pruned/dangling image; (b) omitted the OMNIBASE_COMPAT_REF / OMNIMARKET_REF /
ONEX_CHANGE_CONTROL_REF build-args deploy-runtime.sh always passes, silently
falling back to the Dockerfile's hardcoded defaults; (c) verify_deployment()
never compared the running container's actual image against the digest it
just built, so a reload that silently failed to recreate the container still
reported success. The new tests below are RED-before/GREEN-after for each.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-gateway.sh"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "gateway-lane-deploy.md"


# ---------------------------------------------------------------------------
# Fixture: fake docker + sudo + systemctl on PATH, scratch host paths
# ---------------------------------------------------------------------------

_FAKE_DOCKER = """#!/usr/bin/env bash
set -eu
log="${GW_TEST_DOCKER_LOG:?}"
printf '%s\\n' "$*" >> "${log}"

case "$*" in
  "compose -p omninode-gateway -f docker/docker-compose.gateway.yml build"*)
    exit 0
    ;;
  "image inspect "*"--format={{.Id}}")
    printf 'sha256:%064d\\n' "${GW_TEST_DIGEST_SEED:-1}"
    exit 0
    ;;
  "inspect omninode-gateway-forwarder --format {{.Image}}")
    # resolve_running_container_image() / verify_deployment()'s digest
    # readback. Backed by a state file so a `systemctl reload` (fake sudo,
    # below) can simulate the container actually being recreated onto the
    # new digest -- distinguishing "before this deploy" from "after reload".
    state="${GW_TEST_RUNNING_IMAGE_STATE:?}"
    if [ -f "${state}" ]; then
      cat "${state}"
    fi
    exit 0
    ;;
  "tag "*)
    if [ "${GW_TEST_ROLLBACK_TARGET_MISSING:-0}" = "1" ]; then
      exit 1
    fi
    exit 0
    ;;
  *"--format={{index .Config.Labels \\"org.opencontainers.image.revision\\"}}}"*)
    printf '%s\\n' "${GW_TEST_LABEL_REVISION:-}"
    exit 0
    ;;
  *"image.revision"*)
    printf '%s\\n' "${GW_TEST_LABEL_REVISION:-}"
    exit 0
    ;;
  *"build_source"*)
    printf '%s\\n' "${GW_TEST_LABEL_BUILD_SOURCE:-}"
    exit 0
    ;;
  *"exec omninode-gateway-forwarder test -f /app/src/omnibase_infra/nodes/node_bus_forwarder_effect/services/service_gateway_delivery.py")
    [ "${GW_TEST_DELIVERY_PRESENT:-1}" = "1" ]
    exit $?
    ;;
  *"exec omninode-gateway-forwarder test -f /app/src/omnibase_infra/idempotency/store_sqlite.py")
    [ "${GW_TEST_SQLITE_PRESENT:-1}" = "1" ]
    exit $?
    ;;
  *)
    printf 'fake docker: unexpected invocation: %s\\n' "$*" >&2
    exit 1
    ;;
esac
"""

_FAKE_SUDO = """#!/usr/bin/env bash
# Strip `-o <owner> -g <group>` (real root ownership changes are not available
# in a test sandbox) and special-case systemctl so no live unit is required.
set -eu
args=()
skip_next=0
for a in "$@"; do
  if [ "${skip_next}" = "1" ]; then skip_next=0; continue; fi
  case "${a}" in
    -o|-g) skip_next=1; continue ;;
  esac
  args+=("${a}")
done
if [ "${args[0]:-}" = "systemctl" ]; then
  printf '%s\\n' "${args[*]}" >> "${GW_TEST_SYSTEMCTL_LOG:?}"
  if [ "${args[1]:-}" = "reload" ] && [ "${GW_TEST_RELOAD_TAKES_EFFECT:-1}" = "1" ]; then
    # Simulate the reload actually recreating the container onto whatever
    # digest gateway.env holds at this point (update_gateway_env_digest()
    # already ran before reload_service() is called) -- mirrors real compose
    # behavior. GW_TEST_RELOAD_TAKES_EFFECT=0 simulates a reload that exits 0
    # but does not actually recreate the container (the silent-failure case
    # verify_deployment()'s digest check now catches).
    state="${GW_TEST_RUNNING_IMAGE_STATE:-}"
    env_file="${GATEWAY_ENV_FILE:-}"
    if [ -n "${state}" ] && [ -n "${env_file}" ] && [ -f "${env_file}" ]; then
      new_image="$(awk -F= '/^GATEWAY_IMAGE=/{print $2; exit}' "${env_file}")"
      if [ -n "${new_image}" ]; then
        printf '%s' "${new_image}" > "${state}"
      fi
    fi
  fi
  exit 0
fi
exec "${args[@]}"
"""


def _write_fake_bin(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)

    sudo = bin_dir / "sudo"
    sudo.write_text(_FAKE_SUDO, encoding="utf-8")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)


class _Harness:
    """One scratch environment for a subprocess run of deploy-gateway.sh."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.bin_dir = tmp_path / "bin"
        self.host_dir = tmp_path / "opt-omninode-gateway"
        self.env_file = tmp_path / "gateway.env"
        self.registry_dir = tmp_path / "home" / ".omnibase" / "gateway"
        self.docker_log = tmp_path / "docker.log"
        self.systemctl_log = tmp_path / "systemctl.log"
        self.running_image_state = tmp_path / "running-image.state"
        _write_fake_bin(self.bin_dir)
        self.env_file.write_text(
            "GATEWAY_IMAGE=sha256:" + ("0" * 64) + "\n"
            "GATEWAY_AWS_PROFILE=gateway\n"
            "GATEWAY_AWS_CONFIG_FILE=/dev/null\n"
            "GATEWAY_AWS_CERTIFICATE_FILE=/dev/null\n"
            "GATEWAY_AWS_PRIVATE_KEY_FILE=/dev/null\n"
            "GATEWAY_AWS_SIGNING_HELPER_FILE=/dev/null\n"
            "GATEWAY_TPM_DEVICE=/dev/null\n"
            "GATEWAY_TPM_GROUP_ID=105\n"
            "GATEWAY_CONTAINER_UID=1000\n"
            "GATEWAY_CONTAINER_GID=1000\n",
            encoding="utf-8",
        )
        # Simulates a container already running this digest before the
        # deploy under test -- the rollback-target source of truth now that
        # it is read from `docker inspect`, not gateway.env's own line.
        self.running_image_state.write_text("sha256:" + ("0" * 64), encoding="utf-8")

    def env(self, **overrides: str) -> dict[str, str]:
        e = os.environ.copy()
        e["PATH"] = f"{self.bin_dir}{os.pathsep}{e['PATH']}"
        e["HOME"] = str(self.tmp_path / "home")
        e["GATEWAY_HOST_DIR"] = str(self.host_dir)
        e["GATEWAY_ENV_FILE"] = str(self.env_file)
        e["GATEWAY_REGISTRY_DIR"] = str(self.registry_dir)
        e["GW_TEST_DOCKER_LOG"] = str(self.docker_log)
        e["GW_TEST_SYSTEMCTL_LOG"] = str(self.systemctl_log)
        e["GW_TEST_RUNNING_IMAGE_STATE"] = str(self.running_image_state)
        e.update(overrides)
        return e

    def run(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), *args],
            cwd=REPO_ROOT,
            env=self.env(**env_overrides),
            capture_output=True,
            text=True,
            check=False,
        )

    def registry(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(
            (self.registry_dir / "registry.json").read_text(encoding="utf-8")
        )
        return result


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    return _Harness(tmp_path)


# ---------------------------------------------------------------------------
# AC1 -- committed, repo-resident deploy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_script_exists_and_is_executable() -> None:
    assert DEPLOY_SCRIPT.is_file(), "scripts/deploy-gateway.sh must exist (AC1)"
    mode = DEPLOY_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/deploy-gateway.sh must be executable"


@pytest.mark.unit
def test_runbook_exists_alongside_cold_lane_bringup() -> None:
    assert RUNBOOK.is_file(), (
        "docs/runbooks/gateway-lane-deploy.md must exist alongside "
        "docs/runbooks/cold-lane-full-bringup.md (AC1)"
    )


@pytest.mark.unit
def test_help_documents_runbook() -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OMN-15521" in result.stdout


@pytest.mark.unit
def test_print_compose_cmd_targets_repo_resident_compose_project(
    harness: _Harness,
) -> None:
    """AC1: the deploy path must target compose project omninode-gateway from
    the REPO copy of docker-compose.gateway.yml -- never a hand-copied path.
    """
    result = harness.run("--print-compose-cmd")
    assert result.returncode == 0, result.stderr
    assert "-p omninode-gateway" in result.stdout
    assert "-f docker/docker-compose.gateway.yml" in result.stdout
    assert "/opt/omninode/gateway" not in result.stdout


@pytest.mark.unit
def test_dry_run_default_performs_no_mutation(harness: _Harness) -> None:
    """Bare invocation (no --execute) must not touch the host dir, env file,
    or registry -- mirrors deploy-runtime.sh's dry-run-by-default contract.
    """
    before_env = harness.env_file.read_text(encoding="utf-8")
    result = harness.run()
    assert result.returncode == 0, result.stderr
    assert "Dry Run" in result.stdout
    assert not harness.host_dir.exists()
    assert harness.env_file.read_text(encoding="utf-8") == before_env
    assert not (harness.registry_dir / "registry.json").exists()
    assert not harness.docker_log.exists() or harness.docker_log.read_text() == ""


# ---------------------------------------------------------------------------
# AC2 -- OCI provenance labels
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_command_stamps_oci_provenance_build_args(harness: _Harness) -> None:
    """AC2 mechanism: the constructed build invocation must carry the same
    provenance build-args every omnibase-infra runtime container gets. This is
    the root-cause fix for the ticket's own finding (`rev=(empty)
    src=release` on the hand-built container) -- the original hand build never
    passed these flags at all.
    """
    result = harness.run("--print-compose-cmd")
    assert result.returncode == 0, result.stderr
    assert re.search(r"--build-arg VCS_REF=[0-9a-f]{12}", result.stdout)
    assert re.search(r"--build-arg RUNTIME_VERSION=\d+\.\d+\.\d+", result.stdout)
    assert "--build-arg COMPOSE_PROJECT=omninode-gateway" in result.stdout
    assert re.search(r"--build-arg RUNTIME_SOURCE_HASH=[0-9a-f]{12}", result.stdout)
    assert "--build-arg BUILD_SOURCE=release" in result.stdout
    assert "--build-arg PROMOTION_CLASS=clean-main" in result.stdout
    assert "--build-arg NON_MAIN_LINEAGE=false" in result.stdout


@pytest.mark.unit
def test_build_command_stamps_sibling_ref_build_args(harness: _Harness) -> None:
    """AC2 remediation (OMN-15521): scripts/deploy-runtime.sh's build_images()
    passes OMNIBASE_COMPAT_REF / OMNIMARKET_REF / ONEX_CHANGE_CONTROL_REF
    unconditionally on every build -- a prior version of this script silently
    dropped all three, so the gateway image fell back to the Dockerfile's
    hardcoded ARG defaults (OMNIBASE_COMPAT_REF=v0.5.5,
    ONEX_CHANGE_CONTROL_REF=v0.5.3, OMNIMARKET_REF=dev). That is exactly how
    the deployed gateway container's onex-change-control pin (0.5.3) drifted
    from the omnibase-infra runtime container's pin (0.5.1) on the same
    `.201` box. OMNI_HOME is explicitly cleared so the fallback strings are
    deterministic regardless of the host running this test.
    """
    result = harness.run("--print-compose-cmd", OMNI_HOME="")
    assert result.returncode == 0, result.stderr
    assert "--build-arg OMNIBASE_COMPAT_REF=main" in result.stdout
    assert "--build-arg OMNIMARKET_REF=dev" in result.stdout
    assert "--build-arg ONEX_CHANGE_CONTROL_REF=main" in result.stdout


@pytest.mark.unit
def test_current_compose_file_declares_no_provenance_build_args() -> None:
    """RED-before-the-fix control: docker-compose.gateway.yml's OWN declared
    build.args block (what a bare `docker compose build` would use with no
    extra flags -- i.e. exactly how the lane was hand-built on 2026-07-29)
    carries only BUILD_SOURCE/EXPECTED_BUILD_SOURCE. This is why the running
    container had an empty org.opencontainers.image.revision label; the fix
    lives in the deploy script's explicit --build-arg list, not in the
    compose file itself (matching how docker-compose.infra.yml + deploy-
    runtime.sh's build_images() are already split the same way).
    """
    compose_text = (REPO_ROOT / "docker" / "docker-compose.gateway.yml").read_text(
        encoding="utf-8"
    )
    build_block_match = re.search(
        r"build:\n(.*?)\n    container_name:", compose_text, re.DOTALL
    )
    assert build_block_match is not None
    build_block = build_block_match.group(1)
    assert "VCS_REF" not in build_block
    assert "COMPOSE_PROJECT" not in build_block


@pytest.mark.unit
def test_execute_deploy_produces_non_empty_image_labels(harness: _Harness) -> None:
    """GREEN-after (mechanism level): after --execute, verify_deployment()
    reads back a non-empty org.opencontainers.image.revision from the fake
    docker inspect -- the exact probe cited in the ticket's AC2.
    """
    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "AC2 OK: org.opencontainers.image.revision=3541ac805b86" in result.stdout


# ---------------------------------------------------------------------------
# AC3 -- the two OMN-12912 files present in the running container
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_deployment_red_before_files_absent(harness: _Harness) -> None:
    """RED: the ticket's own §2 probe state (pre-#2556 image) -- both files
    absent -- must be surfaced as a hard failure, not silently accepted.
    """
    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="deadbeef0000",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="0",
        GW_TEST_SQLITE_PRESENT="0",
    )
    assert result.returncode != 0
    assert "AC3 FAILED" in result.stdout + result.stderr


@pytest.mark.unit
def test_verify_deployment_green_after_files_present(harness: _Harness) -> None:
    """GREEN: once the deployed image carries #2556, both files resolve and
    the deploy reports success -- the exact AC3 probe flipping.
    """
    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "AC3 OK" in result.stdout


@pytest.mark.unit
def test_verify_deployment_red_before_reload_silently_fails(harness: _Harness) -> None:
    """RED (OMN-15521 remediation, exists-but-wrong): a `systemctl reload`
    that exits 0 without actually recreating the container must NOT be
    reported as a successful deploy. A prior version of verify_deployment()
    only checked image labels and file presence -- both of which the STALE
    (still-running, pre-deploy) container also satisfies once it has already
    been deployed once -- so a silently-failed recreate on any deploy after
    the first read as success on every subsequent run.
    """
    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_RELOAD_TAKES_EFFECT="0",
    )
    assert result.returncode != 0
    assert "AC-VERIFY FAILED" in result.stdout + result.stderr
    assert "did not take effect" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# AC4 -- host compose file matches the merged-dev repo copy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sync_host_files_red_before_stale_copy_diverges(harness: _Harness) -> None:
    """RED (exists-but-wrong, not merely absent): pre-populate the host dir
    with a stale compose file -- exactly the 2026-07-29 hand-copy scenario,
    where /opt/omninode/gateway/docker-compose.gateway.yml lacked #2556's
    gateway-delivery-state volume. Before any sync, diff is non-empty.
    """
    harness.host_dir.mkdir(parents=True)
    (harness.host_dir / "docker-compose.gateway.yml").write_text(
        "services: {}\n# stale pre-#2556 copy\n", encoding="utf-8"
    )
    real_compose = (REPO_ROOT / "docker" / "docker-compose.gateway.yml").read_text(
        encoding="utf-8"
    )
    assert (
        harness.host_dir / "docker-compose.gateway.yml"
    ).read_text() != real_compose, "fixture must actually diverge from the repo copy"


@pytest.mark.unit
def test_sync_host_files_green_after_diff_is_empty(harness: _Harness) -> None:
    """GREEN: after --execute, the host copy is byte-identical to the repo
    copy that produced the running container -- the exact AC4 probe.
    """
    harness.host_dir.mkdir(parents=True)
    (harness.host_dir / "docker-compose.gateway.yml").write_text(
        "services: {}\n# stale pre-#2556 copy\n", encoding="utf-8"
    )

    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "AC4 OK" in result.stdout

    repo_compose = (REPO_ROOT / "docker" / "docker-compose.gateway.yml").read_text(
        encoding="utf-8"
    )
    host_compose = (harness.host_dir / "docker-compose.gateway.yml").read_text(
        encoding="utf-8"
    )
    assert host_compose == repo_compose

    repo_canary = (
        REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"
    ).read_text(encoding="utf-8")
    host_canary = (harness.host_dir / "gateway" / "beta-gateway-canary.yaml").read_text(
        encoding="utf-8"
    )
    assert host_canary == repo_canary


# ---------------------------------------------------------------------------
# AC6 -- rollback target recorded via registry.json
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_deploy_writes_registry_with_rollback_target(
    harness: _Harness,
) -> None:
    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_DIGEST_SEED="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout

    registry = harness.registry()
    assert registry["compose_project"] == "omninode-gateway"
    assert registry["active_digest"] == "sha256:" + ("0" * 63) + "1"
    assert registry["previous_digest"] == "sha256:" + ("0" * 64)
    assert registry["git_sha"]
    assert registry["rollback_command"]
    rollback_command = str(registry["rollback_command"])
    assert "GATEWAY_IMAGE=sha256:" + ("0" * 64) in rollback_command

    # A reload must actually have been requested.
    assert harness.systemctl_log.exists()
    assert (
        "systemctl reload onex-gateway-forwarder" in harness.systemctl_log.read_text()
    )


@pytest.mark.unit
def test_second_deploy_records_previous_digest_as_first_deploys_active(
    harness: _Harness,
) -> None:
    """The rollback target must chain: deploy #2's `previous_digest` must
    equal deploy #1's `active_digest`, not the pre-existing gateway.env value
    from before either deploy ran.
    """
    first = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="aaaaaaaaaaaa",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_DIGEST_SEED="1",
    )
    assert first.returncode == 0, first.stderr + first.stdout
    first_registry = harness.registry()

    second = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="bbbbbbbbbbbb",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_DIGEST_SEED="2",
    )
    assert second.returncode == 0, second.stderr + second.stdout
    second_registry = harness.registry()

    assert second_registry["previous_digest"] == first_registry["active_digest"]
    assert second_registry["active_digest"] != first_registry["active_digest"]


@pytest.mark.unit
def test_rollback_target_derived_from_running_container_not_env_file(
    harness: _Harness,
) -> None:
    """RED-before-the-fix control / GREEN-after (OMN-15521 remediation): a
    prior version of this script awk'd the rollback target out of
    gateway.env's GATEWAY_IMAGE= line. That line can drift from what the
    container is actually running (a previous deploy that wrote the file but
    was killed before reload; a manual edit) -- exactly what this fixture
    reproduces: gateway.env claims one digest, the running container (the
    fake docker inspect state file) reports a different one. The recorded
    previous_digest must be the ACTUAL running digest, never the stale
    env-file value.
    """
    stale_env_digest = "sha256:" + ("e" * 64)
    harness.env_file.write_text(
        harness.env_file.read_text(encoding="utf-8").replace(
            "GATEWAY_IMAGE=sha256:" + ("0" * 64),
            f"GATEWAY_IMAGE={stale_env_digest}",
        ),
        encoding="utf-8",
    )
    actually_running_digest = "sha256:" + ("a" * 64)
    harness.running_image_state.write_text(actually_running_digest, encoding="utf-8")

    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_DIGEST_SEED="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout

    registry = harness.registry()
    assert registry["previous_digest"] == actually_running_digest
    assert registry["previous_digest"] != stale_env_digest


@pytest.mark.unit
def test_previous_image_retagged_for_retention_before_build(
    harness: _Harness,
) -> None:
    """OMN-15521 remediation: the previous running image must be retagged
    under a durable name (`docker tag <previous_digest>
    docker-gateway-forwarder:previous`) BEFORE the build moves
    BUILD_IMAGE_TAG onto the new image -- otherwise the old image becomes
    untagged/dangling the instant the build succeeds and is eligible for
    collection by a routine `docker image prune` before anyone needs it for
    rollback. Order matters, not just occurrence.
    """
    previous_digest = "sha256:" + ("a" * 64)
    harness.running_image_state.write_text(previous_digest, encoding="utf-8")

    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout

    lines = harness.docker_log.read_text(encoding="utf-8").splitlines()
    tag_idx = next((i for i, line in enumerate(lines) if line.startswith("tag ")), None)
    build_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("compose -p omninode-gateway")
        ),
        None,
    )
    assert tag_idx is not None, f"expected a `docker tag ...` call, got: {lines}"
    assert build_idx is not None, (
        f"expected a `docker compose ... build` call, got: {lines}"
    )
    assert f"tag {previous_digest} docker-gateway-forwarder:previous" in lines[tag_idx]
    assert tag_idx < build_idx, (
        "retention tag must be applied BEFORE the build moves "
        "BUILD_IMAGE_TAG off the previous image"
    )


@pytest.mark.unit
def test_rollback_target_not_recorded_if_previous_image_missing(
    harness: _Harness,
) -> None:
    """Fail-closed (OMN-15521 remediation): if the previous running image no
    longer resolves locally (already pruned), the script must not record it
    as a rollback target -- recording an unvalidated digest verbatim was
    exactly how the previous version produced a registry.json
    rollback_command pointing at an image `docker image inspect` could not
    find (confirmed live on `.201`: registry previous_digest
    sha256:b51b380d... resolved to "No such image").
    """
    previous_digest = "sha256:" + ("a" * 64)
    harness.running_image_state.write_text(previous_digest, encoding="utf-8")

    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
        GW_TEST_ROLLBACK_TARGET_MISSING="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout

    registry = harness.registry()
    assert registry["previous_digest"] is None
    assert registry["rollback_command"] is None, (
        "OMN-15521 remediation round 3: a null previous_digest must produce a "
        "null rollback_command, never a sed command built from an empty "
        "digest -- confirmed live on .201: a prior version emitted "
        '"GATEWAY_IMAGE=" (nothing after the =) here, which would have '
        "corrupted gateway.env's GATEWAY_IMAGE= line to an unparseable value "
        "on the next accidental run, wedging the systemd unit's "
        "ExecStartPre digest-format assertion on the following restart."
    )


@pytest.mark.unit
def test_first_deploy_records_no_rollback_target(harness: _Harness) -> None:
    """First-ever deploy: no container is running yet (the fake docker
    inspect state file is absent), so there is nothing to roll back to --
    registry.json must record previous_digest as null, not a fabricated or
    stale value.
    """
    harness.running_image_state.unlink()

    result = harness.run(
        "--execute",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout

    registry = harness.registry()
    assert registry["previous_digest"] is None
    assert registry["rollback_command"] is None, (
        "first deploy has no rollback target -- rollback_command must be "
        "null, not a sed command with an empty GATEWAY_IMAGE= value"
    )
    docker_log = harness.docker_log.read_text(encoding="utf-8")
    assert not any(line.startswith("tag ") for line in docker_log.splitlines()), (
        "must not attempt to retag an empty previous digest"
    )


@pytest.mark.unit
def test_skip_reload_leaves_container_on_previous_digest(harness: _Harness) -> None:
    """--skip-reload must still write gateway.env + registry but must NOT
    invoke systemctl -- a deliberate escape hatch, not a silent no-op.
    """
    result = harness.run(
        "--execute",
        "--skip-reload",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="release",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (
        "gateway.env is updated but the running container still has the OLD digest"
        in (result.stdout + result.stderr)
    )
    assert not harness.systemctl_log.exists() or harness.systemctl_log.read_text() == ""


@pytest.mark.unit
def test_execute_without_env_file_fails_closed(harness: _Harness) -> None:
    """AC1 fail-closed case: --execute must refuse (not silently proceed)
    when GATEWAY_ENV_FILE is missing, rather than building/deploying with
    unresolved compose interpolation variables.
    """
    harness.env_file.unlink()
    result = harness.run("--execute")
    assert result.returncode != 0
    assert "GATEWAY_ENV_FILE not found" in result.stderr
    assert not (harness.registry_dir / "registry.json").exists()


# ---------------------------------------------------------------------------
# BUILD_SOURCE=workspace staging (OMN-15521 remediation round 3)
#
# A prior version of this script honoured BUILD_SOURCE=workspace for the
# stamped labels (promotion_class/non_main_lineage) but never actually staged
# workspace/sibling-repos/ -- unlike scripts/deploy-runtime.sh's
# build_images(), which always calls stage_workspace_if_needed() first.
# docker/Dockerfile.runtime unconditionally COPYs workspace/sibling-repos/,
# so a workspace-mode build silently used the committed placeholder (or
# whatever stale staging happened to already be sitting in the checkout)
# while still stamping workspace-provenance labels the prod-promotion gate
# and lineage guard consume. A full live staging run needs real OMNI_HOME
# sibling git clones, so -- matching the established convention
# tests/scripts/test_deploy_runtime_build_context.py already uses for the
# identical deploy-runtime.sh wiring (test_deploy_runtime_stages_workspace_
# and_passes_omni_home_arg / test_deploy_runtime_runs_sibling_lock_pin_
# preflight) -- these are static assertions on the wiring, not a live
# staging run.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_deploy_gateway_stages_workspace_before_build() -> None:
    """AC2 remediation: BUILD_SOURCE=workspace must stage sibling repos
    before build_image() runs, or the build silently uses stale/placeholder
    workspace/sibling-repos/ content while still claiming workspace
    provenance in its stamped labels.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'stage_workspace_if_needed "${repo_root}"' in deploy_script
    stage_call_idx = deploy_script.index('stage_workspace_if_needed "${repo_root}"\n')
    build_call_idx = deploy_script.index('build_image "${repo_root}" "${git_sha}"')
    assert stage_call_idx < build_call_idx, (
        "stage_workspace_if_needed must run BEFORE build_image in main(), or "
        "the build reads workspace/sibling-repos/ before it is populated"
    )

    # The staging function itself must invoke the SAME script deploy-runtime.sh
    # uses -- reused machinery, not a parallel reimplementation.
    assert (
        'stage_script="${repo_root}/scripts/runtime_build/stage_workspace.sh"'
        in deploy_script
    )
    assert 'bash "${stage_script}"' in deploy_script

    # Release mode (the default) must not attempt to stage anything.
    assert (
        'if [[ "${build_source}" != "workspace" ]]; then\n        return 0'
        in (deploy_script.split("stage_workspace_if_needed() {", 1)[1])
    )


@pytest.mark.unit
def test_deploy_gateway_requires_omni_home_for_workspace_build_source(
    harness: _Harness,
) -> None:
    """BUILD_SOURCE=workspace with no OMNI_HOME must fail closed before any
    build/mutation -- mirrors deploy-runtime.sh's validate_build_source_config.
    """
    result = harness.run(
        "--execute",
        BUILD_SOURCE="workspace",
        OMNI_HOME="",
        GW_TEST_LABEL_REVISION="3541ac805b86",
        GW_TEST_LABEL_BUILD_SOURCE="workspace",
        GW_TEST_DELIVERY_PRESENT="1",
        GW_TEST_SQLITE_PRESENT="1",
    )
    assert result.returncode != 0
    assert "BUILD_SOURCE=workspace requires OMNI_HOME" in (
        result.stdout + result.stderr
    )
    assert not harness.docker_log.exists() or "compose -p" not in (
        harness.docker_log.read_text(encoding="utf-8")
    ), "must fail before attempting a build, not mid-build"


@pytest.mark.unit
def test_deploy_gateway_runs_sibling_lock_pin_preflight_in_workspace_mode() -> None:
    """Workspace staging must run the OMN-12987 lock-pin preflight before
    build, same as deploy-runtime.sh -- the recurrence guard for a stale
    vendored sibling silently shipping (the 2026-06-11 stability crash).
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'check_sibling_lock_pins "${repo_root}" "${omni_home}"' in deploy_script
    assert "scripts/runtime_build/check_sibling_lock_pins.py" in deploy_script
    assert "Refusing to build a stale image." in deploy_script
    # Current CLI (OMN-12977/12987): --lock / repeatable --repo / --output,
    # never the removed --provenance-out flag.
    assert "--lock" in deploy_script
    assert "--output" in deploy_script
    assert "--provenance-out" not in deploy_script
