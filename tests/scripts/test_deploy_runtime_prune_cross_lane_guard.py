# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""prune_old_deployments() must not delete a bundle another lane still needs (OMN-19910).

Defect: ``prune_old_deployments()`` (``scripts/deploy-runtime.sh``) resolved
"is this deployment still needed" from exactly one source -- the invoking
lane's own ``REGISTRY_FILE`` (``registry.<compose_project>.json``). Every
other lane (dev, stability-test, prepr-1, prepr-2, lakshman, ...) writes its
own registry file under the SAME ``DEPLOY_ROOT``/``deployed/`` tree, and the
judge lane writes no registry file at all. A routine redeploy on one lane
would therefore prune a ``deployed/<version>/`` directory a DIFFERENT lane's
live containers still bind-mounted. Docker then recreates the missing
bind-mount source as an empty root-owned directory on the next container
restart, and the affected lane's services crash-loop or serve out of an
empty tree.

Observed live on ``.201`` at the 2026-09-28 ~12:12Z reboot (OMN-17427): a
stability-test redeploy had pruned bundle ``0.38.13`` -- still bind-mounted
by that lane's own long-lived ``keycloak`` container -- and the judge lane's
bundle ``0.38.4``, which had no registry file to protect it at all and whose
containers still mounted it.

Fix, two layers, both exercised here:
  1. Union every lane's registered ``active_path`` (``registry.*.json``)
     before pruning, not just this invocation's own ``REGISTRY_FILE``.
  2. Before any delete, ask docker directly whether a live container still
     bind-mounts the directory (``containers_bound_to_deploy_dir``,
     OMN-17287) -- the only check that also protects a lane with NO registry
     file, and the only one that protects a version a lane's OWN registry has
     already moved past while its containers have not yet been recreated
     against the new one (the stability-test keycloak case above: registry
     said ``0.38.57``, the live container still mounted ``0.38.13``).

These tests drive the ACTUAL script seam per ``feedback_test_the_artifact_that_
runs``: ``prune_old_deployments()`` and ``containers_bound_to_deploy_dir()``
are extracted VERBATIM from ``deploy-runtime.sh`` and executed under bash,
with only the true I/O boundary (``docker``) replaced by a file-backed fake.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


_LOG_FUNCS = """
log_step() { printf 'STEP: %s\\n' "$*" >&2; }
log_info() { printf 'INFO: %s\\n' "$*" >&2; }
log_warn() { printf 'WARN: %s\\n' "$*" >&2; }
log_error() { printf 'ERR: %s\\n' "$*" >&2; }
log_cmd() { printf 'CMD: %s\\n' "$*" >&2; }
"""


def _write_docker_stub(bin_dir: Path) -> None:
    """Fake `docker` answering `ps --quiet` and `inspect --format` from a
    fixture file: one line per running container, `<name>\\t<src1>,<src2>`.

    Copied verbatim (same fixture shape) from
    test_deploy_runtime_live_mount_cleanup_guard.py so both suites drive
    containers_bound_to_deploy_dir() identically.
    """
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${DOCKER_STUB_DIR}/calls.log"\n'
        'fixture="${DOCKER_STUB_DIR}/containers.tsv"\n'
        '[[ -f "${fixture}" ]] || exit 0\n'
        "\n"
        'if [[ "${1:-}" == "ps" ]]; then\n'
        '    cut -f1 "${fixture}"\n'
        "    exit 0\n"
        "fi\n"
        "\n"
        'if [[ "${1:-}" == "inspect" ]]; then\n'
        '    fmt=""; target=""\n'
        "    shift\n"
        "    while [[ $# -gt 0 ]]; do\n"
        '        case "$1" in\n'
        '            --format|-f) fmt="$2"; shift 2 ;;\n'
        '            --format=*) fmt="${1#--format=}"; shift ;;\n'
        '            *) target="$1"; shift ;;\n'
        "        esac\n"
        "    done\n"
        '    line="$(awk -F"\\t" -v t="${target}" \'$1==t{print;exit}\' "${fixture}" 2>/dev/null || true)"\n'
        '    [[ -n "${line}" ]] || exit 1\n'
        '    if [[ "${fmt}" == *".Name"* ]]; then\n'
        '        printf "/%s\\n" "$(printf "%s" "${line}" | cut -f1)"\n'
        "        exit 0\n"
        "    fi\n"
        '    printf "%s" "${line}" | cut -f2 | tr "," "\\n"\n'
        "    exit 0\n"
        "fi\n"
        "\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_prune(
    tmp_path: Path,
    *,
    versions: list[str],
    own_active_version: str | None,
    other_registries: dict[str, str] | None = None,
    containers: dict[str, list[str]] | None = None,
    max_deployments: int = 1,
    no_own_registry: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Execute the REAL prune_old_deployments() against a fixture DEPLOY_ROOT.

    ``versions`` are created oldest-first (index 0 is oldest, last is
    newest) so retention-count ordering is deterministic across filesystems.
    ``other_registries`` maps compose-project suffix -> active version name,
    each written to its own ``registry.<suffix>.json``.
    """
    deploy_root = tmp_path / "deploy_root"
    deployed_root = deploy_root / "deployed"
    deployed_root.mkdir(parents=True)

    base_time = time.time() - 1000
    for offset, version in enumerate(versions):
        version_dir = deployed_root / version
        (version_dir / "contracts").mkdir(parents=True)
        (version_dir / "contracts" / "marker.txt").write_text(version, encoding="utf-8")
        mtime = base_time + offset
        os.utime(version_dir, (mtime, mtime))

    own_registry_file = deploy_root / "registry.omnibase-infra.json"
    if not no_own_registry and own_active_version is not None:
        own_registry_file.write_text(
            '{"deploy_path": "%s"}\n' % (deployed_root / own_active_version),
            encoding="utf-8",
        )

    for suffix, active_version in (other_registries or {}).items():
        (deploy_root / f"registry.{suffix}.json").write_text(
            '{"deploy_path": "%s"}\n' % (deployed_root / active_version),
            encoding="utf-8",
        )

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)

    # Resolve container fixture entries (version name -> path) into absolute
    # bind-mount source paths under deployed_root.
    resolved_containers: dict[str, list[str]] = {}
    for name, mount_versions in (containers or {}).items():
        resolved_containers[name] = [
            str(deployed_root / v / "contracts") for v in mount_versions
        ]
    (stub_dir / "containers.tsv").write_text(
        "".join(
            f"{name}\t{','.join(srcs)}\n" for name, srcs in resolved_containers.items()
        ),
        encoding="utf-8",
    )

    globals_prelude = "\n".join(
        [
            "set -uo pipefail",
            f'DEPLOY_ROOT="{deploy_root}"',
            f'REGISTRY_FILE="{own_registry_file}"',
            f"MAX_DEPLOYMENTS={max_deployments}",
        ]
    )

    parts = [
        globals_prelude,
        _LOG_FUNCS,
        _extract_function("containers_bound_to_deploy_dir"),
        _extract_function("prune_old_deployments"),
        "prune_old_deployments",
    ]
    harness = tmp_path / "harness.sh"
    harness.write_text("\n".join(parts), encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)

    result = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )
    return result, deployed_root


@pytest.mark.unit
def test_prune_protects_version_active_in_another_lanes_registry(
    tmp_path: Path,
) -> None:
    """AC1: union every lane's registry.*.json before pruning.

    Six versions, retention limit 1 (keep only the newest by mtime). The
    OLDEST version is the active_path of a DIFFERENT lane's registry
    (stability-test) -- not this invocation's own REGISTRY_FILE -- and must
    survive purely because some lane's registry names it active.
    """
    versions = [f"0.3{i}.0" for i in range(6)]
    result, deployed_root = _run_prune(
        tmp_path,
        versions=versions,
        own_active_version=versions[-1],
        other_registries={"omnibase-infra-stability-test": versions[0]},
        containers={},
        max_deployments=1,
    )

    assert (deployed_root / versions[0]).exists(), (
        "prune_old_deployments() deleted a version registered active by a "
        "DIFFERENT lane's registry.*.json -- OMN-19910: only unioning every "
        "lane's registry, not just this invocation's own REGISTRY_FILE, "
        "protects it.\nstdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )


@pytest.mark.unit
def test_prune_protects_version_still_live_mounted_despite_stale_registry(
    tmp_path: Path,
) -> None:
    """AC2: cross-check live docker mounts before delete, even when NO
    registry (own or any other lane's) names the version active any more.

    Mirrors the live stability-test keycloak incident: that lane's own
    registry had already moved on to a newer version, but its keycloak
    container was still bind-mounted to the older one. Registry union alone
    (AC1) would NOT catch this -- only the live mount check does.
    """
    versions = [f"0.4{i}.0" for i in range(6)]
    stale_but_mounted = versions[1]  # old enough to be past the retention window
    result, deployed_root = _run_prune(
        tmp_path,
        versions=versions,
        own_active_version=versions[-1],
        other_registries={
            # This lane's registry has already moved past stale_but_mounted --
            # registry union does NOT list it as anyone's active_path.
            "omnibase-infra-stability-test": versions[-2],
        },
        containers={"omnibase-infra-stability-test-keycloak": [stale_but_mounted]},
        max_deployments=1,
    )

    assert (deployed_root / stale_but_mounted).exists(), (
        "prune_old_deployments() deleted a version no registry named active "
        "but a live container still bind-mounted -- the exact stability-test "
        "keycloak incident (OMN-19910/OMN-17427): registry union is not "
        "enough, the mount check must run regardless of registry state.\n"
        "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )
    assert stale_but_mounted in result.stderr, (
        "the skip must name the bind-mounted version so an operator does not "
        "have to reconstruct it from docker forensics.\nstderr:\n" + result.stderr
    )


@pytest.mark.unit
def test_prune_protects_no_registry_judge_lane_by_mount_check_alone(
    tmp_path: Path,
) -> None:
    """AC3: the judge lane writes NO registry.*.json at all. Union-of-
    registries (AC1) contributes nothing for it by construction -- proving
    this case requires proving the mount check alone is sufficient.
    """
    versions = [f"0.5{i}.0" for i in range(6)]
    judge_version = versions[0]
    result, deployed_root = _run_prune(
        tmp_path,
        versions=versions,
        own_active_version=versions[-1],
        other_registries={},  # no stability-test, no lakshman, no judge -- none
        containers={
            "omninode-judge-runtime": [judge_version],
            "omninode-judge-runtime-effects": [judge_version],
        },
        max_deployments=1,
    )

    assert (deployed_root / judge_version).exists(), (
        "prune_old_deployments() deleted the judge lane's version. The judge "
        "lane has no registry.*.json, so no registry-union fix can protect "
        "it -- only containers_bound_to_deploy_dir() can (OMN-19910 AC3).\n"
        "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )


@pytest.mark.unit
def test_prune_still_removes_true_orphan_beyond_retention(tmp_path: Path) -> None:
    """Regression guard: a version with no registry reference anywhere and no
    live container mount is a true orphan and must still be pruned -- the
    fix must not turn pruning off altogether."""
    versions = [f"0.6{i}.0" for i in range(6)]
    orphan = versions[0]
    result, deployed_root = _run_prune(
        tmp_path,
        versions=versions,
        own_active_version=versions[-1],
        other_registries={"omnibase-infra-stability-test": versions[-2]},
        containers={"unrelated-container": []},
        max_deployments=1,
    )

    assert not (deployed_root / orphan).exists(), (
        "a true orphan deployment (no registry names it active, no live "
        "container mounts it, beyond the retention count) must still be "
        "pruned -- pruning must not regress to a no-op.\nstdout:\n"
        + result.stdout
        + "\nstderr:\n"
        + result.stderr
    )
    # The newest MAX_DEPLOYMENTS=1 kept, plus everything actively protected.
    assert (deployed_root / versions[-1]).exists(), (
        "the newest deployment within the retention count must never be "
        "removed.\nstdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )


@pytest.mark.unit
def test_prune_does_not_crash_when_no_registry_exists_anywhere(
    tmp_path: Path,
) -> None:
    """Regression guard: with ZERO registry.*.json files anywhere under
    DEPLOY_ROOT (a fresh host, or every lane pruning before its own first
    write_registry()), `active_paths` is legitimately empty. Under
    `set -euo pipefail`, an unguarded `"${active_paths[@]}"` expansion of an
    empty array is an unbound-variable hard error on bash < 4.4 (the array
    expansion must use the repo's own `${arr[@]+"${arr[@]}"}` guard, already
    used elsewhere in this same file at deploy-runtime.sh:4469). The script
    must still run to completion and still prune true orphans.
    """
    versions = [f"0.7{i}.0" for i in range(6)]
    orphan = versions[0]
    result, deployed_root = _run_prune(
        tmp_path,
        versions=versions,
        own_active_version=None,
        no_own_registry=True,
        other_registries={},
        containers={},
        max_deployments=1,
    )

    assert result.returncode == 0, (
        "prune_old_deployments() must not hard-fail when no registry file "
        "exists anywhere (empty active_paths array under set -u).\n"
        "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )
    assert not (deployed_root / orphan).exists(), (
        "an orphan must still be pruned even when active_paths is empty.\n"
        "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )
    assert (deployed_root / versions[-1]).exists(), (
        "the newest deployment within the retention count must never be "
        "removed.\nstdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
    )
