# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""cleanup_on_exit() must not delete a deploy dir live containers are mounted to (OMN-17287).

Defect: ``DEPLOY_DIR_TO_CLEANUP`` is armed at Phase 6 (right after
``sync_files``) and, since OMN-15352 made the registry write commit-on-success,
stays armed through Phase 11 (``restart_services``/``bringup_full_stack`` --
which recreates the lane's containers bind-mounted to ``deploy_target``) and
Phase 12 (verify + RT-6 readback). It is disarmed only after
``write_registry()``. So *any* failure or interruption between container start
and the registry commit runs ``cleanup_on_exit()`` -> ``rm -rf
"${DEPLOY_DIR_TO_CLEANUP}"`` while the lane's containers are still running with
that directory bind-mounted at ``/app/contracts``.

The removal is also the FIRST thing ``cleanup_on_exit()`` does -- it happens
before ``reconcile_runtime_container_start_state()`` gets any chance to settle
container state -- so the orphaning is structural, not a race.

Observed live on the ``.201`` dev lane 2026-08-31 (OMN-17287, discovered while
advancing the lane for OMN-17139): the ``0.38.16`` deploy died before
``write_registry()`` (``registry.json`` still pointed at ``0.38.13`` /
``c5a3c2d27325``), the trap removed
``~/.omnibase/infra/deployed/0.38.16/``, and Docker then re-created the missing
bind-mount sources as empty root-owned directories on each container restart.
With ``/app/contracts`` empty, ``load_runtime_config()`` took its "no config
file -> defaults" branch and returned ``ModelRuntimeConfig(name=None)``;
``service_kernel.py``'s ``if config.name:`` guard then never injected
``service_name``/``node_name``, and ``RuntimeHostProcess.__init__`` correctly
fail-fasted with ``ValueError: RuntimeHostProcess requires 'service_name'``.
``runtime-effects`` and ``runtime-worker`` crash-looped; ``omninode-runtime``
survived only because it had booted once (RestartCount=0) while the tree still
existed and held its contracts in memory.

Fix: a directory that live containers are bind-mounted to is by definition NOT
an orphan. ``cleanup_on_exit()`` must refuse to remove it, say so loudly, and
leave the lane recoverable (a re-run rsyncs over it) instead of converting a
recoverable partial deploy into a poisoned lane.

These tests drive the ACTUAL script seam per ``feedback_test_the_artifact_that_
runs``: ``cleanup_on_exit()`` is extracted VERBATIM from deploy-runtime.sh and
executed under bash, with only the true I/O boundary (``docker``) replaced by a
file-backed fake that answers ``ps``/``inspect`` from a fixture describing which
containers are running and what they have mounted.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
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

# Everything cleanup_on_exit() calls that is not central to this defect.
# containers_bound_to_deploy_dir() is deliberately NOT stubbed -- it is the
# function under test and is extracted verbatim once it exists.
_STUB_FUNCS = """
restore_latest_image_tags() { :; }
reconcile_runtime_container_start_state() { :; }
restore_migration_tree_after_revert() { :; }
"""


def _write_docker_stub(bin_dir: Path) -> None:
    """Fake `docker` answering `ps --quiet` and `inspect --format` from a
    fixture file: one line per running container, `<name>\\t<src1>,<src2>`."""
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${DOCKER_STUB_DIR}/calls.log"\n'
        'fixture="${DOCKER_STUB_DIR}/containers.tsv"\n'
        '[[ -f "${fixture}" ]] || exit 0\n'
        "\n"
        "# docker ps --quiet  -> one id per running container (id == name here)\n"
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


def _run_cleanup(
    tmp_path: Path, *, containers: dict[str, list[str]], deploy_target: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Execute the REAL cleanup_on_exit() with DEPLOY_DIR_TO_CLEANUP armed and
    registry.json pointing somewhere else (i.e. an uncommitted deploy)."""
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir(exist_ok=True)
    lock_dir = deploy_root / ".deploy.lock"
    lock_dir.mkdir(exist_ok=True)

    # registry.json points at a DIFFERENT (previous) deployment -- exactly the
    # .201 state: the new version was never committed.
    registry = deploy_root / "registry.json"
    registry.write_text(
        '{"deploy_path": "%s"}\n' % (deploy_root / "deployed" / "0.0.1"),
        encoding="utf-8",
    )

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)
    (stub_dir / "containers.tsv").write_text(
        "".join(f"{name}\t{','.join(srcs)}\n" for name, srcs in containers.items()),
        encoding="utf-8",
    )

    globals_prelude = "\n".join(
        [
            "set -uo pipefail",
            f'DEPLOY_ROOT="{deploy_root}"',
            'REGISTRY_FILE="${DEPLOY_ROOT}/registry.json"',
            'LOCK_DIR="${DEPLOY_ROOT}/.deploy.lock"',
            f'DEPLOY_DIR_TO_CLEANUP="{deploy_target}"',
            'FORCE_BACKUP_DIR=""',
            'MIGRATION_TREE_SNAPSHOT_DIR=""',
            'LATEST_TAG_SNAPSHOT_FILE=""',
            "DEPLOYMENT_COMPLETE=false",
        ]
    )

    parts = [globals_prelude, _LOG_FUNCS, _STUB_FUNCS]
    # Extract the guard helper verbatim once it exists (RED before the fix).
    if re.search(
        r"^containers_bound_to_deploy_dir\s*\(\)", _script_text(), re.MULTILINE
    ):
        parts.append(_extract_function("containers_bound_to_deploy_dir"))
    parts += [_extract_function("cleanup_on_exit"), "cleanup_on_exit"]

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
    return result, deploy_target


@pytest.mark.unit
def test_cleanup_refuses_to_delete_deploy_dir_with_live_container_mounts(
    tmp_path: Path,
) -> None:
    """RED on current behavior: the .201 OMN-17287 scenario exactly.

    Containers are already running with ``<deploy_target>/contracts`` bind
    mounted; the deploy then fails before write_registry(). cleanup_on_exit()
    must NOT remove the directory out from under them.
    """
    deploy_target = tmp_path / "deploy_root" / "deployed" / "0.38.16"
    (deploy_target / "contracts" / "runtime").mkdir(parents=True)
    (deploy_target / "contracts" / "runtime" / "runtime_config.yaml").write_text(
        "name: omnibase_infra\n", encoding="utf-8"
    )

    result, target = _run_cleanup(
        tmp_path,
        containers={
            "omninode-runtime": [str(deploy_target / "contracts")],
            "omninode-runtime-effects": [str(deploy_target / "contracts")],
        },
        deploy_target=deploy_target,
    )

    assert target.exists(), (
        "cleanup_on_exit() removed a deploy directory that live containers are "
        "bind-mounted to. This is the OMN-17287 defect: it strands the lane on "
        "a deleted payload, Docker re-creates the bind sources as empty dirs on "
        "the next restart, and the runtime crash-loops with "
        "\"RuntimeHostProcess requires 'service_name'\".\n" + result.stderr
    )
    assert (target / "contracts" / "runtime" / "runtime_config.yaml").is_file(), (
        "the bind-mounted contracts tree must survive intact.\n" + result.stderr
    )


@pytest.mark.unit
def test_cleanup_names_the_blocking_containers(tmp_path: Path) -> None:
    """The refusal must name which containers hold the directory, so the
    operator is not left to reconstruct it from `docker inspect` forensics."""
    deploy_target = tmp_path / "deploy_root" / "deployed" / "0.38.16"
    (deploy_target / "contracts").mkdir(parents=True)

    result, _ = _run_cleanup(
        tmp_path,
        containers={"omninode-runtime-effects": [str(deploy_target / "contracts")]},
        deploy_target=deploy_target,
    )

    assert "omninode-runtime-effects" in result.stderr, (
        "the refusal must name the container(s) still bind-mounted to the "
        "directory.\n" + result.stderr
    )


@pytest.mark.unit
def test_cleanup_still_removes_orphan_deploy_dir_with_no_live_mounts(
    tmp_path: Path,
) -> None:
    """Regression guard for OMN-15352: with NO container mounted, an
    uncommitted deploy directory is a true orphan and must still be removed."""
    deploy_target = tmp_path / "deploy_root" / "deployed" / "0.38.16"
    (deploy_target / "contracts").mkdir(parents=True)

    result, target = _run_cleanup(
        tmp_path,
        containers={"unrelated-container": ["/some/other/path"]},
        deploy_target=deploy_target,
    )

    assert not target.exists(), (
        "an orphaned deploy directory with no live container mounts must still "
        "be removed (OMN-15352 behavior must not regress).\n" + result.stderr
    )
