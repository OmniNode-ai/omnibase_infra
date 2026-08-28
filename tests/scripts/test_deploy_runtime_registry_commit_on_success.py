# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""deploy-runtime.sh must not write-ahead deployment state (OMN-15352).

Defect: `write_registry()` ran at Phase 9, BEFORE Phase 10 (`build_images`) and
every phase downstream of it (migration preflight, restart, RT-6 readback) that
can actually fail. `DEPLOY_DIR_TO_CLEANUP=""` immediately after the write also
disarmed the orphan-directory cleanup at the same moment the false claim was
made. The `--force` backup-restore branch in `cleanup_on_exit()` restored the
deployed directory + migration tree but explicitly declined to touch
`registry.json`, leaving it asserting the failed run's git_sha/deployed_at. A
failed deploy therefore left `registry.json` claiming a version that was never
actually running -- observed live 2026-07-29T00:26-00:41Z on the `.201` dev lane
(workflow `wf_55998f90`; OMN-15352 description + comment `1564e60c`).
F3 (companion defect): the implicit `docker compose build` retag of
`<compose_project>-<service>:latest` was never protected -- a failed deploy
left `:latest` resolving to an untested image, so a later
`docker compose up -d` without `--build` would silently swap it in.

Fix: `write_registry()` moves to commit-on-success -- it is called once, right
before `DEPLOYMENT_COMPLETE=true`, after every phase that can fail has passed.
`DEPLOY_DIR_TO_CLEANUP` stays armed for the whole deploy instead of being
disabled right after the (now-removed) early write, so a failure at any later
phase lets `cleanup_on_exit()` remove the orphaned directory. A new
`snapshot_latest_image_tags()` / `restore_latest_image_tags()` pair records
each RUNTIME_BUILD_SERVICES image's pre-build `:latest` id and restores it (or
removes an unverified tag that had no prior state) on any non-success exit.

These tests drive the ACTUAL script seam per `feedback_test_the_artifact_that_
runs`: `main()` is extracted verbatim (unmodified control flow, unmodified
`write_registry()` / `cleanup_on_exit()` / `snapshot_latest_image_tags()` /
`restore_latest_image_tags()` / `guard_existing_deployment()` /
`restore_migration_tree_after_revert()` / `snapshot_migration_tree()` /
`assert_deployed_migration_tree_synced()`) and executed under bash with only
the true I/O boundaries stubbed: `docker` is a local file-backed fake (no
daemon required), and the heavy phases with no bearing on this defect
(rsync-driven sync, compose validation, network/Kafka/Postgres readiness,
attribution/lineage preflights) are replaced with no-op or controllable fakes.
This is not a surrogate reimplementation of the ordering -- the real bash
`main()` text runs, and a real `jq`-backed `write_registry()` either does or
does not touch the filesystem depending on where in that real control flow the
injected failure lands.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

FAKE_SERVICES = ["fake-svc-a", "fake-svc-b"]


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _script_noncomment() -> str:
    """deploy-runtime.sh with comment-only lines stripped (see sibling tests)."""
    lines = [
        line
        for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def _extract_function(name: str) -> str:
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


# ---------------------------------------------------------------------------
# Static wiring assertions (repo idiom): prove the ordering in the source text
# without executing anything.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_registry_call_moved_after_readback_in_main() -> None:
    """write_registry() must be called AFTER readback_deployed_ref(), not before build."""
    text = _script_text()
    main_body = text[text.index("\nmain() {") :]
    build_idx = main_body.index('build_images "${deploy_target}"')
    readback_idx = main_body.index('readback_deployed_ref "${git_sha}"')
    write_registry_idx = main_body.index('write_registry "${version}"')
    complete_idx = main_body.index("DEPLOYMENT_COMPLETE=true")

    assert build_idx < readback_idx, "test fixture assumption: build precedes readback"
    assert readback_idx < write_registry_idx, (
        "write_registry() must run AFTER readback_deployed_ref() (commit-on-"
        "success, OMN-15352) -- it must not still run before the phases that "
        "can fail."
    )
    assert write_registry_idx < complete_idx, (
        "write_registry() must run BEFORE DEPLOYMENT_COMPLETE=true is set, so "
        "the registry commit and the completion flag land together."
    )


@pytest.mark.unit
def test_deploy_dir_cleanup_not_disarmed_before_build() -> None:
    """DEPLOY_DIR_TO_CLEANUP must stay armed through build/restart/readback.

    The old code cleared DEPLOY_DIR_TO_CLEANUP="" immediately after the
    (write-ahead) registry write, before build_images() ran -- disarming the
    orphan-directory cleanup at the exact moment the false registry claim was
    made. Only one `DEPLOY_DIR_TO_CLEANUP=""` assignment may exist in main(),
    and it must be co-located with the (now deferred) registry write, after
    build/restart/readback.
    """
    text = _script_noncomment()
    main_body = text[text.index("\nmain() {") :]
    clear_positions = [
        m.start() for m in re.finditer(r'DEPLOY_DIR_TO_CLEANUP=""', main_body)
    ]
    assert len(clear_positions) == 1, (
        f'expected exactly one DEPLOY_DIR_TO_CLEANUP="" in main(), found '
        f"{len(clear_positions)} -- an early clear before build/restart/readback "
        "would re-open the write-ahead window."
    )
    build_idx = main_body.index('build_images "${deploy_target}"')
    readback_idx = main_body.index('readback_deployed_ref "${git_sha}"')
    assert clear_positions[0] > build_idx, (
        "DEPLOY_DIR_TO_CLEANUP must not be disarmed before build_images() runs."
    )
    assert clear_positions[0] > readback_idx, (
        "DEPLOY_DIR_TO_CLEANUP must not be disarmed before readback_deployed_ref() runs."
    )


@pytest.mark.unit
def test_snapshot_latest_image_tags_runs_before_build() -> None:
    text = _script_text()
    main_body = text[text.index("\nmain() {") :]
    snap_idx = main_body.index('snapshot_latest_image_tags "${compose_project}"')
    build_idx = main_body.index('build_images "${deploy_target}"')
    assert snap_idx < build_idx, (
        "snapshot_latest_image_tags() must run BEFORE build_images() so it "
        "captures the pre-build :latest state (OMN-15352 F3)."
    )


@pytest.mark.unit
def test_cleanup_on_exit_restores_latest_tags_only_on_non_success() -> None:
    body = _extract_function("cleanup_on_exit")
    match = re.search(
        r'if \[\[ "\$\{DEPLOYMENT_COMPLETE\}" != "true" \]\]; then\s*\n\s*restore_latest_image_tags',
        body,
    )
    assert match is not None, (
        "restore_latest_image_tags() must be called from cleanup_on_exit() "
        "guarded on DEPLOYMENT_COMPLETE != true, so a successful deploy never "
        "reverts its own freshly-built :latest tag."
    )


@pytest.mark.unit
def test_force_restore_branch_no_longer_claims_registry_is_stale() -> None:
    """The backup-restore branch must not warn about stale registry metadata.

    Now that write_registry() is commit-on-success, the restore branch (which
    only runs when DEPLOYMENT_COMPLETE != true, i.e. write_registry() never ran
    this invocation) can never leave registry.json stale -- the old log_warn
    was prose describing a problem the write-ahead ordering created; it must
    not survive un-mechanized.
    """
    text = _script_text()
    assert "may contain stale metadata" not in text, (
        "the stale-registry warning is obsolete: write_registry() is now "
        "commit-on-success, so a restore branch never runs after it wrote "
        "this invocation's registry entry (OMN-15352)."
    )


# ---------------------------------------------------------------------------
# Executed harness -- drives the real main() control flow end to end.
# ---------------------------------------------------------------------------

_LOG_FUNCS = """
log_step() { printf 'STEP: %s\\n' "$*" >&2; }
log_info() { printf 'INFO: %s\\n' "$*" >&2; }
log_warn() { printf 'WARN: %s\\n' "$*" >&2; }
log_error() { printf 'ERR: %s\\n' "$*" >&2; }
log_cmd() { printf 'CMD: %s\\n' "$*" >&2; }
"""

# Every function main() calls that is NOT central to the OMN-15352 fix is
# replaced with a small controllable fake. The functions central to the fix
# (write_registry, cleanup_on_exit, snapshot_latest_image_tags,
# restore_latest_image_tags, guard_existing_deployment,
# restore_migration_tree_after_revert, snapshot_migration_tree,
# assert_deployed_migration_tree_synced) are extracted VERBATIM from the real
# script below and are never stubbed.
_STUB_FUNCS = """
parse_args() { :; }
validate_prerequisites() { :; }
resolve_repo_root() { printf '%s\\n' "${FAKE_REPO_ROOT}"; }
validate_repo_structure() { :; }
read_version() { printf '%s\\n' "${FAKE_VERSION}"; }
read_git_sha() { printf '%s\\n' "${FAKE_GIT_SHA}"; }
check_git_dirty() { :; }
validate_build_source_config() { :; }
guard_prod_promotion_lineage() { :; }
resolve_compose_project() { printf '%s\\n' "${FAKE_COMPOSE_PROJECT}"; }
guard_cold_bringup_lane_scope() { :; }
guard_lane_deploy_attribution() { :; }
guard_hotpatch_ledger() { :; }
check_compose_project_collision() { :; }
show_preview() { :; }
acquire_lock() { :; }
sync_files() {
    local dst="$2"
    mkdir -p "${dst}"
    printf '%s' "${FAKE_MARKER_CONTENT}" > "${dst}/marker.txt"
}
sanity_check() { :; }
build_images() {
    if [[ "${FAKE_BUILD_FAIL:-0}" == "1" ]]; then
        log_error "fake build failure"
        return 1
    fi
    local project="$2"
    local svc
    for svc in "${RUNTIME_BUILD_SERVICES[@]}"; do
        docker tag "${FAKE_NEW_IMAGE_ID}-${svc}" "${project}-${svc}:latest"
    done
    return 0
}
ensure_core_infra_ready() { :; }
warm_broker_topic_provisioning() { :; }
run_runtime_migration_preflight() {
    if [[ "${FAKE_MIGRATION_FAIL:-0}" == "1" ]]; then
        log_error "fake migration preflight failure"
        return 1
    fi
    return 0
}
bringup_full_stack() { :; }
restart_services() { :; }
verify_deployment() { :; }
readback_deployed_ref() {
    if [[ "${FAKE_READBACK_FAIL:-0}" == "1" ]]; then
        log_error "fake readback failure"
        return 1
    fi
    return 0
}
show_summary() { :; }
prune_old_deployments() { :; }
"""


def _write_docker_stub(bin_dir: Path) -> None:
    """Fake `docker` answering `image inspect`, `tag`, `rmi` from a file-backed
    store under $DOCKER_STUB_DIR/images, logging every call to calls.log."""
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${DOCKER_STUB_DIR}/calls.log"\n'
        "\n"
        'if [[ "$1" == "image" && "$2" == "inspect" ]]; then\n'
        '    ref="$3"\n'
        '    safe="$(printf "%s" "${ref}" | tr "/:" "__")"\n'
        '    file="${DOCKER_STUB_DIR}/images/${safe}"\n'
        '    if [[ -f "${file}" ]]; then\n'
        '        cat "${file}"\n'
        "        exit 0\n"
        "    fi\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        'if [[ "$1" == "tag" ]]; then\n'
        '    src="$2"\n'
        '    dest="$3"\n'
        '    safe="$(printf "%s" "${dest}" | tr "/:" "__")"\n'
        '    printf "%s" "${src}" > "${DOCKER_STUB_DIR}/images/${safe}"\n'
        "    exit 0\n"
        "fi\n"
        "\n"
        'if [[ "$1" == "rmi" ]]; then\n'
        '    ref="$2"\n'
        '    safe="$(printf "%s" "${ref}" | tr "/:" "__")"\n'
        '    rm -f "${DOCKER_STUB_DIR}/images/${safe}"\n'
        "    exit 0\n"
        "fi\n"
        "\n"
        "exit 1\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _read_latest_tag(stub_dir: Path, compose_project: str, service: str) -> str | None:
    safe = f"{compose_project}-{service}:latest".replace("/", "_").replace(":", "_")
    f = stub_dir / "images" / safe
    return f.read_text(encoding="utf-8") if f.is_file() else None


def _build_harness(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """Assemble the full harness script text: log fns, stubs, real extracted
    functions (verbatim), the real main(), then `trap cleanup_on_exit EXIT;
    main "$@"`. Returns (script_text, base_env)."""
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir(exist_ok=True)
    stub_dir = tmp_path / "stubs"
    (stub_dir / "images").mkdir(parents=True, exist_ok=True)
    if not (stub_dir / "calls.log").exists():
        (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)

    fake_repo_root = tmp_path / "fake_repo"
    fake_repo_root.mkdir(exist_ok=True)

    services_literal = " ".join(f'"{s}"' for s in FAKE_SERVICES)

    globals_prelude = "\n".join(
        [
            "set -euo pipefail",
            f'DEPLOY_ROOT="{deploy_root}"',
            'REGISTRY_FILE="${DEPLOY_ROOT}/registry.json"',
            'LOCK_DIR="${DEPLOY_ROOT}/.deploy.lock"',
            'SCRIPT_NAME="deploy-runtime.sh"',
            f"RUNTIME_BUILD_SERVICES=({services_literal})",
            'MIGRATION_TREE_REL_PATH="docker/migrations/forward"',
            'DEPLOY_DIR_TO_CLEANUP=""',
            'FORCE_BACKUP_DIR=""',
            'MIGRATION_TREE_SNAPSHOT_DIR=""',
            'LATEST_TAG_SNAPSHOT_FILE=""',
            'DEPLOY_COMPOSE_PROJECT=""',
            "DEPLOYMENT_COMPLETE=false",
            'LANE_ATTRIBUTION_RECORD_JSON=""',
            'COMPOSE_PROFILE="runtime"',
            'MODE="execute"',
            "PRINT_COMPOSE_CMD=false",
            "PROD_LANE=false",
            "COLD_FULL_BRINGUP=false",
            "COLD_START_KAFKA_TIMEOUT_SECONDS=180",
            'FORCE="${FAKE_FORCE:-false}"',
            'RESTART="${FAKE_RESTART:-false}"',
            "DEPLOY_INVOCATION_ARGS=()",
        ]
    )

    script = "\n".join(
        [
            globals_prelude,
            _LOG_FUNCS,
            _STUB_FUNCS,
            _extract_function("guard_existing_deployment"),
            _extract_function("assert_deployed_migration_tree_synced"),
            _extract_function("snapshot_migration_tree"),
            _extract_function("restore_migration_tree_after_revert"),
            _extract_function("snapshot_latest_image_tags"),
            _extract_function("restore_latest_image_tags"),
            _extract_function("write_registry"),
            _extract_function("cleanup_on_exit"),
            _extract_function("main"),
            "trap 'cleanup_on_exit' EXIT",
            'main "$@"',
        ]
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)
    env["FAKE_REPO_ROOT"] = str(fake_repo_root)
    env["FAKE_COMPOSE_PROJECT"] = "fake-project"
    env["FAKE_MARKER_CONTENT"] = "fresh-sync-content"
    env["FAKE_NEW_IMAGE_ID"] = "built-new"

    return script, env


def _run(
    tmp_path: Path,
    *,
    restart: bool,
    force: bool,
    version: str,
    git_sha: str,
    migration_fail: bool = False,
    readback_fail: bool = False,
    build_fail: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    script, env = _build_harness(tmp_path)
    env["FAKE_VERSION"] = version
    env["FAKE_GIT_SHA"] = git_sha
    env["FAKE_MIGRATION_FAIL"] = "1" if migration_fail else "0"
    env["FAKE_READBACK_FAIL"] = "1" if readback_fail else "0"
    env["FAKE_BUILD_FAIL"] = "1" if build_fail else "0"
    env["FAKE_RESTART"] = "true" if restart else "false"
    env["FAKE_FORCE"] = "true" if force else "false"

    harness = tmp_path / "harness.sh"
    harness.write_text(script, encoding="utf-8")

    args = ["--execute"]
    if restart:
        args.append("--restart")
    if force:
        args.append("--force")

    result = subprocess.run(
        ["bash", str(harness), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )
    deploy_root = tmp_path / "deploy_root"
    deploy_target = deploy_root / "deployed" / version
    return result, deploy_root, deploy_target


@pytest.mark.unit
def test_fresh_deploy_failed_migration_preflight_never_writes_registry(
    tmp_path: Path,
) -> None:
    """RED on the old behavior: a fresh (non--force) deploy that fails at the
    migration preflight (after build) must leave NO registry.json at all --
    write_registry() must never have been reached."""
    result, deploy_root, deploy_target = _run(
        tmp_path,
        restart=True,
        force=False,
        version="9.9.9",
        git_sha="abc123def456",
        migration_fail=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr

    registry_file = deploy_root / "registry.json"
    assert not registry_file.exists(), (
        "registry.json must not exist after a failed fresh deploy -- "
        "write_registry() must run only after the migration preflight (and "
        "everything before it) has succeeded. stderr:\n" + result.stderr
    )
    # Orphan cleanup: DEPLOY_DIR_TO_CLEANUP stayed armed through build/preflight
    # (no early clear), so cleanup_on_exit() must have removed the freshly
    # synced (but never-committed) deploy_target directory.
    assert not deploy_target.exists(), (
        "the orphaned deploy_target directory must be removed by "
        "cleanup_on_exit() when it was never committed to the registry.\n"
        + result.stderr
    )


@pytest.mark.unit
def test_fresh_deploy_failed_migration_preflight_untags_unverified_latest(
    tmp_path: Path,
) -> None:
    """F3: a fresh deploy's :latest tag (no prior state) must be removed, not
    left resolving to the untested image built just before the failure."""
    tmp_path_stubs = tmp_path
    result, _deploy_root, _ = _run(
        tmp_path_stubs,
        restart=True,
        force=False,
        version="9.9.9",
        git_sha="abc123def456",
        migration_fail=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr

    stub_dir = tmp_path / "stubs"
    for svc in FAKE_SERVICES:
        tag = _read_latest_tag(stub_dir, "fake-project", svc)
        assert tag is None, (
            f"fake-project-{svc}:latest must be removed after a failed deploy "
            f"with no prior :latest state, got {tag!r}.\n{result.stderr}"
        )


@pytest.mark.unit
def test_force_redeploy_failed_migration_preflight_leaves_registry_byte_identical(
    tmp_path: Path,
) -> None:
    """AC1/AC2: a --force redeploy over an existing (registered, running)
    deployment that fails at the migration preflight must leave registry.json
    BYTE-IDENTICAL to its pre-run content, and the deploy_target directory and
    :latest tags restored to their pre-run state."""
    version = "9.9.9"
    old_git_sha = "111111111111"
    new_git_sha = "222222222222"

    # --- Seed a pre-existing deployment: registry.json (written by the REAL
    # write_registry(), so byte-for-byte format matches) + a deploy_target
    # directory with OLD content + a prior :latest tag per service.
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir()
    deploy_target = deploy_root / "deployed" / version
    deploy_target.mkdir(parents=True)
    (deploy_target / "marker.txt").write_text("old-content", encoding="utf-8")

    stub_dir = tmp_path / "stubs"
    (stub_dir / "images").mkdir(parents=True)
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)
    for svc in FAKE_SERVICES:
        safe = f"fake-project-{svc}:latest".replace("/", "_").replace(":", "_")
        (stub_dir / "images" / safe).write_text(f"prior-{svc}", encoding="utf-8")

    seed_script = "\n".join(
        [
            "set -euo pipefail",
            f'DEPLOY_ROOT="{deploy_root}"',
            'REGISTRY_FILE="${DEPLOY_ROOT}/registry.json"',
            'LANE_ATTRIBUTION_RECORD_JSON=""',
            'COMPOSE_PROFILE="runtime"',
            _LOG_FUNCS,
            _extract_function("write_registry"),
            (
                f'write_registry "{version}" "{old_git_sha}" '
                f'"{deploy_target}" "/fake/repo" "fake-project"'
            ),
        ]
    )
    seed_env = dict(os.environ)
    seed_result = subprocess.run(
        ["bash", "-c", seed_script],
        capture_output=True,
        text=True,
        check=False,
        env=seed_env,
        timeout=30,
    )
    assert seed_result.returncode == 0, seed_result.stdout + seed_result.stderr
    registry_file = deploy_root / "registry.json"
    pre_run_registry_bytes = registry_file.read_bytes()
    assert b"111111111111" in pre_run_registry_bytes

    # --- Run the real harness: --force redeploy, fails at migration preflight.
    script, env = _build_harness(tmp_path)
    env["FAKE_VERSION"] = version
    env["FAKE_GIT_SHA"] = new_git_sha
    env["FAKE_MIGRATION_FAIL"] = "1"
    env["FAKE_READBACK_FAIL"] = "0"
    env["FAKE_BUILD_FAIL"] = "0"
    env["FAKE_MARKER_CONTENT"] = "new-content"
    env["FAKE_RESTART"] = "true"
    env["FAKE_FORCE"] = "true"

    harness = tmp_path / "harness.sh"
    harness.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(harness), "--execute", "--restart", "--force"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )
    assert result.returncode != 0, result.stdout + result.stderr

    # AC1: registry.json byte-identical to its pre-run content.
    post_run_registry_bytes = registry_file.read_bytes()
    assert post_run_registry_bytes == pre_run_registry_bytes, (
        "registry.json must be byte-identical after a failed --force redeploy "
        "-- write_registry() must never have run this invocation.\n"
        f"pre:  {pre_run_registry_bytes!r}\n"
        f"post: {post_run_registry_bytes!r}\n"
        f"stderr: {result.stderr}"
    )

    # The directory must be restored to its pre-run (OLD) content, not left on
    # the freshly-synced (new, failed) content.
    assert (deploy_target / "marker.txt").read_text(
        encoding="utf-8"
    ) == "old-content", (
        "deploy_target must be restored to its pre-run content on a failed "
        f"--force redeploy.\n{result.stderr}"
    )

    # F3: :latest must resolve to the same image id as pre-run for every
    # service (restored from the snapshot taken before the build).
    for svc in FAKE_SERVICES:
        tag = _read_latest_tag(stub_dir, "fake-project", svc)
        assert tag == f"prior-{svc}", (
            f"fake-project-{svc}:latest must be restored to its pre-run id "
            f"'prior-{svc}', got {tag!r}.\n{result.stderr}"
        )


@pytest.mark.unit
def test_success_path_writes_registry_and_keeps_new_latest_tags(
    tmp_path: Path,
) -> None:
    """AC3: an unregressed success path still ends with registry.json
    reflecting the newly deployed SHA and :latest pointing at the newly built
    image (no restore fires on a completed deploy)."""
    version = "9.9.9"
    git_sha = "abc123def456"

    result, deploy_root, deploy_target = _run(
        tmp_path,
        restart=True,
        force=False,
        version=version,
        git_sha=git_sha,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    registry_file = deploy_root / "registry.json"
    assert registry_file.is_file(), "a successful deploy must write registry.json"
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    assert registry["git_sha"] == git_sha
    assert registry["active_version"] == version
    assert registry["deploy_path"] == str(deploy_target)

    stub_dir = tmp_path / "stubs"
    for svc in FAKE_SERVICES:
        tag = _read_latest_tag(stub_dir, "fake-project", svc)
        assert tag == f"built-new-{svc}", (
            f"a successful deploy must keep its newly built :latest tag for "
            f"{svc}, got {tag!r} (a restore must not have fired).\n{result.stderr}"
        )
