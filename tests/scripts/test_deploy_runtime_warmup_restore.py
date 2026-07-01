# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression coverage for the OMN-13364 stability-redeploy fragility fixes.

Two coupled deploy-runtime.sh defects, observed on the 2026-06-19 stability-test
redeploy, are guarded here:

1. Redpanda warmup false-fails on a prefixed-container-name conflict.
   ``warm_broker_topic_provisioning()`` ran ``docker compose up -d --no-deps
   --wait redpanda`` and treated the compose-wait exit status as the source of
   truth for broker readiness. When the broker container_name collides with
   another project's broker, Docker assigns a random prefix (e.g.
   ``3ed1fdb8d50b_omnibase-infra-redpanda``) and/or leaves the recreate in
   'Created'; ``up -d --wait`` then errors / never reaches healthy even though a
   healthy broker is already reachable on the lane network. The warmup must key
   readiness off ACTUAL broker reachability (``rpk cluster health`` against the
   broker resolved by compose label, not by an exact container-name string), and
   must tolerate a prefixed name and an already-present healthy broker.

2. Backup-restore reverts the vendored forward-migration tree. On that false-fail
   (with ``--force``), ``cleanup_on_exit()`` restores the WHOLE pre-build
   deployment tree, including ``docker/migrations/forward/``, silently regressing
   the deployed migrations to the backup's stale snapshot (it dropped
   ``node_projection_delegation/0015_generation_corpus_acceptance.sql``). The
   restore must re-apply the freshly-synced vendored migration tree so the
   deployed migrations always match the build source (origin/dev), never silently
   losing a forward migration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
MIGRATION_TREE = REPO_ROOT / "docker" / "migrations" / "forward"


def _deploy_script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _deploy_script_noncomment() -> str:
    """deploy-runtime.sh with comment-only lines stripped.

    Assertions about *active* behavior must not be satisfied by a comment that
    merely mentions the token, so the active-code checks run against this view.
    """
    lines = [
        line
        for line in _deploy_script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix 1: warmup tolerates a prefixed / already-healthy broker
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_broker_resolved_by_compose_label_not_exact_name() -> None:
    """The broker must be resolved by compose label, surviving a Docker prefix.

    A Docker name-prefix collision (3ed1fdb8d50b_omnibase-infra-redpanda) breaks
    any exact container-name handle, but the compose service label survives. The
    resolver must filter on com.docker.compose.service=<broker>, not match an
    exact container_name string.
    """
    text = _deploy_script_noncomment()
    assert "resolve_broker_container()" in text, (
        "resolve_broker_container helper missing; the warmup would still key off "
        "an exact container-name string that breaks under Docker's prefix-on-"
        "collision behavior (OMN-13364)."
    )
    assert "label=com.docker.compose.service=${BROKER_READINESS_SERVICE}" in text, (
        "Broker must be resolved by compose service label, not exact name."
    )
    assert "label=com.docker.compose.project=${compose_project}" in text, (
        "Broker resolution must be scoped to the lane's compose project."
    )


@pytest.mark.unit
def test_broker_readiness_keyed_off_rpk_cluster_health() -> None:
    """Broker readiness must be probed via rpk cluster health, not compose-wait."""
    text = _deploy_script_noncomment()
    assert "assert_broker_reachable()" in text, (
        "assert_broker_reachable helper missing; readiness would still depend on "
        "the compose --wait exit status, which false-fails on a name collision."
    )
    assert "rpk cluster health" in text, (
        "Broker readiness must be keyed off actual reachability (rpk cluster "
        "health on TCP/9092 inside the lane), not an exact container-name match."
    )


@pytest.mark.unit
def test_compose_wait_failure_is_tolerated_not_fatal() -> None:
    """A failing compose `up --wait` must NOT abort the warmup by itself.

    Under ``set -e`` an unguarded ``"${broker_up_cmd[@]}"`` aborts the script on
    a name-prefix collision, which triggers the destructive backup-restore. The
    call must be guarded so the reachability probe gets the final say.
    """
    text = _deploy_script_text()
    match = re.search(
        r"warm_broker_topic_provisioning\(\)\s*\{(?P<body>.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match is not None, "warm_broker_topic_provisioning function not found"
    body = match.group("body")
    # The broker compose-up must be guarded (if ! ...; then) rather than called
    # bare, so its failure does not bubble up under set -e.
    assert 'if ! "${broker_up_cmd[@]}"; then' in body, (
        "The broker compose up --wait must be guarded so a name-prefix-collision "
        "failure is tolerated and the rpk reachability probe decides readiness "
        "(OMN-13364)."
    )
    # And the warmup must still hard-fail when the broker is genuinely unreachable.
    assert "assert_broker_reachable" in body
    assert "Broker is not reachable on the lane network after warmup." in body


# ---------------------------------------------------------------------------
# Fix 2: backup-restore preserves the vendored forward-migration tree
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_migration_tree_snapshotted_after_sync() -> None:
    """The freshly-synced vendored migration tree must be snapshotted post-sync."""
    text = _deploy_script_noncomment()
    assert "snapshot_migration_tree()" in text, (
        "snapshot_migration_tree helper missing; without a snapshot the restore "
        "path cannot re-apply the freshly-built migrations (OMN-13364)."
    )
    # The snapshot must be taken right after sync_files in main().
    sync_idx = text.find('sync_files "${repo_root}" "${deploy_target}"')
    snap_idx = text.find('snapshot_migration_tree "${deploy_target}"')
    assert sync_idx != -1, "sync_files is not called in main()"
    assert snap_idx != -1, "snapshot_migration_tree is not called in main()"
    assert sync_idx < snap_idx, (
        "snapshot_migration_tree must run AFTER sync_files so it captures the "
        "freshly-synced (build-source) migration tree, not the backup's."
    )


@pytest.mark.unit
def test_restore_reapplies_migration_tree() -> None:
    """cleanup_on_exit must re-apply the migration tree after a backup-restore."""
    text = _deploy_script_text()
    assert "restore_migration_tree_after_revert()" in text, (
        "restore_migration_tree_after_revert helper missing; a backup-restore "
        "would silently regress the deployed migrations to the pre-build "
        "snapshot (OMN-13364)."
    )
    # The restore must be wired into the successful-restore branch of cleanup.
    match = re.search(
        r"cleanup_on_exit\(\)\s*\{(?P<body>.*?)\nsnapshot_migration_tree",
        text,
        re.DOTALL,
    )
    assert match is not None, "cleanup_on_exit function not found"
    body = match.group("body")
    # The re-apply call must follow a successful `mv` restore (the else branch).
    restore_idx = body.find('restore_migration_tree_after_revert "${original_dir}"')
    assert restore_idx != -1, (
        "restore_migration_tree_after_revert must be called after a successful "
        "backup-restore so the deployed migrations match the build source."
    )


@pytest.mark.unit
def test_restore_uses_delete_to_match_build_source_exactly() -> None:
    """The re-apply must use rsync --delete so the tree matches the build source.

    A plain copy would leave stale files from the backup tree; --delete makes the
    re-applied tree a faithful mirror of the freshly-built migration tree.
    """
    text = _deploy_script_text()
    match = re.search(
        r"restore_migration_tree_after_revert\(\)\s*\{(?P<body>.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match is not None, "restore_migration_tree_after_revert function not found"
    body = match.group("body")
    assert "rsync -a --delete" in body, (
        "The migration-tree re-apply must rsync --delete so it mirrors the "
        "build-source tree exactly (no stale files survive)."
    )


@pytest.mark.unit
def test_snapshot_cleaned_up_on_exit() -> None:
    """The migration-tree snapshot must be removed in cleanup_on_exit."""
    text = _deploy_script_noncomment()
    assert 'rm -rf "${MIGRATION_TREE_SNAPSHOT_DIR}"' in text, (
        "The migration-tree snapshot dir must be cleaned up so deploys do not "
        "accumulate orphaned snapshot trees."
    )


# ---------------------------------------------------------------------------
# Guard: the migration the revert dropped is actually in the source tree
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dropped_delegation_migration_present_in_source() -> None:
    """The forward migration the 2026-06-19 revert dropped must exist in source.

    This is the file a backup-restore silently regressed away. If the vendored
    migration tree ever stops carrying it, the snapshot/re-apply fix protects an
    empty target — fail loudly so the regression is visible.
    """
    dropped = (
        MIGRATION_TREE
        / "nodes"
        / "node_projection_delegation"
        / "0015_generation_corpus_acceptance.sql"
    )
    assert dropped.is_file(), (
        f"Expected vendored forward migration missing: {dropped}. The "
        "snapshot/re-apply fix (OMN-13364) protects this tree; if it is gone the "
        "deploy would have nothing to preserve."
    )
