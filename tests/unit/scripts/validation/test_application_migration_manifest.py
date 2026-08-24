# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED/green proof for the deterministic application migration manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = (
    REPO_ROOT / "scripts" / "validation" / "validate_application_migration_manifest.py"
)
MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER_DIR = MIGRATIONS_DIR / "_ledger"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_application_migration_manifest", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _validate(
    migrations_dir: Path = MIGRATIONS_DIR,
    ledger_dir: Path = LEDGER_DIR,
    *,
    require_complete: bool = False,
) -> Any:
    return validator.validate_manifests(
        migrations_dir,
        ledger_dir / "application-migrations.tsv",
        ledger_dir / "application-migration-blocks.tsv",
        ledger_dir / "legacy-node-migrations.tsv",
        ledger_dir / "cloud-migration-aliases.tsv",
        require_complete=require_complete,
    )


def _minimal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    migrations_dir = tmp_path / "forward"
    ledger_dir = migrations_dir / "_ledger"
    artifact = migrations_dir / "nodes" / "node_example" / "0001.sql"
    artifact.parent.mkdir(parents=True)
    ledger_dir.mkdir(parents=True)
    artifact.write_text("SELECT 1;\n", encoding="utf-8")
    checksum = validator._content_sha256(artifact)
    (ledger_dir / "application-migrations.tsv").write_text(
        "\t".join(
            (
                "nodes/node_example/0001.sql",
                "node:node_example",
                "node:node_example",
                "omninode_internal",
                "node:node_example:0001.sql",
                checksum,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    blocked_artifact = migrations_dir / "nodes" / "node_example" / "0002.sql"
    blocked_artifact.write_text("SELECT 2;\n", encoding="utf-8")
    blocked_checksum = validator._content_sha256(blocked_artifact)
    (ledger_dir / "application-migration-blocks.tsv").write_text(
        "\t".join(
            (
                "nodes/node_example/0002.sql",
                "node:node_example:0002.sql",
                blocked_checksum,
                "OMN-99999",
                "classification pending",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (ledger_dir / "cloud-migration-aliases.tsv").write_text(
        "20260101_example\t20260101_example.sql\n", encoding="utf-8"
    )
    (ledger_dir / "legacy-node-migrations.tsv").write_text("", encoding="utf-8")
    return migrations_dir, ledger_dir


def test_checked_in_manifest_is_exact_and_all_blockers_are_explicit() -> None:
    result = _validate()

    # 98 as of OMN-15819 (rebased onto OMN-15717): 97 from OMN-15717 (94 from
    # the OMN-14894 vendor-parity repair -- the manifest mirrors the vendored
    # node migration tree after restoring the nine house-tenant RLS
    # migrations that are still source-owned in omnimarket dev -- +2 for
    # node_canary_score_reducer/0003 and node_projection_registration/0004,
    # see OMN-15361 exemption fix above for why these are the
    # deadlock-triggering vendor files, +1 for OMN-15717's legacy-declared
    # node_pr_review_bot/001_create_review_bot_bypass_log.sql), +1 for
    # nodes/node_projection_live_events/0002_create_omninode_internal_live_events.sql
    # -- the node-owned replacement that delivers omninode_internal.live_events
    # through the node-owned loop, since the flat
    # 099_create_omninode_internal_live_events.sql migration has no execution
    # path in the k8s Job that applies flat migrations (cross-DB \connect),
    # +1 for OMN-15846's nodes/node_log_persistence_effect/0000_create_log_entries.sql
    # -- the node-owned replacement that delivers log_entries through the
    # node-owned loop, since the flat 083_create_log_entries.sql migration has
    # the same no-execution-path defect (cross-DB \connect),
    # +2 for OMN-16090's node_hook_event_capture pair --
    # 0001_create_hook_events.sql (the table; applies unfenced) and
    # 0002_hook_events_tenant_rls.sql (the RLS posture; FENCED on arrival in
    # fenced-node-migrations.yaml, because the forward runner refuses any new
    # unfenced FORCE-RLS migration). Both are DECLARED here regardless of the
    # fence: a fenced migration is skipped by the runner, not undeclared --
    # dropping its declaration would make the eventual un-fence land an
    # unbound file.
    # +1 for OMN-16146's node_projection_registration/0005_create_projection_watermarks.sql
    # -- vendors the watermark-persistence table BaseProjectionRunner's shared
    # _update_watermark() path needs; landed in omnibase_infra first per the
    # node-migration-vendor-parity-gate ordering, ahead of omnimarket#2092.
    # +1 for OMN-15631's node_delegation_routing_reducer/0001_create_delegation_routing_tenant_overlay.sql
    # -- vendors the v1(a) per-tenant delegation routing overlay table (tenant
    # domain, additive, no RLS in v1(a)); landed in omnibase_infra first per
    # the node-migration-vendor-parity-gate ordering, ahead of omnimarket#2116.
    # +1 for OMN-16316's
    # node_projection_tenant_credentials/0000_create_tenant_inference_credentials.sql
    # -- vendors the BYOK inference-credential-ref catalog table, domain
    # `tenant` (per-tenant credential data, not omninode_internal per the
    # house-tenant ruling); landed in omnibase_infra first per the
    # node-migration-vendor-parity-gate ordering, ahead of omnimarket#2117.
    # +1 for OMN-16293's node_savings_estimation_compute/0001_create_savings_signal_tables.sql
    # -- new node-owned migration creating savings_injection_signals /
    # savings_validator_catch_signals, the Postgres "projection surface" the
    # savings-correlation periodic batch reads instead of an in-memory buffer.
    # +1 for OMN-15683's
    # node_projection_delegation/0031_delegation_events_tenant_id_to_uuid.sql
    # -- converts delegation_events.tenant_id from a legacy TEXT slug to the
    # canonical UUID identity (domain `tenant`, same class as
    # node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql);
    # landed in omnibase_infra first per the node-migration-vendor-parity-gate
    # ordering, ahead of omnimarket#2106.
    assert len(result.declarations) == 106
    assert result.blocked == ()
    assert len(result.legacy_node_declarations) == 2
    assert len(result.cloud_aliases) == 30


def test_completion_gate_is_green_after_domain_classification_is_complete() -> None:
    result = _validate(require_complete=True)

    assert result.blocked == ()


def test_empty_block_set_is_valid_after_all_artifacts_are_classified(
    tmp_path: Path,
) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    (migrations_dir / "nodes" / "node_example" / "0002.sql").unlink()
    (ledger_dir / "application-migration-blocks.tsv").write_text("", encoding="utf-8")

    result = _validate(migrations_dir, ledger_dir, require_complete=True)

    assert result.blocked == ()


@pytest.mark.parametrize(
    ("field_index", "replacement", "signature"),
    [
        (1, "unknown-stream", "unknown migration stream"),
        (3, "unknown_domain", "unknown domain"),
        (5, "0" * 64, "conflicting checksum"),
    ],
)
def test_active_declaration_drift_fails_closed(
    tmp_path: Path,
    field_index: int,
    replacement: str,
    signature: str,
) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    declaration_path = ledger_dir / "application-migrations.tsv"
    fields = declaration_path.read_text(encoding="utf-8").strip().split("\t")
    fields[field_index] = replacement
    declaration_path.write_text("\t".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(validator.ManifestError, match=signature):
        _validate(migrations_dir, ledger_dir)


def test_active_and_blocked_double_declaration_fails_closed(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    active_path = ledger_dir / "application-migrations.tsv"
    active_fields = active_path.read_text(encoding="utf-8").strip().split("\t")
    blocked_path = ledger_dir / "application-migration-blocks.tsv"
    blocked_path.write_text(
        "\t".join(
            (
                active_fields[0],
                active_fields[4],
                active_fields[5],
                "OMN-99999",
                "duplicate declaration RED",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(validator.ManifestError, match="double migration declaration"):
        _validate(migrations_dir, ledger_dir)


def test_duplicate_cloud_alias_fails_closed(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    alias_path = ledger_dir / "cloud-migration-aliases.tsv"
    alias_path.write_text(
        "20260101_example\t20260101_example.sql\n"
        "20260101_example\t20260102_other.sql\n",
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ManifestError, match="duplicate cloud migration alias"
    ):
        _validate(migrations_dir, ledger_dir)


def test_legacy_node_declaration_cannot_shadow_an_active_artifact(
    tmp_path: Path,
) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    (ledger_dir / "legacy-node-migrations.tsv").write_text(
        "\t".join(
            (
                "node:node_example",
                "node:node_example",
                "omninode_internal",
                "node:node_example:0001.sql",
                "hotfix-applied-by-codex",
                "OMN-15717",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ManifestError, match="legacy declaration has vendored artifact"
    ):
        _validate(migrations_dir, ledger_dir)


def test_legacy_node_declaration_requires_a_valid_source_record(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    (ledger_dir / "legacy-node-migrations.tsv").write_text(
        "\t".join(
            (
                "node:node_history",
                "node:node_history",
                "omninode_internal",
                "node:node_history:0001_removed.sql",
                "raw checksum with spaces",
                "OMN-15717",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ManifestError, match="malformed legacy source checksum"
    ):
        _validate(migrations_dir, ledger_dir)


CAPTURED_OMN15717_FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "omn15717"
    / "001_create_review_bot_bypass_log.sql.captured"
)


class TestOmn15717IncidentReplay:
    """Incident replay case (OMN-15547 registry): drives THIS validator
    against the exact captured bytes of node_pr_review_bot's
    001_create_review_bot_bypass_log.sql (git-object:OmniNode-ai/omnimarket@
    cedd24311ed320d34cc5ab5f8f79f5b04e9abf25:src/omnimarket/nodes/
    node_pr_review_bot/migrations/001_create_review_bot_bypass_log.sql),
    vendored with NO manifest declaration -- reproducing the exact pre-fix
    tree shape. This validator existed, unwired, the whole time; had it been
    wired the omission would have failed a PR instead of a live
    bootstrap.sql deploy weeks later ("unknown migration stream/domain:
    adopted node version node:node_pr_review_bot:001_create_review_bot_bypass_log.sql
    has no checked-in declaration", .201:/tmp/refresh_stability_lane_20260805.log).
    """

    def test_captured_undeclared_migration_is_rejected(self, tmp_path: Path) -> None:
        assert CAPTURED_OMN15717_FIXTURE.is_file(), (
            f"incident replay fixture missing: {CAPTURED_OMN15717_FIXTURE}"
        )
        migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
        undeclared_artifact = (
            migrations_dir
            / "nodes"
            / "node_pr_review_bot"
            / "001_create_review_bot_bypass_log.sql"
        )
        undeclared_artifact.parent.mkdir(parents=True)
        undeclared_artifact.write_bytes(CAPTURED_OMN15717_FIXTURE.read_bytes())

        with pytest.raises(
            validator.ManifestError,
            match="migration declaration set differs from the vendored node tree",
        ) as excinfo:
            _validate(migrations_dir, ledger_dir)

        assert "nodes/node_pr_review_bot/001_create_review_bot_bypass_log.sql" in str(
            excinfo.value
        )
