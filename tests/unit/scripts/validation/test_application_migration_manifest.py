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
        ledger_dir / "verified-checksum-adoptions.tsv",
        ledger_dir / "verified-divergent-adoptions.tsv",
        ledger_dir / "verified-cross-source-adoptions.tsv",
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
    (ledger_dir / "verified-checksum-adoptions.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "verified-divergent-adoptions.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "verified-cross-source-adoptions.tsv").write_text(
        "", encoding="utf-8"
    )
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
    # +1 for OMN-14751's nodes/node_projection_intent_classification/
    # 0001_intent_classification_agent_source.sql -- adds the nullable
    # agent_source column to intent_classification_events (the projection
    # half of the OMN-14749 parity seam; the producer half, OMN-14750,
    # landed the field on the wire in omnibase_core). A separate forward
    # migration rather than an edit to 0000_create_intent_classification_events.sql
    # because that file's content SHA-256 is already pinned in the ledger.
    # +1 for OMN-16324's
    # node_projection_tenant_credentials/0001_relax_name_provider_not_null.sql
    # -- relaxes name/provider to nullable so a revoke-before-register
    # tombstone row (revoke arriving on a ref this projection has not yet
    # seen a register for) can persist without violating NOT NULL; landed in
    # omnibase_infra first per the node-migration-vendor-parity-gate
    # ordering, ahead of omnimarket#2144.
    # +1 for OMN-15533's
    # node_projection_savings/084_validate_savings_estimates_token_constraints.sql
    # -- validates the token-count CHECK constraints in a separate migration
    # transaction after 082 adds them as NOT VALID.
    # +2 for OMN-16705's additive repair of the append-only violation in
    # OMN-16450 (#2866): node_delegation_routing_reducer/0002_overlay_positive_
    # bound_constraints.sql and node_projection_tenant_credentials/0002_
    # credential_identity_not_null.sql. Their two parents (routing 0001,
    # credentials 0000) were rewritten in place AFTER the .201 dev lane had
    # applied them, so bootstrap.sql raised "conflicting migration checksum in
    # canonical node history" and exited every forward-migration run; both are
    # restored to their applied bytes here and the deltas re-expressed as new
    # ordinals. Vendored into omnibase_infra first per the
    # node-migration-vendor-parity-gate ordering, ahead of the omnimarket PR.
    # +1 for OMN-16759's
    # node_gateway_link_health_write_effect/0001_create_gateway_link_health.sql
    # -- the gateway_link_health projection, re-homed from flat migration 100
    # onto the node loop. The flat loop reaches only omnibase_infra, which has
    # no omninode_internal schema and whose migration role holds no CREATE on
    # the database (both read live from the managed instance), so 100's
    # CREATE SCHEMA failed with "permission denied for database
    # omnibase_infra" and blocked every staging deploy. The node loop connects
    # to the application database, where that schema exists and where the
    # runtime's own DSN points.
    # +1 for OMN-16777's
    # node_projection_consumer_flow/0000_create_consumer_flow_windows.sql --
    # the per-(consumer_group, topic) throughput read model plus its
    # upstream-production tally (Phase 1 of the platform-observability epic
    # OMN-16776). Vendored here first per the node-migration-vendor-parity-gate
    # ordering, ahead of the omnimarket PR that owns the source file.
    # +1 for OMN-16773's additive 0001 reconciliation migration for
    # node_projection_consumer_flow after the guarded-create-table invariant
    # started requiring explicit ADD COLUMN IF NOT EXISTS guards beside 0000.
    # +1 for OMN-16777's
    # node_projection_consumer_flow/0002_reconcile_consumer_flow_window_shapes.sql
    # -- the OMN-16705 new-ordinal successor that authorises correcting 0000 and
    # 0001, both of which spelled their guarded adds `... NOT NULL` and so could
    # not reconcile a drifted table holding rows (Postgres: "contains null
    # values", ON_ERROR_STOP=1, exit 3). The static gate was green while the
    # failure class stayed open; 0002 re-expresses the corrected, nullable
    # reconciliation additively.
    # +1 for OMN-15425's
    # node_projection_delegation_inference_response/0004_grant_tenant_projection_writer.sql
    # -- the tenant-schema/table authorization half of the
    # `tenant_projection_writer` identity cut. It rides the node-owned loop
    # rather than the flat corpus because the flat corpus is one-database and
    # carries no `\connect`, while these grants are per-database inside
    # omnidash_analytics (the role itself is created cluster-wide by the flat
    # migration 102).
    # +1 for OMN-16993's
    # node_projection_session_replay/0002_grant_omninode_runtime_session_replay_snapshots.sql
    # -- the topology-derived grant the three topology instances already
    # declare for `omninode_runtime` on `public.session_replay_snapshots` but
    # which no migration in either repo ever issued, so the projection failed
    # `InsufficientPrivilege` on every write once OMN-16993's LOGIN half let it
    # authenticate at all. Vendored here first per the
    # node-migration-vendor-parity-gate ordering, ahead of the omnimarket PR
    # (#2214) that owns the source file.
    # +1 for OMN-16180's
    # node_projection_work_events/0001_create_work_events.sql -- the CREATE
    # TABLE for omninode_internal.work_events, the L1 work-ledger surface of
    # the OMN-16176 ladder. Vendored here first for the same
    # node-migration-vendor-parity-gate reason as the two entries above: an
    # omnimarket PR touching src/omnimarket/nodes/*/migrations/*.sql cannot
    # land until omnibase_infra@dev already carries a byte-identical copy. The
    # tsv row is what makes that infra-ahead-by-one state legal rather than
    # drift -- sync-node-migrations.sh --check reads it as preserved history
    # via the OMN-15717 legacy-declared exemption. Unlike every prior node
    # relation created in the legacy default schema, this one is created
    # directly in omninode_internal and issues its own omninode_runtime grant,
    # so it needs no OMN-15359 cutover entry and cannot repeat the
    # OMN-16993 grant-missing failure by construction.
    # +1 for OMN-16180's node_projection_work_events/0002_work_events_summary_
    # bound.sql = 121, over the 120 already on dev (which includes
    # node_projection_delegation_inference_response/0004_grant_tenant_projection_
    # writer.sql, vendored and declared by omnibase_infra#3014 for OMN-15425 --
    # NOT by this branch, which carried a now-superseded copy of it before the
    # rebase onto that merge). 0002 exists for two reasons, both recorded in the
    # file: it adds the CHECK that makes ModelWorkEventRow's 2000-char summary
    # bound a property of the DATA rather than of the writer, and it is the
    # successor named by the migration-supersessions.tsv row authorising 0001's
    # comment-only scrub of an internal LAN address (which omnimarket's
    # leaked-literals gate blocks, while its vendor-parity gate requires the
    # source file to be byte-identical to the copy vendored here).
    # +2 for OMN-16930's registry-resolving replacement of the fenced 0031:
    # node_projection_tenant_registry/0000_create_tenant_registry_mirror.sql
    # (the runtime-populated slug->uuid relation, classified omninode_internal
    # and deliberately RLS-free so the migrate identity can read it) and
    # node_projection_delegation/0032_delegation_events_tenant_id_uuid_via_registry.sql
    # (the superseding conversion, which JOINs that relation instead of inlining
    # a literal CASE). 0031's own declaration is untouched -- its bytes are
    # immutable, the .201 dev lane holds its content_sha256, and supersession is
    # recorded in _ledger/migration-supersessions.tsv rather than by editing it.
    # +1 for OMN-17019's node_projection_open_obligations/
    # 0001_create_open_obligations.sql, the C9 open-obligations projection
    # of the OMN-16176 ledger ladder. Like node_projection_work_events above it
    # CREATEs directly in omninode_internal and issues its own omninode_runtime
    # grant, so it needs no OMN-15359 cutover entry and cannot repeat the
    # OMN-16993 grant-missing failure by construction. It is a NET-NEW file --
    # never applied in any lane -- so vendoring it is purely additive and is not
    # the in-place rewrite of an already-applied migration that caused OMN-17139.
    # The grant it issues is deliberately SELECT/INSERT/UPDATE and NOT DELETE:
    # an obligation leaves the open set only via a recorded terminal event.
    # +1 for OMN-17288's node_projection_delegation/
    # 0033_delegation_events_uuid_via_registry_single_transaction.sql, which
    # supersedes 0032. 0032 put DROP POLICY / CREATE POLICY / GRANT after its
    # DO block, and both of its defects follow from that: `RETURN` exits the
    # block and not the file, so the documented "table absent, nothing to
    # convert" path ran those trailing statements anyway and aborted on
    # `relation "delegation_events" does not exist`; and `END$$` COMMITS -- the
    # runner is `psql -v ON_ERROR_STOP=1 -f` with NO --single-transaction -- so
    # the table was committed with RLS enabled and ZERO policies until the
    # standalone CREATE POLICY landed, an interruption in that window denying
    # every application read. 0033 moves the recreate and the app_dashboard
    # GRANT inside the guarded block. Both defects are proven against 0032's
    # real bytes on a scratch Postgres in omnimarket
    # tests/test_omn17288_migration_policy_atomicity.py.
    #
    # 0032's OWN declaration also moves in this change -- its checksum, not its
    # ordinal -- because a prose comment in it quoted a live customer's tenant
    # slug and this repository is PUBLIC (OMN-17288 finding 2). That edit is
    # authorised by the supersession row above, which is the only escape
    # check_migration_append_only.py accepts. It is safe on the facts as well
    # as on the rule: probed read-only 2026-08-31 against
    # platform_catalog.schema_migrations, 0032 is applied on NO lane -- dev
    # holds 0030+0031, stability-test holds 0030, prod and judge carry no
    # platform_catalog ledger at all.
    #
    # OMN-17298 moves this count 125 -> 126 by ADDING one declaration:
    # nodes/node_canary_score_reducer/0004_capability_scores_policy_atomic_restatement.sql.
    # 0003 is the THIRD instance of the 0032 shape -- it drops tenant_isolation
    # inside its DO block (line 127) and recreates it after `END$$` (163-164),
    # so `END$$` commits with the relation enforcing row-level security and
    # zero policies. It is also the FIRST of the three that actually ran:
    # applied on the .201 dev lane 2026-08-17 02:30:59.157734+00. 0003's bytes
    # are NOT edited here (it is applied with a recorded content_sha256, so an
    # in-place edit would raise 'conflicting migration checksum in canonical
    # node history'); 0004 restates the policy and the OMN-14894 GRANT inside
    # one block. It was found by scripts/validation/
    # check_migration_rls_policy_atomicity.py on its first run, not by review.
    #
    # +1 again for OMN-17316's node_projection_delegation/
    # 0034_delegation_events_uuid_via_registry_role_set_guard.sql, which
    # supersedes 0033. 0033 guarded its role switch with
    # `pg_has_role(current_user, v_owner, 'USAGE')` and then performed the
    # switch two statements later via `set_config('role', ...)`. Since
    # PostgreSQL 16 INHERIT and SET are INDEPENDENT membership options, so a
    # membership granted `WITH INHERIT TRUE, SET FALSE` passes that guard and
    # aborts on a bare `permission denied to set role` -- the opaque refusal
    # the guard exists to replace. 0034 tests BOTH predicates before the
    # switch, naming which one failed; 0033's body is carried over verbatim.
    # Proven by execution against the real vendored bytes in
    # tests/integration/migrations/test_omn17316_role_set_membership_guard.py.
    #
    # 0033's own declaration does NOT move: unlike 0032 in the change above, it
    # is retired in place and its bytes are untouched. Probed read-only
    # 2026-08-31 against platform_catalog.schema_migrations in
    # omnidash_analytics on all four .201 lanes, 0033 is applied on none --
    # dev CLEAN (0030, 0031), stability-test CLEAN (0030), prod and judge
    # NO_LEDGER. Neither fact is what admits the change: the append-only gate
    # keys on manifest DECLARATION rather than lane application, and it is
    # satisfied here by 0034 being an ADD next to an untouched 0033. What the
    # lane probe and the gate jointly settled is that the repair could not be
    # an in-place edit to 0033.
    #
    # OMN-15533 moves this count 127 -> 130 by ADDING three declarations under
    # nodes/node_projection_savings/, all domain `tenant` to match 080-084 on
    # the same relation (savings_estimates carries tenant_id + RLS since
    # 080/081):
    #   085_savings_estimates_provenance.sql adds savings_method /
    #     usage_source / pricing_manifest_version -- nullable, no DEFAULT, and
    #     CHECK-constrained NOT VALID to the consumer contract's vocabulary. A
    #     DEFAULT would manufacture a provenance claim the source never made,
    #     so NULL is retained as "the source stated nothing" and the view reads
    #     it back as a refusal.
    #   086_validate_savings_estimates_provenance_constraints.sql VALIDATEs
    #     those constraints outside the definition transaction -- the same
    #     split 084 already makes for the token-count constraints added in 082.
    #   087_savings_views_read_persisted_provenance.sql CREATE OR REPLACEs both
    #     delegation-savings views so they read the persisted columns instead
    #     of inferring a provenance from token counts.
    #
    # 083's declaration does NOT move and its bytes are untouched. This is the
    # same constraint 0032/0033 hit from the other direction: 083 is applied on
    # the .201 dev lane with a recorded content_sha256, so correcting it in
    # place would raise 'conflicting migration checksum in canonical node
    # history'. 087 is therefore a forward CREATE OR REPLACE rather than an
    # edit -- which is sound here precisely because both views hold no data.
    #
    # OMN-17374 moves this count 131 -> 132 by ADDING one declaration,
    # nodes/node_projection_tenant_registry/0001_grant_omninode_runtime_tenant_registry_mirror.sql,
    # domain `omninode_internal` to match 0000 on the same relation. It issues
    # the SELECT/INSERT/UPDATE grant the topology has always declared for
    # `omninode_runtime` on `tenant_registry_mirror` and that no migration ever
    # issued -- the absence that refused BOTH the delegation writer's identity
    # lookup and this node's own INSERT, which is why the mirror sat at zero
    # rows at consumer LAG 0. 0000's bytes are untouched: it is applied on the
    # .201 dev lane with a recorded content_sha256, so repairing its grant
    # block in place would raise 'conflicting migration checksum in canonical
    # node history' -- the same constraint 083/087 hit above. The grant is
    # therefore a forward ADD, which is also where it belongs under the
    # node_projection_session_replay/0002 convention (one grant, in the lineage
    # that owns the relation).
    # +1 again, 130 -> 131, for OMN-17379's node_pr_merged_projection/
    # 0002_grant_omninode_runtime_pr_merged_events.sql, a pure ADD next to an
    # untouched 0001. It grants the topology-declared omninode_runtime writer
    # on public.pr_merged_events AND on the sequence its BIGSERIAL primary key
    # drives -- the half `GRANT INSERT ON TABLE` does not reach, because a
    # BIGSERIAL column is a plain nextval() DEFAULT over a standalone sequence
    # whose own acl Postgres checks on every INSERT. Without it every write
    # failed `InsufficientPrivilege: permission denied for sequence
    # pr_merged_events_projection_cursor_seq` while the runtime swallowed the
    # error and committed the offset anyway, so the projection sat 24 days
    # behind its topic at TOTAL-LAG 0. Proven live on the .201 dev lane
    # 2026-08-31 by rewinding the group to offset 94 and re-consuming 94..96 on
    # the real wired path: three errors, three quarantine records, zero rows.
    # OMN-17440 moves this count 132 -> 140 by ADDING eight declarations, one
    # per owning node lineage, all domain `omninode_internal` to match the
    # migration that creates each relation:
    #
    #   node_contract_registry/0001                (contract_registry)
    #   node_merge_state_projection/0002           (merge_state_transitions)
    #   node_omnigate_projection/0001              (gate_activity, gate_metrics)
    #   node_pr_lifecycle_state_reducer/0002       (pr_lifecycle_ledger_entries)
    #   node_projection_baselines/0003             (breakdown, comparisons,
    #                                               snapshots, trend)
    #   node_projection_intent_classification/0002 (intent_classification_events)
    #   node_projection_overnight/0002             (session_phases, sessions)
    #   node_projection_receipt_gate/0001          (receipt_gate_rows)
    #
    # Thirteen relations in total, each declared for `omninode_runtime` by the
    # generated topology since it was generated and issued by NO migration
    # until now. Every one is a pure forward ADD next to an untouched creating
    # file, for the same reason the OMN-17374 and OMN-17379 rows above are:
    # those creating migrations are applied on the .201 dev lane with recorded
    # content_sha256 values, so repairing a grant block in place would raise
    # 'conflicting migration checksum in canonical node history'.
    #
    # The tranche is the BIGSERIAL/SERIAL-keyed set specifically, because those
    # are the relations where the TABLE grant alone still cannot write -- the
    # INSERT fails at the sequence first, which is precisely the failure
    # OMN-17379 proved on pr_merged_events. The sequence half is OMN-17447.
    # OMN-17447 moves this count 140 -> 150 by ADDING ten more, the SEQUENCE
    # half of the same defect class. A SERIAL/BIGSERIAL key is a nextval()
    # DEFAULT over a STANDALONE sequence whose own acl PostgreSQL checks on
    # every INSERT, so a relation can hold a complete TABLE grant and still
    # refuse every write -- which is what kept pr_merged_events 24 days behind
    # its topic at consumer LAG 0 (OMN-17379).
    #
    # Seven for omninode_runtime (contract_registry, gate_activity,
    # intent_classification_events, merge_state_transitions,
    # overnight_session_phases, pr_lifecycle_ledger_entries, receipt_gate_rows)
    # and three for tenant_projection_writer (capability_scores,
    # delegation_routing_tenant_overlay, dep_health_findings) -- the latter
    # three found by DERIVING the requirement from declared INSERT grants plus
    # the corpus's own column shapes, rather than from OMN-17447's hand list,
    # which was scoped to omninode_runtime and never looked at them.
    #
    # Each resolves its sequence via pg_get_serial_sequence rather than
    # spelling <table>_<column>_seq, RAISEs if the column is not sequence-backed
    # instead of no-oping into another silent half-grant, and asserts
    # has_sequence_privilege post-grant.
    assert len(result.declarations) == 150
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


# ---------------------------------------------------------------------------
# OMN-15857: verified checksum adoptions
# ---------------------------------------------------------------------------
#
# An adoption row tells bootstrap.sql to accept a hand-written sentinel checksum
# for one version, on the strength of a mechanical schema-equivalence proof. The
# row is only trustworthy while every fact it pins is still true, so the
# validator re-checks all of them at PR time rather than at deploy time.

_ADOPTION_RECEIPT = "b" * 64


def _write_adoption(ledger_dir: Path, *fields: str) -> None:
    (ledger_dir / "verified-checksum-adoptions.tsv").write_text(
        "\t".join(fields) + "\n", encoding="utf-8"
    )


def _adoption_fields(ledger_dir: Path, **overrides: str) -> tuple[str, ...]:
    declared = next(
        line.split("\t")
        for line in (ledger_dir / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    row = {
        "version": declared[4],
        "source_checksum": "hotfix-applied-by-codex",
        "manifest_checksum": declared[5],
        "ticket": "OMN-15857",
        "receipt_sha256": _ADOPTION_RECEIPT,
        "verified_at": "2026-08-28",
    }
    row.update(overrides)
    return tuple(row.values())


def test_checked_in_verified_adoptions_are_valid() -> None:
    result = _validate()

    assert len(result.verified_adoptions) >= 1
    for adoption in result.verified_adoptions:
        assert adoption.ticket.startswith("OMN-")
        assert len(adoption.receipt_sha256) == 64


def test_a_valid_verified_adoption_passes(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    _write_adoption(ledger_dir, *_adoption_fields(ledger_dir))

    result = _validate(migrations_dir, ledger_dir)

    assert len(result.verified_adoptions) == 1


def test_verified_adoption_for_an_undeclared_version_is_rejected(
    tmp_path: Path,
) -> None:
    """An adoption can only restate a checksum the manifest already owns."""
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    _write_adoption(
        ledger_dir,
        *_adoption_fields(ledger_dir, version="node:node_ghost:0001_missing.sql"),
    )

    with pytest.raises(
        validator.ManifestError, match="has no active migration declaration"
    ):
        _validate(migrations_dir, ledger_dir)


def test_verified_adoption_pinned_to_stale_content_is_rejected(tmp_path: Path) -> None:
    """The load-bearing check: rewriting the file invalidates the proof.

    Without this, an adoption written against one version of a migration would
    silently keep vouching for whatever bytes replaced it -- reintroducing the
    OMN-16705 class of failure through a new door.
    """
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    _write_adoption(
        ledger_dir, *_adoption_fields(ledger_dir, manifest_checksum="a" * 64)
    )

    with pytest.raises(validator.ManifestError, match="was proven against content"):
        _validate(migrations_dir, ledger_dir)


def test_verified_adoption_of_a_canonical_checksum_is_rejected(tmp_path: Path) -> None:
    """A 64-hex source checksum needs no adoption; bootstrap compares it."""
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    fields = _adoption_fields(ledger_dir)
    _write_adoption(
        ledger_dir, *_adoption_fields(ledger_dir, source_checksum=fields[2])
    )

    with pytest.raises(validator.ManifestError, match="carries a 64-hex source"):
        _validate(migrations_dir, ledger_dir)


def test_verified_adoption_of_the_runner_literal_is_rejected(tmp_path: Path) -> None:
    """``applied-by-runner`` is already adopted; declaring it adds only noise."""
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    _write_adoption(
        ledger_dir, *_adoption_fields(ledger_dir, source_checksum="applied-by-runner")
    )

    with pytest.raises(validator.ManifestError, match="carries the runner literal"):
        _validate(migrations_dir, ledger_dir)


def test_verified_adoption_requires_a_receipt_hash(tmp_path: Path) -> None:
    """The receipt hash is what makes the claim chaseable, so it must be a hash."""
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    _write_adoption(
        ledger_dir, *_adoption_fields(ledger_dir, receipt_sha256="see-the-ticket")
    )

    with pytest.raises(validator.ManifestError, match="malformed receipt sha256"):
        _validate(migrations_dir, ledger_dir)


def test_verified_adoption_requires_a_ticket_and_a_date(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)

    _write_adoption(ledger_dir, *_adoption_fields(ledger_dir, ticket="hotfix"))
    with pytest.raises(validator.ManifestError, match="invalid adoption ticket"):
        _validate(migrations_dir, ledger_dir)

    _write_adoption(ledger_dir, *_adoption_fields(ledger_dir, verified_at="yesterday"))
    with pytest.raises(validator.ManifestError, match="malformed verified_at"):
        _validate(migrations_dir, ledger_dir)


def test_duplicate_verified_adoptions_are_rejected(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    row = "\t".join(_adoption_fields(ledger_dir))
    (ledger_dir / "verified-checksum-adoptions.tsv").write_text(
        row + "\n" + row + "\n", encoding="utf-8"
    )

    with pytest.raises(
        validator.ManifestError, match="duplicate verified checksum adoption"
    ):
        _validate(migrations_dir, ledger_dir)
