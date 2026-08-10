# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail CI when changed deployable SQL uses unsafe application targets."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_created_catalog_identities,
    application_database_sql_target_requirements,
    lint_application_database_sql,
    load_application_database_ownership_identities,
)

_NON_DEPLOYABLE_SQL_EXACT_PATHS = frozenset(
    {
        Path("docker/application-domain-enforcement/seed.sql"),
        Path("docker/migrations/forward/_ledger/bootstrap.sql"),
        Path("src/omnibase_infra/migration/cutover/sql/bootstrap.sql"),
    }
)
_NON_DEPLOYABLE_SQL_PREFIXES = (Path("docker/legacy-rds-fixture"),)
_LEGACY_DEFAULT_SCHEMA_SQL_EXACT_PATHS = frozenset(
    {
        # OMN-15503 adds one migration to the existing node_projection_delegation
        # stream. That stream still creates and mutates delegation_events in the
        # legacy default schema; qualifying only this new ALTER would target a
        # table that does not exist in the migration runner.
        Path(
            "docker/migrations/forward/nodes/node_projection_delegation/"
            "0029_delegation_terminal_failure_cause.sql"
        ),
        # OMN-15655 adds the house-tenant tenant_id/RLS tranche for relations
        # whose physical tables intentionally remain in public until the
        # governed OMN-15359 schema cutover moves the full classified set.
        Path(
            "docker/migrations/forward/nodes/node_canary_score_reducer/"
            "0002_capability_scores_tenant_id_and_rls.sql"
        ),
        # OMN-15732: 0003 is the direct sequel to 0002 above -- it continues
        # the OMN-15356 tenant_id TEXT->UUID conversion on the SAME physical
        # public.capability_scores table 0002 already exempted for the
        # identical reason (physical table intentionally remains in public
        # until the governed OMN-15359 schema cutover). OMN-15732 AC2
        # adjudication (2026-08-08) rejected a standalone, schema-qualified
        # mapping function as inadmissible for a node migration stream (no
        # `node:%` domain -- {tenant, omninode_internal} -- admits
        # platform_catalog); the mapping is now inlined into this same file's
        # `ALTER COLUMN ... USING` clause, so there is no separate object to
        # qualify. node-migration-sync (OMN-13332) forces this file to be
        # vendored verbatim from omnimarket dev regardless of this gate, so
        # exempting it here (not editing the SQL) is the canonical fix -- see
        # docs/tracking/ROLLING_WORK_LEDGER.md 2026-08-08
        # [mergesweep-0808-deadlock] for the full deadlock analysis.
        Path(
            "docker/migrations/forward/nodes/node_canary_score_reducer/"
            "0003_capability_scores_tenant_id_to_uuid.sql"
        ),
        # OMN-15732: node_service_registry has been unqualified (default/
        # public schema) since its 0000 CREATE TABLE; 0004 only flips its
        # FORCE ROW LEVEL SECURITY posture (OMN-15336 item 4 / operator
        # ruling R-q, 2026-08-05) inside a DO block that references the same
        # already-unqualified table -- it introduces no new physical-schema
        # authority. Same node-migration-sync-forced-vendoring rationale as
        # above.
        Path(
            "docker/migrations/forward/nodes/node_projection_registration/"
            "0004_node_service_registry_no_force_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_context_roi/"
            "003_context_roi_scores_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_cost_summary/"
            "0002_llm_cost_aggregates_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_delegation/"
            "0030_delegation_budget_state_house_tenant_rekey.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_dep_health/"
            "002_dep_health_findings_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_instruction_eval/"
            "0002_instruction_eval_aggregate_snapshots_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_pattern_learning/"
            "0001_pattern_learning_artifacts_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_routing_decision/"
            "0022_agent_routing_decisions_tenant_id_and_rls.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_projection_skill_executions/"
            "0002_skill_execution_snapshots_tenant_id_and_rls.sql"
        ),
        # OMN-15655 also reconciles historical root migration shapes for
        # fixture parity. These are legacy default-schema repair paths, not new
        # application-database authority, and they must retain compatibility with
        # the existing migration runner until the governed schema cutover lands.
        Path(
            "docker/migrations/forward/031_create_llm_call_metrics_and_cost_aggregates.sql"
        ),
        Path("docker/migrations/forward/050_create_baselines_tables.sql"),
        # OMN-15359: 099 performs the governed physical-schema cutover itself --
        # it creates NEW omninode_internal-domain authority
        # (omninode_internal.live_events, ownership declared in omnimarket's
        # application-relation-ownership.yaml via omnimarket#2031) AND, in the
        # same file, transform-copies from the existing legacy public.live_events
        # table as the one-time migration source. The schema-qualification
        # scanner admits neither form of that reference: unqualified
        # 'live_events' is rejected as "must be schema-qualified", and explicit
        # 'public.live_events' is rejected as "prohibited in public" -- there is
        # no third form this gate accepts for a cross-schema data copy. This is
        # exactly the class of migration the OMN-15420 cutover-journal machinery
        # (omnibase_infra.migration.cutover) is designed for, but that machinery
        # has no runnable CLI yet (P5 scope, OMN-15426/OMN-15360, not started --
        # see docs/migrations/2026-08-06-omninode-internal-schema-transformation-receipt.md's
        # deferred-work list). Exempted here rather than blocked on building
        # that machinery from scratch inside this ticket's slice; 099's own
        # reconciliation logic (count + key-set + row-content-hash parity,
        # fail-closed via RAISE EXCEPTION) is the substitute proof this gate
        # would otherwise provide, verified live against a real ephemeral
        # Postgres cluster in
        # tests/integration/migrations/test_099_omninode_internal_live_events_omn15359.py.
        Path("docker/migrations/forward/099_create_omninode_internal_live_events.sql"),
    }
)


def _is_non_deployable_sql_path(relative_path: Path) -> bool:
    if relative_path in _NON_DEPLOYABLE_SQL_EXACT_PATHS:
        return True
    return any(
        relative_path == prefix or relative_path.is_relative_to(prefix)
        for prefix in _NON_DEPLOYABLE_SQL_PREFIXES
    )


def _is_legacy_default_schema_sql_path(relative_path: Path) -> bool:
    return relative_path in _LEGACY_DEFAULT_SCHEMA_SQL_EXACT_PATHS


def changed_sql_paths(
    repository: Path,
    base_revision: str,
    head_revision: str,
) -> tuple[Path, ...]:
    """Return changed deployable SQL, excluding the exact ephemeral proof seed."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{base_revision}...{head_revision}",
            "--",
            "*.sql",
        ],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot resolve changed SQL range: {detail}")
    relative_paths = tuple(
        Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw
    )
    resolved: list[Path] = []
    repository_root = repository.resolve()
    for relative_path in relative_paths:
        if _is_non_deployable_sql_path(relative_path):
            # These SQL files initialize proof fixtures or internal control
            # ledgers. Their synthetic authority universes are validated by
            # dedicated gates and must not be composed with production owners.
            continue
        candidate = (repository_root / relative_path).resolve()
        if not candidate.is_relative_to(repository_root):
            raise RuntimeError(f"changed SQL path escapes repository: {relative_path}")
        if candidate.is_file():
            resolved.append(candidate)
    return tuple(resolved)


def validate_changed_sql(
    repository: Path,
    base_revision: str,
    head_revision: str,
    *,
    ownership_manifest_paths: Sequence[Path],
) -> tuple[str, ...]:
    """Lint every changed deployable SQL file against typed topology authority."""
    topology = load_topology_profile("local")
    violations: list[str] = []
    try:
        ownership_identities = load_application_database_ownership_identities(
            ownership_manifest_paths
        )
    except ValueError as exc:
        ownership_identities = ()
        violations.append(f"ownership manifests: {exc}")
    declared_identities = {identity.identity for identity in ownership_identities}
    for path in changed_sql_paths(repository, base_revision, head_revision):
        relative_path = path.relative_to(repository.resolve())
        # A legacy-default-schema exemption narrows to (1) the schema-qualification
        # LINT and (2) the target-requirement ownership check for ALTER/DROP/
        # GRANT/etc against relations the exemption's own justification names as
        # already-existing legacy tables (docker/migrations/forward/031 and 050's
        # own `public.legacy_shape` ALTERs are exactly this case -- modifying a
        # pre-existing legacy table is not new authority and was never required
        # to carry an ownership declaration). CREATED-OBJECT ownership validation
        # stays active regardless of the exemption: an exempted file that CREATEs
        # new schema-qualified authority (e.g. 099's `omninode_internal.live_events`)
        # must still carry a declared owner -- narrowing this way (OMN-15359,
        # CodeRabbit) closes the gap the prior blanket `continue` left open, where
        # ownership was unchecked even for an exempted file's newly created objects.
        is_legacy_exempt = _is_legacy_default_schema_sql_path(relative_path)
        sql = path.read_text(encoding="utf-8")
        if not is_legacy_exempt:
            for violation in lint_application_database_sql(sql, topology):
                violations.append(f"{relative_path}: {violation}")
            for requirement in application_database_sql_target_requirements(
                sql, topology
            ):
                location_matches = tuple(
                    identity
                    for identity in ownership_identities
                    if identity.schema == requirement.schema
                    and identity.name == requirement.name
                )
                kind_matches = tuple(
                    identity
                    for identity in location_matches
                    if identity.kind in requirement.allowed_kinds
                )
                if not location_matches:
                    violations.append(
                        f"{relative_path}: application target "
                        f"{requirement.schema}.{requirement.name} requires exactly "
                        "one ownership declaration"
                    )
                elif not kind_matches:
                    violations.append(
                        f"{relative_path}: application target "
                        f"{requirement.schema}.{requirement.name} requires an exact "
                        "object-kind ownership declaration"
                    )
                elif requirement.function_signature is not None and all(
                    identity.function_signature != requirement.function_signature
                    for identity in kind_matches
                ):
                    violations.append(
                        f"{relative_path}: application target "
                        f"{requirement.schema}.{requirement.name}"
                        f"{requirement.function_signature} requires an exact "
                        "routine ownership declaration"
                    )
        for identity in application_database_created_catalog_identities(sql):
            if identity.identity not in declared_identities:
                violations.append(
                    f"{relative_path}: created application object "
                    f"{identity.identity!r} lacks an authoritative ownership declaration"
                )
    return tuple(sorted(set(violations)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", default="HEAD")
    parser.add_argument(
        "--ownership-manifest",
        action="append",
        type=Path,
        required=True,
        help="Typed ownership manifest; repeat for every authoritative source",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    violations = validate_changed_sql(
        args.repository,
        args.base_revision,
        args.head_revision,
        ownership_manifest_paths=tuple(
            path if path.is_absolute() else args.repository / path
            for path in args.ownership_manifest
        ),
    )
    if violations:
        print("application_database_sql_gate=FAIL")
        for violation in violations:
            print(violation)
        return 1
    print("application_database_sql_gate=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
