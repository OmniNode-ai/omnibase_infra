# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail CI when changed deployable SQL uses unsafe application targets."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
            "docker/migrations/forward/nodes/node_canary_score_reducer/"
            "0003_capability_scores_tenant_id_to_uuid.sql"
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
        # OMN-16090: node_hook_event_capture is a new node-owned relation whose
        # physical table intentionally remains in `public` until the governed
        # OMN-15359 schema cutover -- same posture as every sibling exemption
        # above (node_canary_score_reducer, node_projection_delegation, et al).
        # 0001 creates the unqualified table + trigger function; 0002 is the
        # split-out tenant-RLS file and, per the same sibling pattern, refers to
        # the table as `public.hook_events` explicitly. Neither file moves or
        # widens application-database authority -- the CREATE's ownership is
        # still proven via the (separate, unconditional) created-object check.
        Path(
            "docker/migrations/forward/nodes/node_hook_event_capture/"
            "0001_create_hook_events.sql"
        ),
        Path(
            "docker/migrations/forward/nodes/node_hook_event_capture/"
            "0002_hook_events_tenant_rls.sql"
        ),
        # OMN-15655 also reconciles historical root migration shapes for
        # fixture parity. These are legacy default-schema repair paths, not new
        # application-database authority, and they must retain compatibility with
        # the existing migration runner until the governed schema cutover lands.
        Path(
            "docker/migrations/forward/031_create_llm_call_metrics_and_cost_aggregates.sql"
        ),
        Path("docker/migrations/forward/050_create_baselines_tables.sql"),
        # OMN-15717: node_pr_review_bot's 001_create_review_bot_bypass_log.sql is
        # a legacy-declared, vendor-synced migration (node-migration-sync,
        # OMN-13332) restored byte-identical to the original omnimarket commit --
        # sha256-verified against the historical file and cross-checked against
        # the live omnidash_analytics.platform_catalog.schema_migrations row
        # this ticket exists to declare. Its unqualified `review_bot_bypass_log`
        # target predates this gate; qualifying it now would break both the
        # byte-identity proof and the checksum this migration's own declaration
        # row binds to. Same node-migration-sync-forced-vendoring rationale as
        # the node_canary_score_reducer / node_projection_registration entries
        # above -- exempting here (not editing the SQL) is the canonical fix.
        Path(
            "docker/migrations/forward/nodes/node_pr_review_bot/"
            "001_create_review_bot_bypass_log.sql"
        ),
        # OMN-15359: 099 performs the governed physical-schema cutover itself --
        # it creates NEW omninode_internal-domain authority
        # (omninode_internal.live_events, ownership declared in omnimarket's
        # application-relation-ownership.yaml via omnimarket#2031). Still
        # exempted after OMN-15838 removed 099's transform-copy/reconciliation
        # block (the original reason this file first needed the exemption -- it
        # read unqualified `live_events` / `public.live_events` as its
        # migration source, which the schema-qualification scanner has no
        # accepting form for): this file's 3 remaining role/grant DO blocks
        # (guarded CREATE ROLE, fail-loud existence check, CONNECT grant) are
        # untouched by OMN-15838 and still trip `_requires_dynamic_sql_rejection`
        # on their own. `changed_sql_paths()` only lints files an actual PR
        # diff touches, so 098/096/094's own DO blocks have never needed an
        # entry here -- but 099 IS touched by both OMN-15359 and this OMN-15838
        # follow-up, so it still needs one. Removing this file from the
        # exemption list is therefore NOT safe post-OMN-15838 -- it would newly
        # fail the lint on the untouched DO blocks, not clear it.
        Path("docker/migrations/forward/099_create_omninode_internal_live_events.sql"),
        # OMN-15846: this file is TOMBSTONED (undeliverable via the k8s Job --
        # cross-DB \connect, no execution path -- see the file's own header
        # and docker/migrations/forward/cross-database-flat-migrations.yaml).
        # The tombstone convention requires the SQL body stay byte-unchanged
        # below the header (migration files are append-only ledgered
        # history), so its pre-existing unqualified `log_entries` target
        # cannot be qualified in place -- the node-owned replacement
        # (docker/migrations/forward/nodes/node_log_persistence_effect/) is
        # what physically creates the schema-qualified
        # omninode_internal.log_entries table now. This entry only exists
        # because OMN-15846 is the first PR to touch this file (a header-only
        # comment addition) since this gate started linting changed files --
        # its own DDL was never in scope before.
        Path("docker/migrations/forward/083_create_log_entries.sql"),
        # OMN-15846: this file is TOMBSTONED (undeliverable via the k8s Job,
        # and unneeded on RDS -- role_omnidash already owns the schema there;
        # see the file's own header). Same byte-unchanged-below-header
        # append-only convention as 083 above -- its `GRANT ... ON ALL TABLES
        # IN SCHEMA public` (unqualified `all` target) and role-DDL DO blocks
        # predate this gate's scope and are not being re-authored. First PR
        # to touch this file (header-only) since this gate started linting
        # changed files.
        Path(
            "docker/migrations/forward/096_grant_role_omnidash_omnidash_analytics.sql"
        ),
        # OMN-15846: the OMN-15376 shape-reconciliation blocks in this NEW
        # node-owned migration (NOT NULL convergence + primary-key guard,
        # added for CodeRabbit finding 3754510314 on omnimarket#2046) use
        # `EXECUTE format(...)` for per-column dynamic SQL -- the exact same
        # idiom `nodes/node_projection_live_events/0000_create_live_events.sql`
        # already uses for its own NOT NULL/PK reconciliation (OMN-15376
        # original). This gate cannot prove a dynamic relation target
        # statically regardless of how well-guarded/idempotent the block is;
        # 099's own exemption entry above establishes this is an accepted,
        # already-precedented limitation for legitimate reconciliation DO
        # blocks in this corpus, not something unique to this file.
        Path(
            "docker/migrations/forward/nodes/node_log_persistence_effect/"
            "0000_create_log_entries.sql"
        ),
        # OMN-15336 item 4 narrows the historical FORCE-RLS posture of the
        # existing node_service_registry table. The table is logically
        # omninode_internal but remains physically in the legacy default schema
        # until the governed schema cutover moves it; this migration only
        # applies NO FORCE ROW LEVEL SECURITY when the existing physical table
        # is present.
        Path(
            "docker/migrations/forward/nodes/node_projection_registration/"
            "0004_node_service_registry_no_force_rls.sql"
        ),
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


_ZERO_REVISION = "0" * 40

# The trusted CI step resolves the diff base from workflow event context:
# pull_request.base.sha, merge_group.base_sha, or push event.before. A pinned
# fallback SHA is forbidden -- a commit reachable only through a since-deleted
# stacked branch is absent from every checkout, so the first push-event run of
# this gate crashed on a raw git fatal instead of a diagnosable verdict
# (OMN-16076). Validate the base up front and fail with the remediation.


def _assert_base_revision_resolvable(repository: Path, base_revision: str) -> None:
    if not base_revision or base_revision == _ZERO_REVISION:
        raise RuntimeError(
            "no usable base revision for changed-SQL linting: got "
            f"{base_revision!r}; the workflow event context must supply "
            "pull_request.base.sha, merge_group.base_sha, or push event.before"
        )
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base_revision}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"base revision {base_revision} is not reachable in this checkout; "
            "it must come from the workflow event context "
            "(pull_request.base.sha, merge_group.base_sha, or push "
            "event.before), never a hardcoded pin"
        )


def changed_sql_paths(
    repository: Path,
    base_revision: str,
    head_revision: str,
) -> tuple[Path, ...]:
    """Return changed deployable SQL, excluding the exact ephemeral proof seed."""
    _assert_base_revision_resolvable(repository, base_revision)
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


# ---------------------------------------------------------------------------
# OMN-15361 frozen baseline ratchet, mirroring the OMN-14443 deploy-gate
# grandfather pattern.
#
# WHY: this gate lints SQL *changed against the PR base*. On a dev PR that is a
# handful of files. At the dev->main promotion boundary the base is main, so the
# whole accumulated migration corpus counts as changed and every latent
# violation in it fires at once -- none of it newly authored, all of it already
# deployed. Rewriting deployed migrations to satisfy a gate at release time is
# the more dangerous path, so pre-existing violations are recorded in a frozen
# snapshot and soft-passed.
#
# SHRINK-ONLY: a violation NOT in the snapshot is held to the full bar and fails
# closed. An entry whose file is gone, or whose violation no longer fires on a
# file this run actually linted, is a STALE entry and FAILS -- the baseline may
# only shrink. Never widen it by hand; regenerate a shrunk one with
# scripts/ci/generate_application_database_sql_baseline.py.
# ---------------------------------------------------------------------------
_BASELINE_PATH = Path(__file__).parent / "application_database_sql_baseline.yaml"


@dataclass(frozen=True)
class SqlGateOutcome:
    """Effective violations plus how many the frozen baseline absorbed."""

    violations: tuple[str, ...]
    grandfathered: int


def violation_key(violation: str) -> str:
    """Content hash of one ``<path>: <message>`` violation line.

    Keyed on content, never on a line number: SQL edits that shift lines must
    not silently re-key an entry and grandfather a violation that is actually
    new.
    """
    return hashlib.sha256(violation.encode("utf-8")).hexdigest()


def load_sql_baseline(baseline_path: Path) -> dict[str, str]:
    """Load the frozen snapshot as ``{violation_key: recorded_relative_path}``.

    A missing or unparseable snapshot fails CLOSED to an EMPTY mapping, which
    holds every violation to the full bar. A broken baseline must never
    silently grandfather anything.

    The path is explicit and has no default: grandfathering is opt-in at the
    call site, so a caller that forgets it gets the full bar rather than a
    silent soft-pass.
    """
    if not baseline_path.exists():
        return {}
    import yaml

    try:
        data = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("violations", [])
    if not isinstance(entries, list):
        return {}
    loaded: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        path = entry.get("path")
        if isinstance(key, str) and isinstance(path, str):
            loaded[key] = path
    return loaded


def _stale_baseline_entries(
    repository: Path,
    baseline: dict[str, str],
    linted_paths: Iterable[str],
    fired_keys: Iterable[str],
) -> tuple[str, ...]:
    """Report baseline entries that no longer describe a live violation.

    Only files this run actually linted can be judged: an entry for a file
    outside the changed set is unobservable, not stale. A deleted file is
    always stale -- it can never fire again.
    """
    linted = set(linted_paths)
    fired = set(fired_keys)
    stale: list[str] = []
    for key, path in baseline.items():
        if not (repository / path).exists():
            stale.append(
                f"{path}: stale baseline entry {key[:12]} -- the recorded file no "
                f"longer exists; drop it from {_BASELINE_PATH.name} "
                "(the baseline is shrink-only)"
            )
        elif path in linted and key not in fired:
            stale.append(
                f"{path}: stale baseline entry {key[:12]} -- the recorded violation "
                f"no longer fires; drop it from {_BASELINE_PATH.name} "
                "(the baseline is shrink-only)"
            )
    return tuple(sorted(stale))


def validate_changed_sql(
    repository: Path,
    base_revision: str,
    head_revision: str,
    *,
    ownership_manifest_paths: Sequence[Path],
    baseline_path: Path | None = None,
) -> SqlGateOutcome:
    """Lint every changed deployable SQL file against typed topology authority.

    ``baseline_path`` is opt-in. Omitting it means no grandfathering at all --
    every observed violation is returned. Production wiring passes
    ``_BASELINE_PATH`` explicitly from ``main``.
    """
    topology = load_topology_profile("local")
    linted_paths: list[str] = []
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
            linted_paths.append(str(relative_path))
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

    observed = tuple(sorted(set(violations)))
    baseline = load_sql_baseline(baseline_path) if baseline_path is not None else {}
    if not baseline:
        return SqlGateOutcome(violations=observed, grandfathered=0)

    fired = {violation_key(violation): violation for violation in observed}
    unbaselined = tuple(
        violation for key, violation in fired.items() if key not in baseline
    )
    stale = _stale_baseline_entries(repository, baseline, linted_paths, fired)
    return SqlGateOutcome(
        violations=tuple(sorted(unbaselined + stale)),
        grandfathered=sum(1 for key in fired if key in baseline),
    )


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
    outcome = validate_changed_sql(
        args.repository,
        args.base_revision,
        args.head_revision,
        ownership_manifest_paths=tuple(
            path if path.is_absolute() else args.repository / path
            for path in args.ownership_manifest
        ),
        baseline_path=_BASELINE_PATH,
    )
    # Announced every run, green or red: a silent grandfather count is how a
    # frozen baseline quietly becomes permanent.
    print(
        f"application_database_sql_gate grandfathered={outcome.grandfathered} "
        f"(frozen baseline {_BASELINE_PATH.name}; shrink-only)"
    )
    if outcome.violations:
        print("application_database_sql_gate=FAIL")
        for violation in outcome.violations:
            print(violation)
        return 1
    print("application_database_sql_gate=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
