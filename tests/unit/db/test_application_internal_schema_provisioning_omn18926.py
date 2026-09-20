# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18926 — the corpus must be able to build ``omnidash_analytics`` from EMPTY.

WHAT WAS BROKEN
---------------
``omninode_internal`` is the application database's own schema and more than thirty
node migrations write into it. Nothing DELIVERABLE created it:

* ``098_create_omninode_internal_schema.sql`` is the only flat migration whose
  ``CREATE SCHEMA`` targets that database, and the OMN-15819 ledger declares it
  ``undeliverable`` -- the runner prints UNDELIVERABLE and moves on.
* ``100_create_gateway_link_health.sql`` creates the schema but is a FLAT migration, so
  it runs against ``PGDB`` (``omnibase_infra``), never against ``NODE_PGDB``.
* The node files that mention ``CREATE SCHEMA`` only do so in prose.

A warm lane carries the schema only as applied history from a retired revision of
``nodes/node_projection_registration/0005_create_projection_watermarks.sql``. Measured
2026-09-20 against a genuinely fresh pair: 107 migrations apply, then
``nodes/node_gateway_link_health_write_effect/0001_create_gateway_link_health.sql``
raises ``ERROR: division by zero`` from its own precondition probe and the run exits 3.

WHAT THIS MODULE PINS
---------------------
The fix is a SUPERUSER provisioning seam in ``scripts/run-forward-migrations.sh``, which
is how this corpus already provisions a schema: ``platform_catalog`` is created by the
same runner, as the same identity, from ``_ledger/bootstrap.sql``. The rule the corpus
follows is *the runner provisions SCHEMAS, the corpus provisions the OBJECTS inside
them*, and ``omninode_internal`` was simply never added to the provisioned set.

These tests therefore assert BOTH halves, because either alone is satisfiable by the
wrong change:

1. the runner really does provision the schema, before the node phase, with a readback
   that fails under a NAMED reason rather than letting the corpus die on an unnamed
   division by zero; and
2. the OMN-16759 prohibition -- no ``CREATE SCHEMA`` in any migration -- is UNWEAKENED.
   That gate was written by two production incidents and "fixing" this defect by
   punching a hole in it is the failure mode this module exists to refuse.

Every assertion over a scanned corpus carries a positive control, because a scan that
matches nothing and a scan pointed somewhere wrong are indistinguishable from a green
test (CLAUDE.md operating rule 16).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
CROSS_DB_LEDGER = FORWARD_DIR / "cross-database-flat-migrations.yaml"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FIXTURE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "legacy-rds-fixture-proof.yml"

APPLICATION_SCHEMA = "omninode_internal"
SEAM_FUNCTION = "provision_application_internal_schema"
FIXTURE_GATE = (
    "Sanitized Legacy RDS Fixture Proof / PostgreSQL 16 Fresh + Legacy Fixture"
)

# Anchored at line start so the long rationale headers this repo's migrations carry --
# which quote the forbidden statement on purpose -- are not hits. Same shape as the
# OMN-16759 gate it guards.
_DATABASE_LEVEL_DDL = re.compile(
    r"^\s*(?:CREATE\s+SCHEMA|CREATE\s+DATABASE|ALTER\s+DATABASE)\b",
    re.IGNORECASE,
)
_COMMENT_LINE = re.compile(r"^\s*--")

pytestmark = pytest.mark.unit


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _corpus_files() -> list[Path]:
    """Every forward migration either runner actually executes, both loops."""
    return sorted(FORWARD_DIR.glob("*.sql")) + sorted(FORWARD_DIR.glob("nodes/*/*.sql"))


def _statement_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not _COMMENT_LINE.match(line)
    ]


def _undeliverable_files() -> frozenset[str]:
    """Filenames the OMN-15819 ledger declares as having no execution path.

    Derived from the ledger, never from a list maintained here -- a hardcoded copy
    would keep passing after the ledger changed, which is the drift this ticket's
    AC-2 falsifier names.
    """
    ledger = yaml.safe_load(CROSS_DB_LEDGER.read_text(encoding="utf-8"))
    return frozenset(
        str(entry["file"])
        for entry in ledger["entries"]
        if str(entry.get("disposition")) == "undeliverable"
    )


class TestCorpusScanIsPointedSomewhereReal:
    """Positive controls. Every zero below is only evidence if these pass."""

    def test_the_corpus_scan_finds_migrations(self) -> None:
        files = _corpus_files()
        assert len(files) > 100, (
            f"the corpus scan found {len(files)} files under {FORWARD_DIR}; every "
            "absence assertion in this module is vacuous unless this scan is real"
        )

    def test_the_corpus_contains_the_file_that_surfaced_the_defect(self) -> None:
        probe = (
            FORWARD_DIR
            / "nodes"
            / "node_gateway_link_health_write_effect"
            / "0001_create_gateway_link_health.sql"
        )
        assert probe.is_file(), f"{probe} is missing; this module's premise is gone"

    def test_the_forbidden_ddl_pattern_actually_matches(self) -> None:
        # Without this the prohibition test below would pass against a broken regex.
        assert _DATABASE_LEVEL_DDL.match(
            "CREATE SCHEMA IF NOT EXISTS omninode_internal;"
        )
        assert _DATABASE_LEVEL_DDL.match("  create database foo;")
        assert not _DATABASE_LEVEL_DDL.match(
            "CREATE TABLE omninode_internal.t (a int);"
        )

    def test_the_undeliverable_ledger_parses_and_is_non_empty(self) -> None:
        entries = _undeliverable_files()
        assert entries, (
            f"{CROSS_DB_LEDGER} yielded no undeliverable entries; the AC-2 assertion "
            "below would pass vacuously"
        )


class TestRunnerProvisionsTheApplicationSchema:
    """The delivery path is the runner, as it already is for platform_catalog."""

    def test_the_seam_exists_and_creates_the_schema(self) -> None:
        text = _runner_text()
        assert f"{SEAM_FUNCTION}()" in text, (
            f"{RUNNER} no longer defines {SEAM_FUNCTION}; nothing deliverable creates "
            f"{APPLICATION_SCHEMA} in the application database and a from-empty build "
            "dies on a precondition probe"
        )
        assert "CREATE SCHEMA IF NOT EXISTS ${APPLICATION_INTERNAL_SCHEMA}" in text, (
            f"{SEAM_FUNCTION} no longer issues the CREATE SCHEMA it exists to issue"
        )
        assert f'APPLICATION_INTERNAL_SCHEMA="{APPLICATION_SCHEMA}"' in text

    def test_the_seam_targets_the_node_database_and_runs_before_the_node_phase(
        self,
    ) -> None:
        text = _runner_text()
        call = f'{SEAM_FUNCTION} "$NODE_PGDB"'
        assert call in text, (
            f"{SEAM_FUNCTION} must be called with $NODE_PGDB. The schema belongs to the "
            "APPLICATION database; provisioning it into $PGDB is the OMN-16759 mistake"
        )
        call_at = text.index(call)
        scan_at = text.index("for node-owned migrations in")
        assert call_at < scan_at, (
            "the provisioning seam is called AFTER the node migration scan begins; it "
            "must run before, or the first migration needing the schema still fails"
        )

    def test_the_seam_reads_back_and_fails_with_a_named_reason(self) -> None:
        # The second defect this ticket names: an assertion that CRASHES instead of
        # failing with a reason. The corpus asserts this precondition with
        # `SELECT 1 / count(*)`, so an absent schema surfaces as `division by zero`
        # -- a true failure with a useless reason, 107 migrations after the cause.
        text = _runner_text()
        assert (
            "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = "
            "'${APPLICATION_INTERNAL_SCHEMA}'" in text
        ), (
            "the seam no longer reads the schema back. A CREATE that reported success "
            "without creating anything would be indistinguishable from one that worked"
        )
        seam = text[text.index(f"{SEAM_FUNCTION}()") :]
        seam = seam[: seam.index("\n}\n")]
        assert "OMN-18926" in seam
        assert seam.count("exit 3") >= 2, (
            "both the CREATE failure and the readback failure must exit non-zero; a "
            "seam that warns and continues hands the corpus the same unnamed crash"
        )
        # The seam names the schema through the variable, never a second literal --
        # one spelling, defined once, is what keeps the CREATE, the readback and the
        # failure text from drifting apart.
        assert "${APPLICATION_INTERNAL_SCHEMA}" in seam
        assert APPLICATION_SCHEMA not in seam, (
            "the seam body hardcodes the schema name instead of using "
            "APPLICATION_INTERNAL_SCHEMA; a second spelling can drift from the first"
        )

    def test_the_runner_is_the_only_creator_and_platform_catalog_is_the_precedent(
        self,
    ) -> None:
        bootstrap = (FORWARD_DIR / "_ledger" / "bootstrap.sql").read_text(
            encoding="utf-8"
        )
        assert "CREATE SCHEMA IF NOT EXISTS platform_catalog;" in bootstrap, (
            "the precedent this fix follows is gone: platform_catalog was created by "
            "the runner as superuser, which is why provisioning a schema here is the "
            "corpus's existing shape rather than a new one"
        )


class TestTheOmn16759ProhibitionIsUnweakened:
    """The fix must not reintroduce the privilege the gate exists to keep out."""

    @pytest.mark.parametrize(
        "migration", _corpus_files(), ids=lambda p: str(p.relative_to(FORWARD_DIR))
    )
    def test_no_migration_issues_database_level_ddl(self, migration: Path) -> None:
        undeliverable = _undeliverable_files()
        if migration.name in undeliverable and migration.parent == FORWARD_DIR:
            pytest.skip(
                f"{migration.name} is declared undeliverable by the OMN-15819 ledger; "
                "the runner never executes it"
            )
        offenders = [
            line
            for line in _statement_lines(migration)
            if _DATABASE_LEVEL_DDL.match(line)
        ]
        assert not offenders, (
            f"{migration.relative_to(REPO_ROOT)} issues database-level DDL: {offenders}. "
            "CREATE SCHEMA needs CREATE on the DATABASE, which role_omnidash does not "
            "hold on the managed lane, and IF NOT EXISTS does not help because Postgres "
            "checks the privilege before it checks existence. OMN-18926 was fixed by "
            "provisioning the schema from the runner as superuser, NOT by relaxing this"
        )

    def test_the_fix_added_no_exemption_to_the_undeliverable_ledger(self) -> None:
        # AC-2, derived from the ledger. 098 stays undeliverable: the flat corpus is
        # NOT the delivery path and this change did not try to make it one.
        undeliverable = _undeliverable_files()
        assert "098_create_omninode_internal_schema.sql" in undeliverable, (
            "098 is no longer declared undeliverable. If it was made deliverable to "
            "create the schema, that is the OMN-16759 incident being re-run: it is a "
            "FLAT migration and would issue CREATE SCHEMA against omnibase_infra"
        )


class TestTheFromEmptyProofIsAGate:
    """Detection wired as enforcement (CLAUDE.md operating rule 5)."""

    def test_the_fixture_proof_is_registered_strict(self) -> None:
        gate_source = (REPO_ROOT / "scripts" / "ci" / "ci_summary_gate.py").read_text(
            encoding="utf-8"
        )
        assert f'"{FIXTURE_GATE}"' in gate_source, (
            "the from-empty fixture proof is not in STRICT_GATE_JOBS. This repo "
            "requires exactly one context (CI Summary), so an unregistered job that "
            "is absent or skipped yields SUCCESS -- which is how a corpus that could "
            "not build from empty merged with every gate green"
        )

    def test_ci_calls_the_fixture_proof_unconditionally(self) -> None:
        workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        job = workflow["jobs"]["legacy-rds-fixture-proof"]
        assert job["name"] == "Sanitized Legacy RDS Fixture Proof"
        assert job["uses"] == "./.github/workflows/legacy-rds-fixture-proof.yml"
        # A STRICT gate may never legitimately skip.
        assert "if" not in job
        assert "needs" not in job

    def test_the_fixture_workflow_does_not_self_trigger_on_pull_request(self) -> None:
        called = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML parses the `on:` key as the boolean True (YAML 1.1).
        triggers = called.get(True, called.get("on"))
        assert isinstance(triggers, dict)
        assert "workflow_call" in triggers, (
            "ci.yml cannot call a workflow that does not declare workflow_call"
        )
        assert "pull_request" not in triggers, (
            "a self-trigger would double-run the proof on every PR and re-open the "
            "path-filter blind spot the caller closes"
        )

    def test_the_fixture_proof_still_builds_the_application_database_from_empty(
        self,
    ) -> None:
        # The gate is only worth requiring if it exercises the thing that broke.
        prove = (REPO_ROOT / "docker" / "legacy-rds-fixture" / "prove.sh").read_text(
            encoding="utf-8"
        )
        assert "NODE_POSTGRES_DB=omnidash_analytics" in prove, (
            "the fixture proof no longer points the node loop at omnidash_analytics, "
            "so it no longer covers the database OMN-18926 was about"
        )
        assert "run-forward-migrations.sh" in prove, (
            "the fixture proof no longer invokes the real runner"
        )
