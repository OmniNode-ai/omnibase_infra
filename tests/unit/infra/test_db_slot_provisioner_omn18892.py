# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18892: a pre-PR verify slot gets its own databases AND its own principals.

WHY THIS EXISTS
---------------
A verify slot reuses the dev lane's Postgres SERVER. Suffixing the database
names alone does not isolate it, because **roles in Postgres are cluster-wide
objects**, and both provisioning seams in this repository reset the password of
every role they manage on their existing-role branch, unconditionally:

    docker/migrations/forward/000_create_multiple_databases.sh
        create_role()            -> ALTER ROLE ... WITH LOGIN PASSWORD
        create_login_only_role() -> ALTER ROLE ... WITH LOGIN PASSWORD
    scripts/run-forward-migrations.sh
        reassert_login_only_role_credential()   -> ALTER ROLE ... WITH LOGIN PASSWORD
        reassert_service_role_database_access() -> ALTER ROLE ... WITH LOGIN PASSWORD

A second consumer of the shared server that ran either seam with its own
credentials in the environment would rewrite the DEV LANE's role passwords
cluster-wide. The dev lane's running containers hold the old values in their
environment and would begin failing authentication at their next reconnect.

**Correction to the ticket's premise, measured here.** The ticket names the
INITIALISER as the hazard. The initialiser runs from
``/docker-entrypoint-initdb.d`` only when the data directory is empty, and a
slot never has an empty data directory -- it shares a warm volume. The seam a
slot actually reaches is ``scripts/run-forward-migrations.sh``, which the
``forward-migration`` one-shot runs on **every** compose up. Both carry the
hazard; only the second one is on a slot's path. Both are fenced below.

THE FENCE, AND WHY IT IS TWO INDEPENDENT CONTROLS
-------------------------------------------------
1. **Name fence.** With a slot token set, every object name the provisioner
   touches must carry the slot suffix. A derived name that does not is a
   refusal, not a warning.
2. **Ownership fence.** A pre-existing role is altered only when it is a member
   of the slot's own group role. Suffix-matching alone is refutable by a role
   that merely happens to end in the same characters; group membership answers
   "did this provisioner create it", which is the question that matters.

Neither replaces the other. Control 1 bounds what can be named; control 2 bounds
what can be mutated among the things control 1 admits.

WHY THESE ASSERTIONS RUN WITH NO POSTGRES AND NO DOCKER
--------------------------------------------------------
``--print-scope`` performs the derivation, the grammar check and the fence
classification and opens no connection. That is the whole refusal table, so the
refusal table is testable on any host with bash -- which is where a silent
revert would otherwise survive until a lane rediscovered it against a live
cluster. The connection-refusal half is inherently a live property and is proven
on the lab, recorded in the ticket's evidence rather than asserted here.

Ticket: OMN-18892. Parent epic: OMN-18888 (AC-4). Depends on: OMN-18890.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVISIONER = REPO_ROOT / "scripts" / "provision_db_slot.sh"
BOOTSTRAP = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "000_create_multiple_databases.sh"
)
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

# Named refusal codes. The script declares these as shell constants; the test
# reads the exit status rather than matching prose, so a reworded message does
# not turn a real refusal into a passing test.
EXIT_OK = 0
EXIT_SLOT_TOKEN_MISSING = 2
EXIT_SLOT_TOKEN_MALFORMED = 3
EXIT_DERIVED_NAME_OUT_OF_FENCE = 4
EXIT_ROLE_OUTSIDE_SLOT_GROUP = 5
EXIT_REACHES_UNSUFFIXED_DATABASE = 6
EXIT_CANNOT_REACH_OWN_DATABASE = 7
EXIT_IDENTIFIER_TOO_LONG = 8

# The six service databases and two infrastructure databases the initialiser
# declares. Pinned here so a map edited in one seam and not the other is a red
# test rather than a slot that silently provisions a short set.
SERVICE_DATABASES = (
    "omnibase_infra",
    "omniintelligence",
    "omniclaude",
    "omnimemory",
    "omninode_cloud",
    "omnidash_analytics",
)
INFRA_DATABASES = ("infisical_db", "omniweb")
SERVICE_ROLES = (
    "role_omnibase",
    "role_omniintelligence",
    "role_omniclaude",
    "role_omnimemory",
    "role_omninode",
    "role_omnidash",
)
LOGIN_ONLY_ROLES = (
    "omninode_runtime",
    "tenant_projection_writer",
    "chain_canary_reader",
)


def _run_scope(slot: str | None, *extra: str) -> subprocess.CompletedProcess[str]:
    """Invoke the provisioner's connectionless scope mode."""
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    if slot is not None:
        env["ONEX_DB_SLOT"] = slot
    return subprocess.run(
        ["bash", str(PROVISIONER), "--print-scope", *extra],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,  # the exit status IS the assertion
    )


def _scope_values(stdout: str, key: str) -> list[str]:
    return [
        line.split("=", 1)[1]
        for line in stdout.splitlines()
        if line.startswith(f"{key}=")
    ]


class TestProvisionerExists:
    def test_the_provisioner_is_present_and_executable_by_bash(self) -> None:
        assert PROVISIONER.is_file(), f"{PROVISIONER} is missing"
        assert PROVISIONER.read_text(encoding="utf-8").startswith("#!/"), (
            "the provisioner must carry a shebang so it is runnable the same way "
            "the other migration one-shots are"
        )


class TestSlotTokenGrammar:
    """A typo must never provision an unfenced set. Malformed fails closed."""

    def test_an_absent_slot_token_is_refused_with_its_own_code(self) -> None:
        result = _run_scope(None)
        assert result.returncode == EXIT_SLOT_TOKEN_MISSING, result.stderr

    def test_an_empty_slot_token_is_refused_rather_than_treated_as_the_dev_path(
        self,
    ) -> None:
        # The dev path is the initialiser's, and it is reached by NOT running
        # this tool. An empty token here is a mis-set variable, not a request
        # for the unsuffixed set, and provisioning the unsuffixed set is the
        # exact accident the whole ticket exists to prevent.
        result = _run_scope("")
        assert result.returncode == EXIT_SLOT_TOKEN_MISSING, result.stderr

    @pytest.mark.parametrize(
        "token",
        [
            "Prepr1",  # uppercase
            "1prepr",  # leading digit
            "pre-pr",  # hyphen
            "pre pr",  # space
            "pre.pr",  # dot
            "pre;pr",  # statement separator
            'pre"pr',  # identifier quote
            "pre'pr",  # string quote
            "pre$pr",  # dollar quote
            "prepr" * 8,  # over length
            "_prepr",  # leading underscore: reserved for the separator
            "pre_pr",  # underscore: reserved for the separator
        ],
    )
    def test_a_malformed_slot_token_is_refused(self, token: str) -> None:
        result = _run_scope(token)
        assert result.returncode == EXIT_SLOT_TOKEN_MALFORMED, (
            f"token {token!r} was not refused; stdout={result.stdout!r}"
        )

    @pytest.mark.parametrize("token", ["p", "prepr1", "prepr2", "s", "abc123"])
    def test_a_well_formed_slot_token_is_accepted(self, token: str) -> None:
        result = _run_scope(token)
        assert result.returncode == EXIT_OK, result.stderr


class TestDerivedScope:
    """Every derived name carries the suffix, and nothing else is named."""

    def test_every_managed_database_is_derived_with_the_slot_suffix(self) -> None:
        result = _run_scope("prepr1")
        assert result.returncode == EXIT_OK, result.stderr
        derived = _scope_values(result.stdout, "database")
        expected = [f"{db}_prepr1" for db in SERVICE_DATABASES + INFRA_DATABASES]
        assert derived == expected

    def test_every_managed_role_is_derived_with_the_slot_suffix(self) -> None:
        result = _run_scope("prepr1")
        derived = _scope_values(result.stdout, "role")
        expected = [f"{r}_prepr1" for r in SERVICE_ROLES + LOGIN_ONLY_ROLES]
        assert derived == expected

    def test_the_scope_names_a_group_role_that_marks_the_slots_own_objects(
        self,
    ) -> None:
        result = _run_scope("prepr1")
        assert _scope_values(result.stdout, "group_role") == ["onex_slot_prepr1"]

    def test_no_unsuffixed_name_appears_anywhere_in_the_printed_scope(self) -> None:
        result = _run_scope("prepr1")
        # Positive control: an empty scope would satisfy the loop below for the
        # wrong reason. Assert the scope is populated before reading a zero out
        # of it.
        assert len(_scope_values(result.stdout, "database")) == len(
            SERVICE_DATABASES + INFRA_DATABASES
        )
        assert len(_scope_values(result.stdout, "role")) == len(
            SERVICE_ROLES + LOGIN_ONLY_ROLES
        )
        for name in (
            SERVICE_DATABASES + INFRA_DATABASES + SERVICE_ROLES + LOGIN_ONLY_ROLES
        ):
            for line in result.stdout.splitlines():
                if "=" not in line:
                    continue
                value = line.split("=", 1)[1]
                assert value != name, (
                    f"unsuffixed object {name!r} is inside the slot's scope; the "
                    "name fence is not holding"
                )

    def test_two_slots_derive_disjoint_scopes(self) -> None:
        one = _run_scope("prepr1")
        two = _run_scope("prepr2")
        names_one = set(_scope_values(one.stdout, "database")) | set(
            _scope_values(one.stdout, "role")
        )
        names_two = set(_scope_values(two.stdout, "database")) | set(
            _scope_values(two.stdout, "role")
        )
        # Positive control: two empty sets are disjoint. Assert both are
        # populated, and populated identically in size, before reading the
        # disjointness as isolation.
        expected_size = len(
            SERVICE_DATABASES + INFRA_DATABASES + SERVICE_ROLES + LOGIN_ONLY_ROLES
        )
        assert len(names_one) == expected_size
        assert len(names_two) == expected_size
        assert names_one.isdisjoint(names_two)

    def test_a_derived_identifier_over_the_postgres_limit_is_refused(self) -> None:
        # Postgres truncates an identifier over 63 bytes SILENTLY. A truncated
        # name can collide with another slot's, which is isolation failing with
        # no error at all -- so length is a refusal, never a truncation.
        result = _run_scope("abcdefghijkl", "--check-identifier", "a" * 60)
        assert result.returncode == EXIT_IDENTIFIER_TOO_LONG, result.stderr


class TestDirectiveDatabasesAreProvisionedForTheSlot:
    """A database an ``onex-create-database`` directive names exists before the runner asks for it.

    The runner asks the slot's migration principal, which is NOCREATEDB, to create the
    directive's database, and the first live slot bring-up died on 042_create_keycloak_db.sql
    with "permission denied to create database" (OMN-19064). The provisioner holds the admin
    connection, so it creates the slot-suffixed database itself.
    """

    @staticmethod
    def _scope_over(migrations_dir: Path) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "ONEX_DB_SLOT": "prepr1",
            "MIGRATIONS_DIR": str(migrations_dir),
        }
        return subprocess.run(
            ["bash", str(PROVISIONER), "--print-scope"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )

    def test_the_keycloak_directive_database_is_in_the_slot_scope(self) -> None:
        result = _run_scope("prepr1")
        assert result.returncode == EXIT_OK, result.stderr
        assert "keycloak_prepr1" in _scope_values(result.stdout, "directive_database")

    def test_a_database_the_maps_already_derive_is_not_listed_twice(self) -> None:
        # omnidash_analytics carries the directive and is also in the service map.
        result = _run_scope("prepr1")
        directive = _scope_values(result.stdout, "directive_database")
        managed = _scope_values(result.stdout, "database")
        assert "omnidash_analytics_prepr1" in managed
        assert directive, "positive control: the directive scan found nothing"
        assert set(directive).isdisjoint(managed)

    def test_the_list_is_read_from_the_corpus_not_named_in_the_script(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "001_a.sql").write_text(
            "-- onex-create-database: widgetdb\nSELECT 1;\n", encoding="utf-8"
        )
        result = self._scope_over(tmp_path)
        assert result.returncode == EXIT_OK, result.stderr
        assert _scope_values(result.stdout, "directive_database") == ["widgetdb_prepr1"]

    def test_an_empty_corpus_is_a_broken_scan_and_is_refused(
        self, tmp_path: Path
    ) -> None:
        result = self._scope_over(tmp_path)
        assert result.returncode == EXIT_DERIVED_NAME_OUT_OF_FENCE, result.stderr

    def test_apply_creates_each_directive_database_and_closes_it_to_public(
        self,
    ) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        apply_section = text.split("directive-created databases (OMN-19064)", 1)[1]
        apply_section = apply_section.split("grants, slot's own set only", 1)[0]
        assert "for _db in $SLOT_DIRECTIVE_DATABASES" in apply_section
        assert "CREATE DATABASE" in apply_section
        assert "REVOKE CONNECT ON DATABASE" in apply_section


class TestTenantProjectionWriterReachesTheSlotOmnidashDatabase:
    """Migration 103 needs the CONNECT grant and says the provisioning seam owns it.

    Live slot bring-up (OMN-19064): 103 refused because role_omnibase_prepr1
    holds no grant option on omnidash_analytics_prepr1, so the slot provisioner
    grants CONNECT to the slot's own tenant_projection_writer.
    """

    def test_apply_grants_connect_on_the_slot_omnidash_database_to_the_slot_writer(
        self,
    ) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert 'derive omnidash_analytics)"' in text
        assert 'derive tenant_projection_writer)"' in text
        assert (
            'GRANT CONNECT ON DATABASE \\"${_tpw_db}\\" TO \\"${_tpw_role}\\";' in text
        )

    def test_both_names_are_inside_the_slot_fence(self) -> None:
        result = _run_scope("prepr1")
        assert "omnidash_analytics_prepr1" in _scope_values(result.stdout, "database")
        assert "tenant_projection_writer_prepr1" in _scope_values(result.stdout, "role")


class TestRuntimePrincipalReachesTheSlotOmnidashDatabase:
    """The slot runtime opens its savings pool as omninode_runtime_<slot> (OMN-17427).

    Live bring-up of pre-PR slot 2 (2026-10-19): the runtime died 5/5 boots on
    "HandlerSavingsCorrelation requires constructor parameters ['pool']" because
    its OMNINODE_INTERNAL_DB_URL pool was refused with "permission denied for
    database omnidash_analytics_prepr2". The only flat migration that grants
    omninode_runtime CONNECT is 099, which a slot skips as undeliverable, so the
    slot provisioner owns that grant.
    """

    def test_the_only_flat_runtime_connect_grant_is_undeliverable_under_a_slot(
        self,
    ) -> None:
        forward = REPO_ROOT / "docker" / "migrations" / "forward"
        grant = re.compile(
            r"GRANT\s+CONNECT\s+ON\s+DATABASE\s+omnidash_analytics\s+TO\s+omninode_runtime\b"
        )
        granting = sorted(
            path.name
            for path in forward.glob("*.sql")
            if grant.search(path.read_text(encoding="utf-8"))
        )
        manifest = yaml.safe_load(
            (forward / "cross-database-flat-migrations.yaml").read_text(
                encoding="utf-8"
            )
        )
        undeliverable = {
            entry["file"]
            for entry in manifest["entries"]
            if entry.get("disposition") == "undeliverable"
        }
        assert granting, "no flat migration grants omninode_runtime CONNECT"
        assert set(granting) <= undeliverable, (
            f"{granting} now deliver the runtime CONNECT grant under a slot; "
            "the provisioner grant may be redundant"
        )

    def test_apply_grants_connect_to_the_slot_runtime_and_reads_it_back(
        self,
    ) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert 'derive omninode_runtime)"' in text
        assert (
            'GRANT CONNECT ON DATABASE \\"${_tpw_db}\\" TO \\"${_runtime_role}\\";'
            in text
        )
        assert (
            "SELECT has_database_privilege('${_runtime_role}', '${_tpw_db}', 'CONNECT')"
            in text
        )

    def test_the_runtime_principal_is_inside_the_slot_fence(self) -> None:
        result = _run_scope("prepr1")
        assert "omninode_runtime_prepr1" in _scope_values(result.stdout, "role")


class TestOmnidashPrincipalCanCreateTheLedgerSchema:
    """The ledger convergence opens with CREATE SCHEMA, which needs CREATE on the database (OMN-19064)."""

    def test_apply_grants_create_on_the_slot_omnidash_database_and_reads_it_back(
        self,
    ) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert 'derive role_omnidash)"' in text
        assert (
            'GRANT CREATE ON DATABASE \\"${_dash_db}\\" TO \\"${_dash_role}\\";' in text
        )
        assert (
            "has_database_privilege('${_dash_role}', '${_dash_db}', 'CREATE')" in text
        )


class TestIntelligencePrincipalCanCreateItsExtensions:
    """The intelligence corpus opens with CREATE EXTENSION, which needs CREATE on the database.

    Live slot bring-up (2026-10-19, slot 1): the intelligence migration
    stopped with 'permission denied to create extension "pgcrypto"' against the
    fresh omniintelligence_prepr1, before the runtime came up. Every extension
    the corpus names is trusted, and a trusted extension waives SUPERUSER but
    still needs CREATE on the database, which the slot principal did not hold.
    """

    def test_the_intelligence_corpus_creates_extensions(self) -> None:
        corpus = REPO_ROOT / "docker" / "migrations" / "intelligence"
        assert any(
            re.search(r"^\s*CREATE EXTENSION", f.read_text(encoding="utf-8"), re.M)
            for f in corpus.glob("*.sql")
        )

    def test_apply_grants_create_on_the_slot_intelligence_database_and_reads_it_back(
        self,
    ) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert 'derive omniintelligence)"' in text
        assert 'derive role_omniintelligence)"' in text
        assert (
            'GRANT CREATE ON DATABASE \\"${_intel_db}\\" TO \\"${_intel_role}\\";'
            in text
        )
        assert (
            "has_database_privilege('${_intel_role}', '${_intel_db}', 'CREATE')" in text
        )


class TestSeamsAreFenced:
    """Both password-resetting seams refuse an out-of-fence role under a slot."""

    @pytest.mark.parametrize("seam", [BOOTSTRAP, RUNNER])
    def test_the_seam_reads_the_slot_token(self, seam: Path) -> None:
        assert "ONEX_DB_SLOT" in seam.read_text(encoding="utf-8"), (
            f"{seam.name} carries an unconditional ALTER ROLE ... PASSWORD on its "
            "existing-role branch and does not read the slot token, so a slot "
            "running it would reset the dev lane's role passwords cluster-wide"
        )

    @pytest.mark.parametrize("seam", [BOOTSTRAP, RUNNER])
    def test_the_seam_declares_its_refusal_rather_than_warning(
        self, seam: Path
    ) -> None:
        text = seam.read_text(encoding="utf-8")
        assert "slot_fence_refusal" in text, (
            f"{seam.name} must name its refusal so a fence breach is an exit "
            "status rather than a WARNING line in a log nobody reads"
        )


class TestMapsArePinnedAcrossSeams:
    """A database added in one seam and not the other is a silently short slot.

    The provisioner MIRRORS the initialiser's three maps. There is no shared
    source of truth for them -- they are shell arrays in two files that run in
    two different containers -- so the only thing standing between them and
    drift is this comparison. A slot provisioned from a short map fails at
    connect on the one database nobody noticed was missing, and that reads as a
    branch defect rather than as a provisioning gap.
    """

    @staticmethod
    def _bootstrap_map(block_start: str) -> list[str]:
        text = BOOTSTRAP.read_text(encoding="utf-8")
        body = text.split(block_start, 1)[1].split(")", 1)[0]
        # Entries are one-per-line in the two multi-line maps and all on one
        # line in INFRA_DATABASES; tokenise on the quotes rather than on the
        # newlines so one parser reads all three.
        return re.findall(r'"([^"]+)"', body)

    @staticmethod
    def _provisioner_map(name: str) -> list[str]:
        text = PROVISIONER.read_text(encoding="utf-8")
        body = text.split(f'{name}="', 1)[1].split('"', 1)[0]
        return [line.strip() for line in body.splitlines() if line.strip()]

    def test_the_service_database_to_role_map_matches_the_initialiser(self) -> None:
        bootstrap = [
            entry.rsplit(":", 1)[0] for entry in self._bootstrap_map("SERVICE_DB_MAP=(")
        ]
        assert bootstrap, "positive control: the initialiser's map parsed as empty"
        assert self._provisioner_map("SERVICE_DB_MAP") == bootstrap

    def test_the_infrastructure_database_list_matches_the_initialiser(self) -> None:
        bootstrap = self._bootstrap_map("INFRA_DATABASES=(")
        assert bootstrap, "positive control: the initialiser's list parsed as empty"
        assert self._provisioner_map("INFRA_DATABASES") == bootstrap

    def test_the_login_only_role_map_matches_the_initialiser(self) -> None:
        bootstrap = [
            entry.split(":", 1)[0]
            for entry in self._bootstrap_map("LOGIN_ONLY_ROLE_MAP=(")
        ]
        assert bootstrap, "positive control: the initialiser's map parsed as empty"
        assert self._provisioner_map("LOGIN_ONLY_ROLES") == bootstrap


class TestTheInitialiserRefusesUnderASlotToken:
    """The refusal is reached before any statement, so it is testable dry."""

    def test_it_refuses_and_names_the_replacement_tool(self) -> None:
        result = subprocess.run(
            ["bash", str(BOOTSTRAP)],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "ONEX_DB_SLOT": "prepr1",
                "POSTGRES_USER": "postgres",
                "POSTGRES_DB": "postgres",
            },
            timeout=60,
            check=False,  # the exit status IS the assertion
        )
        assert result.returncode == EXIT_SLOT_TOKEN_MALFORMED, result.stdout
        assert "provision_db_slot.sh" in result.stderr
        # It must refuse BEFORE it starts provisioning, not part way through.
        assert "Phase 1" not in result.stdout, (
            "the initialiser began creating databases before refusing"
        )

    def test_with_no_slot_token_it_does_not_refuse_at_the_fence(self) -> None:
        # Positive control for the test above: without the token the script
        # proceeds past the fence and fails later, for its own reasons (no
        # server). If it exited 3 here too, the test above would be asserting
        # nothing.
        result = subprocess.run(
            ["bash", str(BOOTSTRAP)],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "POSTGRES_USER": "postgres",
                "POSTGRES_DB": "postgres",
            },
            timeout=60,
            check=False,  # the exit status IS the assertion
        )
        assert result.returncode != EXIT_SLOT_TOKEN_MALFORMED
        assert "slot_fence_refusal" not in result.stderr


class TestTheRunnerFence:
    """Unset is byte-identical; malformed fails closed before any connection."""

    def test_a_malformed_slot_token_is_refused_before_any_connection(self) -> None:
        result = subprocess.run(
            ["sh", str(RUNNER)],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "ONEX_DB_SLOT": "Prepr-1"},
            timeout=60,
            check=False,  # the exit status IS the assertion
        )
        assert result.returncode == EXIT_SLOT_TOKEN_MALFORMED, result.stdout
        assert "slot_fence_refusal" in result.stderr

    def test_the_fence_is_inert_with_the_token_unset(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        assert 'ONEX_DB_SLOT="${ONEX_DB_SLOT:-}"' in text, (
            "the runner must default the token to empty, so every lane running "
            "today takes exactly the path it takes now"
        )
        assert "SLOT_ACTIVE=0" in text

    def test_every_role_provisioning_seam_is_skipped_under_a_slot(self) -> None:
        # All four seams end in an unconditional ALTER ROLE ... PASSWORD on a
        # cluster-wide literal. Under a slot, none of them may run: role
        # provisioning has exactly one owner, and it is the provisioner.
        text = RUNNER.read_text(encoding="utf-8")
        seams = [
            "login-only role credentials (section 0)",
            "login-only role grants (OMN-18060)",
            "service-role database access (OMN-18438)",
            "corpus-applier database CREATE (OMN-18508)",
        ]
        for seam in seams:
            assert f'slot_skip_role_seam "{seam}"' in text, (
                f"seam {seam!r} is not skipped under a slot; it would reset a "
                "dev-lane role's password cluster-wide"
            )
        # Positive control: the count of skip call sites equals the count of
        # seams, so a seam added later without a guard is a red test rather
        # than a silent regression.
        # The definition is `slot_skip_role_seam() {`, with no space before the
        # parenthesis, so this counts CALL SITES only.
        assert text.count("slot_skip_role_seam ") == len(seams)

    def test_the_directive_database_takes_the_suffix_under_a_slot(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        assert 'database="${database}_${ONEX_DB_SLOT}"' in text, (
            "an `onex-create-database` directive under a slot must create the "
            "SLOT's database; creating the shared one is a cross-fence mutation"
        )


class TestTeardownEnumeratesFromTheCatalog:
    """A map-driven teardown leaks, measured on this tool's first live run.

    A slot acquires objects the maps do not name. ``run-forward-migrations.sh``
    creates one database per ``onex-create-database`` directive, suffixed under a
    slot, and the first live teardown left ``keycloak_p18892`` behind on the
    shared server for exactly that reason. A teardown that misses an object leaks
    it once per slot claim, forever, onto a server every lane shares.

    Enumerating from the catalog is complete by construction. The name fence
    still runs on every row, so widening the SOURCE of the list does not widen
    what may be dropped.
    """

    def test_databases_are_enumerated_from_pg_database(self) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert "FROM pg_database" in text.split("--drop", 1)[-1] or (
            "_live_databases" in text
        ), "teardown must read the live database list, not the declared maps"
        assert "_live_databases" in text

    def test_roles_are_enumerated_by_slot_group_membership(self) -> None:
        text = PROVISIONER.read_text(encoding="utf-8")
        assert "_live_roles" in text
        assert "pg_auth_members" in text, (
            "teardown must select roles by membership of the slot group, which is "
            "what 'this tool created it' means; a name suffix is not ownership"
        )

    def test_the_name_fence_still_runs_on_every_enumerated_row(self) -> None:
        # The fence is what keeps a catalog-wide enumeration safe. Without it,
        # reading the list from pg_database would be a licence to drop anything
        # the query returned.
        text = PROVISIONER.read_text(encoding="utf-8")
        drop_block = text.split('if [ "$MODE" = "--drop" ]; then', 1)[1]
        assert drop_block.count('assert_in_fence "$_db" "database"') == 1
        assert drop_block.count('assert_in_fence "$_role" "role"') == 1


class TestTheDirectiveScanDistinguishesAbsenceFromABrokenScan:
    """A zero directive count and a broken scan are different things.

    The first revision of the PUBLIC-revocation seam failed the whole migration
    run whenever the directive scan returned nothing, reasoning that the real
    corpus carries directives so an empty read means the scan broke. That is
    true of the real corpus and false in general: the OMN-15422 fresh-plus-legacy
    fixture points the runner at a synthetic corpus of one flat and one node
    migration which declares no directive by design, and it went red on a
    perfectly correct input.

    The positive control belongs on the CORPUS, not on the directive count. A
    directory holding no ``.sql`` at all is a scan pointed somewhere wrong; a
    directory holding migrations and no directive is an absence.
    """

    def test_the_fatal_is_conditioned_on_an_empty_corpus_not_an_empty_scan(
        self,
    ) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        seam = text.split("BEGIN directive-created database PUBLIC revocation seam", 1)[
            1
        ].split("END directive-created database PUBLIC revocation seam", 1)[0]
        assert 'if [ -z "$directive_corpus_size" ]; then' in seam, (
            "the run may only fail when the corpus itself is empty"
        )
        assert "no .sql files under" in seam

    def test_a_corpus_with_no_directive_is_reported_and_continues(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        seam = text.split("BEGIN directive-created database PUBLIC revocation seam", 1)[
            1
        ].split("END directive-created database PUBLIC revocation seam", 1)[0]
        assert 'if [ -z "$directive_databases" ]; then' in seam
        # It must NAME the absence rather than pass silently: a seam that
        # revoked nothing and said nothing is indistinguishable from one that
        # never ran, which is the whole reason the readbacks exist.
        assert "this corpus declares no onex-create-database directive" in seam
        no_directive_branch = seam.split('if [ -z "$directive_databases" ]; then', 1)[
            1
        ].split("fi", 1)[0]
        assert "exit" not in no_directive_branch


INTELLIGENCE_RUNNER = REPO_ROOT / "scripts" / "run-intelligence-migrations.sh"
OVERLAY = REPO_ROOT / "docker" / "docker-compose.prepr.yml"


@pytest.mark.parametrize("runner", [RUNNER, INTELLIGENCE_RUNNER])
def test_slot_runner_refuses_shared_principal_before_connecting(
    runner: Path, tmp_path: Path
) -> None:
    log = tmp_path / "connections"
    stub = tmp_path / "psql"
    stub.write_text('#!/bin/sh\nprintf "connected\\n" >> "$PROBE_LOG"\nexit 23\n')
    stub.chmod(0o700)
    proc = subprocess.run(
        ["sh", str(runner)],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "ONEX_DB_SLOT": "prepr1",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "test-only",
            "PG_WAIT_RETRIES": "1",
            "MIGRATION_LOCK_WAIT_SECONDS": "1",
            "MIGRATIONS_DIR": str(REPO_ROOT / "docker" / "migrations" / "forward"),
            "NODE_POSTGRES_DB": "omnidash_analytics",
            "PROBE_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "slot_fence_refusal" in proc.stderr
    assert not log.exists(), "the unsafe identity reached PostgreSQL"


def _compose_mapping(node: yaml.Node) -> dict[str, yaml.Node]:
    assert isinstance(node, yaml.MappingNode)
    return {key.value: value for key, value in node.value}


@pytest.mark.parametrize(
    ("service", "role", "password"),
    [
        ("forward-migration", "role_omnibase", "ROLE_OMNIBASE_PASSWORD"),
        (
            "intelligence-migration",
            "role_omniintelligence",
            "ROLE_OMNIINTELLIGENCE_PASSWORD",
        ),
    ],
)
def test_slot_overlay_binds_migrations_to_existing_slot_principals(
    service: str, role: str, password: str
) -> None:
    # Compose nodes retain the override tags without constructing objects.
    root = yaml.compose(OVERLAY.read_text(), Loader=yaml.SafeLoader)
    assert root is not None
    services = _compose_mapping(_compose_mapping(root)["services"])
    env = _compose_mapping(_compose_mapping(services[service])["environment"])
    assert env["POSTGRES_USER"].value.startswith(f"{role}_${{ONEX_DB_SLOT:?")
    assert env["POSTGRES_PASSWORD"].value.startswith(f"${{{password}:?")
    if service == "forward-migration":
        assert env["ROLE_OMNIDASH_PASSWORD"].value.startswith(
            "${ROLE_OMNIDASH_PASSWORD:?"
        )
        assert env["ROLE_OMNINODE_PASSWORD"].value.startswith(
            "${ROLE_OMNINODE_PASSWORD:?"
        )


def _slot_connection_helpers() -> str:
    text = RUNNER.read_text()
    start = text.index("# ---- BEGIN slot migration connections (OMN-18892) ----")
    end = text.index("# ---- END slot migration connections (OMN-18892) ----")
    return text[start:end]


def _run_slot_connection(
    tmp_path: Path,
    database: str,
    *,
    safe: str = "t",
    credential_value: str = "test-only",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    log = tmp_path / "connections"
    stub = tmp_path / "psql"
    stub.write_text(
        """#!/bin/sh
user=""
db=""
previous=""
for arg in "$@"; do
    [ "$previous" = "-U" ] && user="$arg"
    [ "$previous" = "-d" ] && db="$arg"
    previous="$arg"
done
printf '%s %s\n' "$db" "$user" >> "$PROBE_LOG"
if [ "$PGPASSWORD" != "$PROBE_EXPECTED_CREDENTIAL" ]; then
    echo "authentication mismatch" >&2
    exit 23
fi
case "$*" in
    *"rolsuper"*) printf '%s\n' "$PROBE_SAFE" ;;
    *"-f"*) printf 'applied\n' ;;
esac
"""
    )
    stub.chmod(0o700)
    proc = subprocess.run(
        [
            "sh",
            "-c",
            _slot_connection_helpers()
            + '\nslot_verify_migration_identity "$1" && psql -d "$1" -U "$PGUSER" -f candidate.sql',
            "sh",
            database,
        ],
        env={
            "PATH": f"{tmp_path}{os.pathsep}/usr/bin:/bin",
            "SLOT_ACTIVE": "1",
            "ONEX_DB_SLOT": "prepr1",
            "PGUSER": "role_omnibase_prepr1",
            "PGHOST": "postgres",
            "PGPORT": "5432",
            "POSTGRES_PASSWORD": f"{credential_value}-infra"
            if credential_value
            else "",
            "ROLE_OMNIDASH_PASSWORD": f"{credential_value}-analytics"
            if credential_value
            else "",
            "ROLE_OMNINODE_PASSWORD": f"{credential_value}-cloud"
            if credential_value
            else "",
            "PROBE_EXPECTED_CREDENTIAL": f"{credential_value}-"
            + {
                "omnibase_infra_prepr1": "infra",
                "omnidash_analytics_prepr1": "analytics",
                "omninode_cloud_prepr1": "cloud",
            }.get(database, "outside"),
            "PROBE_SAFE": safe,
            "PROBE_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    return proc, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    ("database", "role"),
    [
        ("omnibase_infra_prepr1", "role_omnibase_prepr1"),
        ("omnidash_analytics_prepr1", "role_omnidash_prepr1"),
        ("omninode_cloud_prepr1", "role_omninode_prepr1"),
    ],
)
def test_forward_connections_use_the_database_own_slot_principal(
    tmp_path: Path, database: str, role: str
) -> None:
    proc, calls = _run_slot_connection(tmp_path, database)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "applied", (
        "positive control: the file was not applied"
    )
    assert calls == [f"{database} {role}", f"{database} {role}"]


@pytest.mark.parametrize(
    "database", ["postgres", "omnibase_infra", "omnibase_infra_prepr2"]
)
def test_forward_connections_refuse_out_of_slot_databases(
    tmp_path: Path, database: str
) -> None:
    proc, calls = _run_slot_connection(tmp_path, database)
    assert proc.returncode == 4, proc.stderr
    assert "slot_fence_refusal" in proc.stderr
    assert calls == []


@pytest.mark.parametrize("safe", ["f", "", "unexpected"])
def test_forward_migration_refuses_escalated_or_unreadable_identity(
    tmp_path: Path, safe: str
) -> None:
    proc, calls = _run_slot_connection(tmp_path, "omnibase_infra_prepr1", safe=safe)
    assert proc.returncode == 4, proc.stderr
    assert "slot_fence_refusal" in proc.stderr
    assert len(calls) == 1
    assert "applied" not in proc.stdout


def test_forward_connections_refuse_missing_slot_credential(tmp_path: Path) -> None:
    proc, calls = _run_slot_connection(
        tmp_path, "omnidash_analytics_prepr1", credential_value=""
    )
    assert proc.returncode == 4, proc.stderr
    assert "slot_fence_refusal" in proc.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("active", "expected"), [("0", "omninode_cloud"), ("1", "omninode_cloud_prepr1")]
)
def test_cloud_history_discovery_uses_the_lanes_own_database(
    active: str, expected: str
) -> None:
    text = RUNNER.read_text()
    helper = (
        "import_cloud_history() {"
        + text.split("import_cloud_history() {", 1)[1].split("\n}\n", 1)[0]
        + "\n}\n"
    )
    proc = subprocess.run(
        [
            "sh",
            "-c",
            'validate_database_identifier() { :; }\nslot_fence_assert() { :; }\ndatabase_exists() { printf "%s\\n" "$1" >&2; return 1; }\n'
            + helper
            + "\nimport_cloud_history analytics",
        ],
        env={"PATH": "/usr/bin:/bin", "SLOT_ACTIVE": active, "ONEX_DB_SLOT": "prepr1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip() == expected


@pytest.mark.parametrize("safe", ["t", "f", "", "unexpected"])
def test_intelligence_checks_identity_before_applying_branch_sql(
    tmp_path: Path, safe: str
) -> None:
    log = tmp_path / "connections"
    stub = tmp_path / "psql"
    stub.write_text(
        """#!/bin/sh
user=""
db=""
previous=""
for arg in "$@"; do
    [ "$previous" = "-U" ] && user="$arg"
    [ "$previous" = "-d" ] && db="$arg"
    previous="$arg"
done
printf '%s %s\n' "$db" "$user" >> "$PROBE_LOG"
case "$*" in
    *"rolsuper"*) printf '%s\n' "$PROBE_SAFE" ;;
    *"FROM pg_database WHERE datname"*) echo 1 ;;
    *"-f"*) echo applied >> "$PROBE_LOG" ;;
esac
"""
    )
    stub.chmod(0o700)
    proc = subprocess.run(
        ["sh", str(INTELLIGENCE_RUNNER)],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "ONEX_DB_SLOT": "prepr1",
            "POSTGRES_USER": "role_omniintelligence_prepr1",
            "POSTGRES_PASSWORD": "test-only",
            "PG_WAIT_RETRIES": "1",
            "MIGRATIONS_DIR": str(REPO_ROOT / "docker" / "migrations" / "intelligence"),
            "PROBE_SAFE": safe,
            "PROBE_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    calls = log.read_text().splitlines()
    assert calls, "positive control: the runner did not connect"
    assert set(calls) <= {
        "omniintelligence_prepr1 role_omniintelligence_prepr1",
        "applied",
    }
    if safe == "t":
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "applied" in calls
    else:
        assert proc.returncode == 4, proc.stdout + proc.stderr
        assert len(calls) == 2
        assert "applied" not in calls
        assert "slot_fence_refusal" in proc.stderr
