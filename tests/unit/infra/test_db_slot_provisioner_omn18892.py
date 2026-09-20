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

import re
import subprocess
from pathlib import Path

import pytest

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
