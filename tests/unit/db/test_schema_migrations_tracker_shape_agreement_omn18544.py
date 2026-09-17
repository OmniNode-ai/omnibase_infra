# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every declarer of a ``schema_migrations`` tracker must agree, per database (OMN-18544).

## The defect this gate exists to refuse

``docker/migrations/cloud/run-cloud-migrations.sh`` bootstrapped
``public.schema_migrations`` in ``omninode_cloud`` as
``(migration_name text PRIMARY KEY, applied_at timestamptz)`` **before** the
manifest loop, so it always won the name. The corpus's own tracking migration,
``omninode_infra`` ``db/migrations/00000000_migrations_tracking.sql``, sorts
first in the MANIFEST and therefore ran second in wall-clock order, found a
table of that name already present, and its ``CREATE TABLE IF NOT EXISTS`` was a
silent no-op. The shape the corpus declares -- ``(version, applied_at,
checksum)`` -- never landed, and the one corpus file that self-registers,
``021_workflow_results.sql:135``, died on ``column "version" of relation
"schema_migrations" does not exist``.

This is OMN-4627 recurring in the opposite direction. That earlier instance was
the same mechanism with the two sides swapped: the SQL file declared one shape,
the k8s Job runner had already bootstrapped another, the idempotent CREATE
no-opped, and the next statement hard-failed. OMN-4627 was closed by reconciling
the two literals by hand. Reconciling literals by hand is exactly what does not
survive: the compose runner was written *after* that reconciliation
(OMN-17530) and reintroduced the divergence, because nothing mechanical
connected the two declarations.

## What this module asserts, and why it is shaped this way

The invariant is per DATABASE, not per table name. Three different tables named
``public.schema_migrations`` live on the same host, and only one of them is in
conflict. ``omnibase_infra`` keys on ``migration_id`` and is declared by this
repo's own corpus and its two runners. ``omniintelligence`` keys on ``id`` and is
declared by this repo's intelligence runner. ``omninode_cloud`` keys on
``version`` and is declared by the ``omninode_infra`` corpus -- and by nothing
here, once the defect above is removed.

A gate that flagged every same-named table would flag ``omnibase_infra``'s
tracker, which is correct as it stands and is a different database with its own
runner. So ``omnibase_infra`` is carried here deliberately as the POSITIVE
CONTROL: it has three independent declarers in this repo -- the corpus file
``docker/migrations/forward/036_create_schema_migrations.sql`` and the two
runners ``scripts/run-migrations.py`` and ``scripts/run-forward-migrations.sh``
-- and they must all parse to the same column set and pass in the same run that
``omninode_cloud`` is checked in. A run where every database is flagged, and a
run where none is, are then distinguishable from each other.

``omninode_cloud`` is handled by a different rule, and the difference is not a
weakening. Its corpus lives in ``omninode_infra``, a separate repository, so a
test in this repo cannot parse that side at source and any comparison would have
to restate the shape here -- which is the second independent literal the defect
is made of. The rule that IS locally decidable, and is strictly stronger than a
comparison, is that this repo declares **no** shape for that database at all:
the runner applies the corpus's own tracking file as its bootstrap and derives
its bookkeeping key from the live primary key, so there is exactly one
declaration of that table in existence and nothing to drift against. A
reintroduced ``CREATE TABLE ... schema_migrations`` in the runner is refused by
:func:`test_a_corpus_owned_database_has_no_local_shape_declaration` whatever
columns it names -- including a copy that happens to be correct today, because a
correct copy is how OMN-4627 came back.

:func:`test_the_agreement_check_flags_a_reintroduced_divergence` is the
falsifier the acceptance criterion asks for: it feeds the real parser the exact
pre-fix runner text and asserts the comparison reports a conflict. Without it,
a parser that silently matched nothing would pass every other assertion here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

TRACKER_TABLE = "schema_migrations"

#: Which files may speak about the tracker of which database. This registry
#: names LOCATIONS only -- no column, type or key is written down here, because
#: restating a shape in the test is the same defect the test exists to refuse.
#: Every shape below is parsed out of the file at the path given.
TRACKER_DECLARERS: dict[str, tuple[str, ...]] = {
    "omnibase_infra": (
        "docker/migrations/forward/036_create_schema_migrations.sql",
        "scripts/run-migrations.py",
        "scripts/run-forward-migrations.sh",
    ),
    "omniintelligence": ("scripts/run-intelligence-migrations.sh",),
    "omninode_cloud": ("docker/migrations/cloud/run-cloud-migrations.sh",),
}

#: Databases whose tracker shape is owned by a corpus in ANOTHER repository.
#: This repo may not declare their shape at all -- see the module docstring.
CORPUS_OWNED_ELSEWHERE: frozenset[str] = frozenset({"omninode_cloud"})

#: Files that declare a tracker this registry deliberately does NOT govern, each
#: with the reason. A path earns a place here by argument, never by being
#: inconvenient, and the completeness check below fails on any declarer that is
#: in neither this mapping nor TRACKER_DECLARERS -- so the registry cannot fall
#: quietly behind the tree the way a hand-reconciled pair of literals did.
UNGOVERNED_DECLARERS: dict[str, str] = {
    "docker/migrations/forward/_ledger/bootstrap.sql": (
        "platform_catalog.schema_migrations in omnidash_analytics -- a different "
        "schema in a different database, with its own migration-stream ledger "
        "shape and its own immutability gate"
    ),
    "docker/legacy-rds-fixture/legacy-seed.sql": (
        "a fixture that deliberately reproduces RETIRED shapes, including the "
        "pre-OMN-17537 nullable-checksum one, so migrations can be proven against "
        "them; agreeing with the corpus would defeat its purpose"
    ),
    "tests/integration/migrations/test_application_migration_ledger_omn15413.py": (
        "builds a throwaway ledger in a test database, including a deliberately "
        "extra-columned variant, to prove the verifier notices"
    ),
    "tests/integration/migrations/test_verified_cross_source_adoption_omn16919.py": (
        "builds a throwaway ledger in a test database"
    ),
    "tests/integration/migrations/test_divergent_checksum_verifier_omn16915.py": (
        "builds a throwaway ledger in a test database"
    ),
    "tests/integration/migrations/test_node_migration_discovery_applies.py": (
        "builds a throwaway ledger in a test database"
    ),
    "tests/scripts/test_forward_migration_advisory_lock.py": (
        "builds a throwaway ledger in a test database"
    ),
    "tests/fixtures/omn15547/legacy-rds-fixture-prove.sh.captured": (
        "a captured transcript of the legacy fixture harness, carrying the same "
        "retired shapes it recorded"
    ),
    "tests/unit/db/test_schema_migrations_tracker_shape_agreement_omn18544.py": (
        "this module, which holds the verbatim pre-fix bootstrap as the input to "
        "its own falsifier -- see PRE_FIX_RUNNER_BOOTSTRAP"
    ),
    "docker/legacy-rds-fixture/prove.sh": (
        "the harness asserting against that fixture, carrying the same retired "
        "shapes as its expectations"
    ),
}

#: The compose runner's bootstrap, once it stops declaring a shape of its own.
CORPUS_TRACKING_FILE = "00000000_migrations_tracking.sql"

COMPOSE_RUNNER = (
    REPO_ROOT / "docker" / "migrations" / "cloud" / "run-cloud-migrations.sh"
)

#: The exact pre-fix bootstrap, kept verbatim so the falsifier exercises the
#: real defect rather than a paraphrase of it. Never a source of truth for any
#: assertion about the current tree -- only an input to the parser.
PRE_FIX_RUNNER_BOOTSTRAP = """
psql_db -c "CREATE TABLE IF NOT EXISTS public.schema_migrations (
              migration_name text PRIMARY KEY,
              applied_at timestamptz NOT NULL DEFAULT now()
            )"
"""

#: The ``\\b`` sits BEFORE the optional closing quote, not after it. After it the
#: boundary is between ``"`` and a space -- both non-word -- so the whole pattern
#: silently stopped matching ``public."schema_migrations" (``, which is the
#: evasion the quote handling was added to catch.
#:
#: ``UNLOGGED``/``TEMP`` and quoted identifiers are matched deliberately: a
#: reintroduced declaration that spells the table ``"schema_migrations"`` or
#: creates it UNLOGGED is the same defect, and a matcher that missed them would
#: be a gate with a documented way around it.
_CREATE_RE = re.compile(
    r"CREATE\s+(?:GLOBAL\s+|LOCAL\s+)?(?:TEMP(?:ORARY)?\s+|UNLOGGED\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:\"?[A-Za-z_][\w$]*\"?\s*\.\s*)?"
    rf"\"?{TRACKER_TABLE}\b\"?\s*\(",
    re.IGNORECASE,
)

#: A tracker can also be brought into existence with no column list at all --
#: ``CREATE TABLE ... AS SELECT`` and ``CREATE TABLE ... OF <type>`` both declare
#: a shape by reference, and ``SELECT ... INTO`` creates a table outright. None
#: has a body for :func:`declared_column_sets` to parse, so they are refused
#: outright in the corpus-owned runner rather than parsed.
_BODYLESS_CREATE_RE = re.compile(
    r"(?:CREATE\s+(?:GLOBAL\s+|LOCAL\s+)?(?:TEMP(?:ORARY)?\s+|UNLOGGED\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?(?:\"?[A-Za-z_][\w$]*\"?\s*\.\s*)?"
    rf"\"?{TRACKER_TABLE}\b\"?\s*(?:AS|OF)\b"
    r"|"
    rf"(?<!INSERT )\bINTO (?:\"?[A-Za-z_][\w$]*\"? ?\. ?)?\"?{TRACKER_TABLE}\b\"?)",
    re.IGNORECASE,
)


#: :data:`_BODYLESS_CREATE_RE` is matched against whitespace-collapsed text. Its
#: ``SELECT ... INTO`` branch is a fixed-width negative lookbehind on ``INSERT ``,
#: which only distinguishes the two forms when the gap between the words is
#: exactly one space -- and the runner's own bookkeeping INSERTs are wrapped
#: across lines.
def _strip_shell_comments(script: str) -> str:
    """Drop whole-line ``#`` comments from a shell script.

    Whole-line only. A trailing ``#`` can be a parameter expansion
    (``${verdict#SKIP }``) rather than a comment, and cutting at one would delete
    live code and turn a presence check into a false negative.
    """
    return "\n".join(
        "" if line.lstrip().startswith("#") else line for line in script.splitlines()
    )


def _collapsed(sql: str) -> str:
    """Comments removed and every whitespace run flattened to one space."""
    return re.sub(r"\s+", " ", _strip_comments(sql))


#: Shaping DDL that is not a CREATE. The convergence step legitimately issues
#: ``ALTER TABLE ... RENAME TO`` to retire the table it once created; anything
#: that ADDs, ALTERs or DROPs a COLUMN, or adds a CONSTRAINT, is this repo
#: declaring a shape by another spelling.
_SHAPING_ALTER_RE = re.compile(
    rf"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?"
    rf"(?:\"?[A-Za-z_][\w$]*\"?\s*\.\s*)?\"?{TRACKER_TABLE}\b\"?[^;]*?"
    r"(?:\b(?:ADD|DROP|ALTER)\b|\bRENAME\s+COLUMN\b)",
    re.IGNORECASE | re.DOTALL,
)

#: Leading words that begin a TABLE constraint rather than a column definition.
_CONSTRAINT_LEADERS = frozenset(
    {"primary", "unique", "check", "foreign", "constraint", "exclude", "like"}
)


def _balanced_body(text: str, open_paren: int) -> str:
    """Return the text between ``open_paren`` and its matching ``)``.

    Depth-counted rather than regex-matched: a column default such as
    ``NOT NULL DEFAULT now()`` carries nested parentheses, and a lazy
    ``\\(.*?\\)`` truncates the body at the first one.
    """
    depth = 0
    for index in range(open_paren, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : index]
    raise AssertionError(
        f"unbalanced parentheses after a CREATE TABLE {TRACKER_TABLE} at offset {open_paren}"
    )


def _strip_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` blocks.

    Order matters: this runs BEFORE the body is comma-split and depth-counted.
    Stripping afterwards -- the shape this module shipped with first -- drops a
    real column whenever a comment contains a comma (``-- the key, unique``
    swallows the next column entirely) and raises on a comment containing an
    unbalanced parenthesis. Neither fires on any registered file today, which is
    exactly why it needed finding rather than waiting for.
    """
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", sql)


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are not inside parentheses."""
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    items.append("".join(current))
    return items


def declared_column_sets(text: str) -> list[tuple[str, ...]]:
    """Parse every ``CREATE TABLE ... schema_migrations`` in ``text``.

    Returns one ordered column tuple per declaration found. Table-level
    constraints are dropped; only column definitions contribute a name.
    """
    found: list[tuple[str, ...]] = []
    text = _strip_comments(text)
    for match in _CREATE_RE.finditer(text):
        body = _balanced_body(text, match.end() - 1)
        columns: list[str] = []
        for item in _split_top_level(body):
            stripped = item.strip()
            if not stripped:
                continue
            first = stripped.split()[0].strip('"').lower()
            if first in _CONSTRAINT_LEADERS:
                continue
            # A quoted-empty or punctuation-only token is not a column. Appending
            # it would make two unrelated shapes compare equal on a pair of empty
            # strings, so agreement and divergence would both stop meaning
            # anything -- measured on two files in this tree.
            if not re.fullmatch(r"[a-z_][\w$]*", first):
                continue
            columns.append(first)
        found.append(tuple(columns))
    return found


def _read(relative: str) -> str:
    path = REPO_ROOT / relative
    assert path.is_file(), (
        f"{relative} is named in TRACKER_DECLARERS but does not exist"
    )
    return path.read_text(encoding="utf-8")


def _shapes_for(database: str) -> dict[str, tuple[str, ...]]:
    """Map each declaring file of ``database`` to the single shape it declares."""
    shapes: dict[str, tuple[str, ...]] = {}
    for relative in TRACKER_DECLARERS[database]:
        for columns in declared_column_sets(_read(relative)):
            # A file that declares the tracker twice is itself a divergence
            # risk; key on file+ordinal so both are compared, not collapsed.
            key = relative if relative not in shapes else f"{relative}#{len(shapes)}"
            shapes[key] = columns
    return shapes


@pytest.mark.unit
@pytest.mark.parametrize(
    "database", sorted(set(TRACKER_DECLARERS) - CORPUS_OWNED_ELSEWHERE)
)
def test_every_local_declarer_of_a_database_agrees_on_the_column_set(
    database: str,
) -> None:
    """All in-repo declarers of one database's tracker declare one column set.

    ``omnibase_infra`` is the positive control described in the module
    docstring: three independent declarers, which must agree with each other
    and which must NOT be compared against any other database's tracker.
    """
    shapes = _shapes_for(database)
    assert shapes, (
        f"no CREATE TABLE {TRACKER_TABLE} parsed for {database} from "
        f"{TRACKER_DECLARERS[database]} -- the registry points at files that no "
        "longer declare the tracker, so this check is asserting nothing"
    )
    distinct = set(shapes.values())
    assert len(distinct) == 1, (
        f"{database}'s {TRACKER_TABLE} is declared with more than one column set. "
        "Every declarer of one database's tracker must agree:\n"
        + "\n".join(f"  {where}: {cols}" for where, cols in sorted(shapes.items()))
    )


@pytest.mark.unit
def test_the_positive_control_database_is_scoped_separately_from_the_others() -> None:
    """Per-database scoping is real: the trackers genuinely differ across databases.

    If every database happened to share a column set, the parametrised check
    above would pass even when implemented as a single global comparison, and a
    reader could not tell the two apart. Asserting the sets differ is what makes
    "agreement is checked per database" a claim with content.
    """
    per_database = {
        database: set(_shapes_for(database).values())
        for database in sorted(set(TRACKER_DECLARERS) - CORPUS_OWNED_ELSEWHERE)
    }
    collapsed = {
        database: next(iter(shapes))
        for database, shapes in per_database.items()
        if shapes
    }
    assert len(collapsed) >= 2, (
        "fewer than two locally-declared databases remain, so per-database "
        f"scoping cannot be demonstrated: {collapsed}"
    )
    assert len(set(collapsed.values())) == len(collapsed), (
        "two databases declare identical tracker shapes, so this run cannot "
        f"distinguish per-database agreement from a global one: {collapsed}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("database", sorted(CORPUS_OWNED_ELSEWHERE))
def test_a_corpus_owned_database_has_no_local_shape_declaration(database: str) -> None:
    """This repo declares no tracker shape for a database whose corpus owns it.

    Stronger than a comparison, and locally decidable: with zero declarations
    here there is exactly one in existence, so there is no second literal for a
    future edit to drift away from. A reintroduced ``CREATE TABLE`` is refused
    even if its columns are correct today -- a correct copy is precisely how
    OMN-4627 came back as OMN-18544.
    """
    offenders = {
        relative: declared_column_sets(_read(relative))
        for relative in TRACKER_DECLARERS[database]
        if declared_column_sets(_read(relative))
    }
    assert not offenders, (
        f"{database}'s {TRACKER_TABLE} shape is owned by the migration corpus in "
        "another repository, but this repo declares it too:\n"
        + "\n".join(f"  {where}: {cols}" for where, cols in sorted(offenders.items()))
        + f"\nThe runner must apply the corpus's own {CORPUS_TRACKING_FILE} instead "
        "of bootstrapping a table of its own."
    )


@pytest.mark.unit
def test_the_agreement_check_flags_a_reintroduced_divergence() -> None:
    """Falsifier: the real parser must report the pre-fix runner as a divergence.

    Feeds :func:`declared_column_sets` the verbatim pre-fix bootstrap. If the
    parser matched nothing -- a broken regex, a renamed table, a changed quoting
    style -- every other assertion in this module would pass vacuously and this
    one is the only thing that says so.
    """
    parsed = declared_column_sets(PRE_FIX_RUNNER_BOOTSTRAP)
    assert parsed == [("migration_name", "applied_at")], (
        "the parser no longer recognises the pre-fix compose bootstrap, so a "
        f"reintroduced divergence would go unreported: parsed {parsed!r}"
    )
    # And with that declaration present, the corpus-owned rule reports it.
    assert declared_column_sets(
        COMPOSE_RUNNER.read_text(encoding="utf-8") + PRE_FIX_RUNNER_BOOTSTRAP
    ), "a runner carrying the pre-fix bootstrap must parse as declaring a shape"


@pytest.mark.unit
def test_the_compose_runner_bootstraps_from_the_corpus_tracking_file() -> None:
    """The runner's bootstrap is the corpus's own tracking migration, fail-closed.

    Naming the file is not enough on its own -- an absent one must abort with a
    named error rather than fall through to a loop whose first probe reads a
    table that does not exist.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    assert CORPUS_TRACKING_FILE in text, (
        f"{COMPOSE_RUNNER.name} does not reference {CORPUS_TRACKING_FILE}; with no "
        "CREATE TABLE of its own it has nothing to bootstrap the tracker from"
    )
    assert re.search(r'psql_db -f "\$\{?TRACKING\}?"', text), (
        f"{COMPOSE_RUNNER.name} names the corpus tracking file but never applies it. "
        "Naming it is not bootstrapping it: with no CREATE TABLE of its own the "
        "tracker would simply not exist and the loop's first probe would die. The "
        "previous form of this assertion was satisfied by a variable named TRACKING"
    )
    assert re.search(r'\[\s+-f\s+"\$\{?TRACKING\}?"\s+\]\s*\|\|', text), (
        f"{COMPOSE_RUNNER.name} must fail closed on an absent tracking file rather "
        "than fall through to a loop that reads a table nothing created"
    )


@pytest.mark.unit
def test_the_compose_runner_derives_its_bookkeeping_key_from_the_live_primary_key() -> (
    None
):
    """The runner reads the tracker's key column instead of naming one.

    A hardcoded key column is the one-column form of the same two-literals
    defect: the corpus could rename it and nothing here would fail until a lane
    aborted at runtime. Reading ``indisprimary`` off the table the corpus just
    created leaves the runner with no opinion about the shape at all.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    assert "indisprimary" in text, (
        f"{COMPOSE_RUNNER.name} does not introspect the tracker's primary key; its "
        "bookkeeping column must be derived, not written down"
    )
    # Scoped to statements that read or write the ledger's CONTENT. The stash
    # probe addresses ``schema_migrations_legacy_omn18544`` by its own literal
    # name, and the retired-column assertion queries ``pg_attribute`` rather
    # than the tracker -- neither is bookkeeping and neither may be required to
    # use the derived key.
    ledger_content = re.compile(
        rf"(?:FROM|INTO)\s+public\.{TRACKER_TABLE}\b", re.IGNORECASE
    )
    bookkeeping = [line for line in text.splitlines() if ledger_content.search(line)]
    assert bookkeeping, (
        f"{COMPOSE_RUNNER.name} has no {TRACKER_TABLE} probe or insert to check"
    )
    for line in bookkeeping:
        # Two legitimate spellings of "derived", and no third. ``KEY_COLUMN`` is
        # the shell variable read off the live primary key; ``%I`` is the same
        # value reaching a server-side ``format()`` inside the convergence DO
        # block. A line naming a canonical column outright has neither -- which
        # is how the reintroduced ``applied_at`` literal in the first form of the
        # carry-forward was caught.
        assert "KEY_COLUMN" in line or "%I" in line, (
            f"{COMPOSE_RUNNER.name} names a tracker column literally instead of using "
            f"the derived key: {line.strip()}"
        )


@pytest.mark.unit
def test_the_compose_runner_converges_a_legacy_lane_without_dropping_anything() -> None:
    """An already-migrated lane converges by moving the table aside, not by a wipe.

    The ``.201`` dev lane carries the pre-fix table with rows in it. Convergence
    must preserve those rows and must never reach for ``DROP DATABASE`` or a
    ``DROP TABLE`` of the live tracker, both of which turn a shape fix into data
    loss on a lane somebody is using.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    assert "migration_name" in text, (
        f"{COMPOSE_RUNNER.name} no longer mentions the legacy key, so a lane already "
        "bootstrapped in the old shape has no convergence path and would need its "
        "volume wiped"
    )
    assert re.search(r"ALTER\s+TABLE[^\n]*RENAME\s+TO", text, re.IGNORECASE), (
        f"{COMPOSE_RUNNER.name} must move a legacy tracker aside by rename so the "
        "corpus can create the canonical table and the old rows can be copied back"
    )
    # Postgres does not rename a table's indexes with the table. Measured on
    # postgres:16-alpine: leave them and the corpus's own
    # ``CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at`` matches the
    # name the STASH still holds, skips, and is then dropped with the stash --
    # so the converged lane silently ends up with no applied_at index and the
    # canonical primary key lands as ``schema_migrations_pkey1``. Nothing fails
    # at the time, which is what makes it worth a gate rather than a comment.
    assert re.search(r"ALTER\s+INDEX[^\n]*RENAME\s+TO", text, re.IGNORECASE), (
        f"{COMPOSE_RUNNER.name} renames the legacy tracker aside but leaves its index "
        "names held by the stash, so the corpus's own index creation skips and the "
        "converged lane loses that index when the stash is dropped"
    )
    forbidden = re.search(
        rf"DROP\s+DATABASE|DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\s*\.\s*)?{TRACKER_TABLE}\b",
        text,
        re.IGNORECASE,
    )
    assert not forbidden, (
        f"{COMPOSE_RUNNER.name} drops the live tracker or its database: "
        f"{forbidden.group(0)!r}"
    )


@pytest.mark.unit
def test_the_compose_runner_makes_no_claim_about_a_shape_it_does_not_declare() -> None:
    """AC2: the seam comment must not assert a shape agreement the code cannot hold.

    The pre-fix comment said the bootstrap "is created with the same shape the
    corpus creates" while creating a different one. A false comment in the seam
    is what let this survive review the first time and the time before it, so it
    is refused mechanically rather than left to a reader.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    assert "same shape the corpus creates" not in text, (
        f"{COMPOSE_RUNNER.name} still claims its bootstrap matches the corpus shape. "
        "The runner no longer declares a shape at all; say that instead."
    )


#: Roots the completeness walk covers. ``tests`` is in scope deliberately: seven
#: files under it declare a tracker, and a walk that skipped them while its own
#: name said "every tracker declaration in the tree" would be the false claim
#: this module exists to refuse.
WALK_ROOTS: tuple[str, ...] = ("docker", "scripts", "src", "tests")

#: Suffixes the walk reads, compared case-INSENSITIVELY. ``.captured`` is here
#: because a captured fixture is still a file that declares a shape; ``.psql``
#: because psql scripts are ordinarily spelled that way. An extensionless file
#: whose first line is a shebang is read too: ``Path.suffix`` is empty for those
#: and a declarer in one would otherwise be in no registry, trip no walk, and
#: match no pre-commit pattern -- the exact invisibility this check exists for.
WALK_SUFFIXES: frozenset[str] = frozenset(
    {".sql", ".psql", ".sh", ".py", ".yaml", ".yml", ".captured"}
)


def _walk_reads(path: Path, suffixes: frozenset[str]) -> bool:
    """Whether the completeness walk reads ``path``."""
    if path.suffix.lower() in suffixes:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def _walk_declarers(
    roots: tuple[str, ...] = WALK_ROOTS,
    suffixes: frozenset[str] = WALK_SUFFIXES,
) -> tuple[dict[str, list[tuple[str, ...]]], int]:
    """Return every tracker declaration under ``roots``, and the files read.

    The file count is returned rather than discarded because a walk that read
    nothing returns the same empty mapping as a clean tree. ``Path.rglob`` on a
    directory that does not exist yields nothing and raises nothing, so a renamed
    root is exactly the silent zero this check would otherwise claim to prevent.
    """
    declarations: dict[str, list[tuple[str, ...]]] = {}
    read = 0
    for root in roots:
        base = REPO_ROOT / root
        assert base.is_dir(), (
            f"WALK_ROOTS names {root!r}, which is not a directory in this repo. "
            "rglob would return nothing and this check would report a clean tree"
        )
        for path in sorted(base.rglob("*")):
            if not path.is_file() or not _walk_reads(path, suffixes):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            read += 1
            if TRACKER_TABLE not in text:
                continue
            shapes = declared_column_sets(text)
            if shapes:
                declarations[path.relative_to(REPO_ROOT).as_posix()] = shapes
    return declarations, read


@pytest.mark.unit
def test_the_registry_accounts_for_every_tracker_declaration_in_the_tree() -> None:
    """No file may declare a tracker without this registry knowing about it.

    The thesis of this change is that hand-reconciled literals do not survive. A
    hand-maintained list of paths with no completeness check is the same class
    one level up: a new declarer would sit in no registry, match no pre-commit
    ``files:`` pattern, and be invisible to both layers. This walks the tree
    instead of trusting the list.
    """
    governed = {rel for paths in TRACKER_DECLARERS.values() for rel in paths}
    declarations, read = _walk_declarers()
    assert read > 100, (
        f"the completeness walk read only {read} files, so a zero result would "
        "prove nothing about the tree"
    )
    unaccounted = {
        where: shapes
        for where, shapes in declarations.items()
        if where not in governed and where not in UNGOVERNED_DECLARERS
    }
    assert not unaccounted, (
        f"these files declare a {TRACKER_TABLE} tracker and are in neither "
        "TRACKER_DECLARERS nor UNGOVERNED_DECLARERS, so nothing checks their "
        "agreement and no pre-commit hook fires on them:\n"
        + "\n".join(f"  {where}: {cols}" for where, cols in sorted(unaccounted.items()))
    )


@pytest.mark.unit
def test_the_completeness_walk_reaches_every_path_the_registry_names() -> None:
    """Positive control that exercises the WALK, not just the parser.

    An earlier form of this control read each exempted path directly with
    ``read_text``. That passes unchanged when the roots tuple is wrong, when the
    suffix filter excludes every real file, or when ``rglob`` is pointed at a
    directory that does not exist -- three of the four ways the walk can go
    silently empty, and precisely the ones its docstring claims to guard. This
    asserts the walk itself reaches every path the registry names.
    """
    declarations, _ = _walk_declarers()
    expected = set(UNGOVERNED_DECLARERS) | {
        rel
        for paths in TRACKER_DECLARERS.values()
        for rel in paths
        if declared_column_sets(_read(rel))
    }
    missing = expected - set(declarations)
    assert not missing, (
        "the completeness walk did not reach paths the registry names, so its "
        f"zero result proves nothing about the tree: {sorted(missing)}"
    )


@pytest.mark.unit
def test_a_corpus_owned_runner_declares_no_shape_by_any_other_spelling() -> None:
    """``CREATE TABLE`` is not the only way to declare a shape.

    The corpus-owned rule is the single gate over the subject database, so the
    ways around it matter more than usual. ``ALTER TABLE ... ADD COLUMN`` shapes
    the tracker just as surely, and applying some other vendored ``.sql`` would
    move the declaration into a file this registry never reads. The convergence
    step's own ``RENAME TO`` stays allowed on purpose: retiring a table this
    runner once created is not declaring a shape for it.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    shaping = _SHAPING_ALTER_RE.search(_strip_comments(text))
    assert shaping is None, (
        f"{COMPOSE_RUNNER.name} shapes {TRACKER_TABLE} through ALTER rather than "
        f"CREATE, which the corpus-owned rule alone would miss: "
        f"{shaping.group(0)!r}"
        if shaping
        else ""
    )
    bodyless = _BODYLESS_CREATE_RE.search(_collapsed(text))
    assert bodyless is None, (
        f"{COMPOSE_RUNNER.name} brings {TRACKER_TABLE} into existence with no column "
        f"list ({bodyless.group(0)!r} -- CREATE ... AS/OF, or SELECT ... INTO), which "
        "declares a shape by reference and has no body for the parser to compare"
        if bodyless
        else ""
    )
    # Every rule above matches a declaration that NAMES the table. This one does
    # not need to: the runner legitimately contains no CREATE TABLE at all, so
    # any is refused. That closes the spellings a name-matching rule cannot see --
    # `EXECUTE format('CREATE TABLE ... public.%I ...', 'schema_migrations')`,
    # which is this file's OWN idiom for the convergence, and a table name held
    # in a shell variable. Both reintroduce the defect in a form that looks
    # exactly like the surrounding code.
    body = _strip_shell_comments(text)
    stray_create = re.search(r"CREATE\s+(?:\w+\s+)*TABLE\b", body, re.IGNORECASE)
    assert stray_create is None, (
        f"{COMPOSE_RUNNER.name} contains a CREATE TABLE ({stray_create.group(0)!r}). "
        "This runner creates no table of its own by design; the corpus's tracking "
        "migration is its only bootstrap"
        if stray_create
        else ""
    )
    # The INSERT target list must be the derived key and nothing spelled out. A
    # per-line check for the derived token is satisfied by a line that ALSO names
    # a literal column, which is how `("${KEY_COLUMN}", applied_at)` survived a
    # review round.
    for target in re.findall(
        rf"INSERT\s+INTO\s+public\.{TRACKER_TABLE}\s*\(([^)]*)\)", _collapsed(body)
    ):
        assert re.fullmatch(r'(?:%I%s|%I|\\?"\$\{KEY_COLUMN\}\\?")', target.strip()), (
            f"{COMPOSE_RUNNER.name} spells a literal column in an INSERT target list on "
            f"the canonical tracker: ({target.strip()}). Only the derived key belongs there"
        )
    # Matched by SHAPE, not by quoting. The previous form keyed on the literal
    # `psql_db -f "$VAR`, so `psql_db -f /migrations/fixup.sql`, `--file`, a bare
    # `| psql_db`, and a hand-rolled `psql -h ... -f ...` each applied a file the
    # gate never reads while the assertion reported full coverage.
    appliers = [
        line.strip()
        for line in _strip_shell_comments(text).splitlines()
        if re.search(r"\bpsql", line)
        and re.search(r"(?:\s-f\b|\s--file\b|\|\s*psql)", line)
    ]
    allowed = {
        'psql_db -f "$BASELINE"',
        'psql_db -f "$TRACKING"',
        '{ manifest_guc_prelude "$conditions"; cat "${MIGRATION_DIR}/${name}"; } | psql_db -f -',
    }
    assert set(appliers) == allowed, (
        f"{COMPOSE_RUNNER.name}'s set of SQL-applying invocations changed. Each one "
        "either applies a file this gate reads, or moves a declaration into a file "
        "it never will:\n"
        f"  unexpected: {sorted(set(appliers) - allowed)}\n"
        f"  missing:    {sorted(allowed - set(appliers))}"
    )


@pytest.mark.unit
def test_the_compose_runner_refuses_a_lane_the_convergence_did_not_reach() -> None:
    """Every convergence step is conditional, so the runner must prove one ran.

    A skipped rename leaves the retired table in place, the corpus's
    ``CREATE TABLE IF NOT EXISTS`` no-ops against it exactly as it did before the
    fix, and the derived key resolves to the retired column -- announcing the old
    key on a line that reads like success and dying 46 files later on the
    original error. Without this the change cannot tell anyone it did not work.
    """
    text = COMPOSE_RUNNER.read_text(encoding="utf-8")
    assert re.search(r"FATAL[^\n]*(?:retired|migration_name)", text, re.IGNORECASE), (
        f"{COMPOSE_RUNNER.name} never refuses a lane where the retired column "
        "survived the bootstrap, so a skipped convergence reports success"
    )
    assert "attisdropped" in text, (
        f"{COMPOSE_RUNNER.name} must read the retired column from pg_catalog: "
        "information_schema is privilege-filtered and answers 'absent' for a "
        "table the role cannot read, which is the silent skip this guards"
    )
