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

_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:[A-Za-z_][\w$]*\s*\.\s*)?"
    rf"{TRACKER_TABLE}\b\s*\(",
    re.IGNORECASE,
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
    for match in _CREATE_RE.finditer(text):
        body = _balanced_body(text, match.end() - 1)
        columns: list[str] = []
        for item in _split_top_level(body):
            stripped = item.strip()
            if not stripped:
                continue
            # Drop SQL comments so a commented column name is not read as one.
            stripped = re.sub(r"--[^\n]*", "", stripped).strip()
            if not stripped:
                continue
            first = stripped.split()[0].strip('"').lower()
            if first in _CONSTRAINT_LEADERS:
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
    assert re.search(
        r"FATAL[^\n]*TRACKING|TRACKING[^\n]*\bFATAL", text, re.IGNORECASE
    ) or ("$TRACKING" in text or "${TRACKING" in text), (
        f"{COMPOSE_RUNNER.name} must resolve the corpus tracking file through a fail-closed guard"
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
    # Word-bounded on purpose: the convergence step probes for the retired
    # ``schema_migrations_legacy_omn18544`` stash, which is not the tracker and
    # is correctly addressed by its own literal name.
    live_tracker = re.compile(rf"\b{TRACKER_TABLE}\b")
    bookkeeping = [
        line
        for line in text.splitlines()
        if live_tracker.search(line)
        and ("SELECT count(*)" in line or "INSERT INTO" in line)
    ]
    assert bookkeeping, (
        f"{COMPOSE_RUNNER.name} has no {TRACKER_TABLE} probe or insert to check"
    )
    for line in bookkeeping:
        assert "KEY_COLUMN" in line, (
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
