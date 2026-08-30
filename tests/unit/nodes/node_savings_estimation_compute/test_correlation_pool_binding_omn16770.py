# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16770: the savings-correlation pool must bind the ``application`` database.

WHAT SHIPPED BROKEN
-------------------
``node_savings_estimation_compute``'s migration creates its two signal relations
in the **application** database (physical ``omnidash_analytics``), but
``service_kernel``'s correlation wiring built its asyncpg pool from
``OMNIBASE_INFRA_DB_URL`` — a *different* physical database. Every tick of the
periodic batch therefore raised, on the .201 dev lane, once per minute since the
node shipped in ``d4c48c68f`` (OMN-16293, 2026-08-23)::

    asyncpg.exceptions.UndefinedTableError:
        relation "omninode_internal.savings_injection_signals" does not exist

WHY ``application`` IS THE ANSWER, AND NOT A JUDGEMENT CALL
----------------------------------------------------------
OMN-16770 AC5 requires the data-ownership decision to be recorded with its
reason, because routing the migration to ``omnibase_infra`` and repointing the
reader at the application database are two *different* answers, not two
implementations of one. Four checked-in authorities already agree, and they are
what this test pins:

1. ``omnimarket/scripts/application-relation-ownership.yaml`` declares BOTH
   relations ``database_ref: application``, ``schema: omninode_internal``. That
   file is the ownership authority ``scripts/ci/check_application_database_sql.py``
   reads via ``--ownership-manifest``.
2. The migration's own trailing ``GRANT`` names ``omninode_runtime`` — which is
   the principal of the ``omninode_runtime_service`` binding, and that binding's
   ``dsn_env`` is ``OMNINODE_INTERNAL_DB_URL`` (``topology/instances/local.yaml``).
3. ``_ledger/application-migrations.tsv`` declares the artifact's domain as
   ``omninode_internal``, and the node-migration runner applies that tree to the
   application database.
4. Three of the five relations the handler joins — ``llm_call_metrics``,
   ``session_outcomes``, ``savings_estimates`` — are written ONLY into
   ``omnidash_analytics`` by omnimarket's projection nodes. Moving the signal
   tables to ``omnibase_infra`` would strand the join, so that direction was
   never actually available.

Only two surfaces disagreed: the 0001 migration's header comment, and the
``service_kernel`` DSN read. This test pins the DSN read. The header comment is
deliberately NOT rewritten — 0001 is already applied on every lane and its
content SHA-256 is pinned in the manifest, so editing it in place is the
OMN-17139 defect (a rewritten applied migration), not a fix.

STATIC BY DESIGN: these assertions fire on a host with no Docker and no
Postgres, which is where a silent revert would otherwise go unnoticed until the
next lane readback.

Ticket: OMN-16770. Node: OMN-16293. Binding identity epic: OMN-15426 / OMN-16843.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

SERVICE_KERNEL = REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "service_kernel.py"
LOCAL_INSTANCE = (
    REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances" / "local.yaml"
)
NODE_MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_savings_estimation_compute"
    / "0001_create_savings_signal_tables.sql"
)
MANIFEST = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)

MIGRATION_ARTIFACT = (
    "nodes/node_savings_estimation_compute/0001_create_savings_signal_tables.sql"
)
RUNTIME_BINDING = "omninode_runtime_service"
RUNTIME_PRINCIPAL = "omninode_runtime"
WRONG_DSN_KEY = "OMNIBASE_INFRA_DB_URL"

# The savings-correlation wiring block in service_kernel.py, delimited by its
# own section markers so an unrelated edit elsewhere in the kernel cannot move
# this test's goalposts.
_SAVINGS_BLOCK = re.compile(
    r"# 3\.9\. Wire savings estimation correlation.*?"
    r"(?=\n        # 3\.\d{2}\.|\n        # 4\.|\Z)",
    re.DOTALL,
)


def _savings_block() -> str:
    source = SERVICE_KERNEL.read_text(encoding="utf-8")
    match = _SAVINGS_BLOCK.search(source)
    assert match is not None, (
        "the savings-correlation wiring block ('# 3.9. Wire savings estimation "
        "correlation') is no longer locatable in service_kernel.py — this test "
        "pins that block's DSN binding and cannot silently pass without it"
    )
    return match.group(0)


def _savings_block_code() -> str:
    """The wiring block with comment lines stripped.

    The negative guard below asserts that no *executable* line resolves the
    wrong DSN. Prose may — and does — name ``OMNIBASE_INFRA_DB_URL`` when
    explaining the defect this ticket closes, and a guard that cannot tell a
    citation from a call would forbid documenting the bug it protects against.
    """
    lines = _savings_block().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


# The pool's DSN read, captured so the assertion is on the key the call actually
# resolves — not on the key appearing anywhere in the block. A log message that
# merely NAMES the right variable must not be able to turn this test green.
_DSN_READ = re.compile(
    r"_savings_dsn\s*=\s*os\.environ\.get\(.*?[\"'](?P<key>[A-Z0-9_]+)[\"']",
    re.DOTALL,
)


def _savings_dsn_env_key() -> str:
    """The env key the correlation pool's own ``os.environ.get`` resolves."""
    match = _DSN_READ.search(_savings_block_code())
    assert match is not None, (
        "could not locate the `_savings_dsn = os.environ.get(...)` read in the "
        "savings-correlation wiring block — this test pins which env key that "
        "call resolves and must not pass when it cannot find the call"
    )
    return match.group("key")


def _application_runtime_dsn_env() -> str:
    """The DSN env the topology binds for omninode_runtime on `application`."""
    instance = yaml.safe_load(LOCAL_INSTANCE.read_text(encoding="utf-8"))
    binding = instance["databases"]["application"]["bindings"][RUNTIME_BINDING]
    assert binding["database_ref"] == "application"
    assert binding["principal"] == RUNTIME_PRINCIPAL
    dsn_env: str = binding["dsn_env"]
    return dsn_env


def test_topology_binds_application_runtime_to_the_internal_dsn() -> None:
    """Pin the authority the kernel is required to agree with."""
    assert _application_runtime_dsn_env() == "OMNINODE_INTERNAL_DB_URL"
    instance = yaml.safe_load(LOCAL_INSTANCE.read_text(encoding="utf-8"))
    assert instance["databases"]["application"]["physical_name"] == "omnidash_analytics"


def test_savings_correlation_pool_reads_the_application_binding_dsn() -> None:
    """RED before the fix: the block read OMNIBASE_INFRA_DB_URL.

    This is the whole defect in one assertion. The pool must resolve the same
    DSN env the topology binds for ``omninode_runtime`` on ``application`` —
    the database the node's own migration actually writes into.
    """
    resolved = _savings_dsn_env_key()
    expected = _application_runtime_dsn_env()
    assert resolved == expected, (
        f"the savings-correlation pool resolves {resolved!r} but must resolve "
        f"{expected!r} (the {RUNTIME_BINDING!r} binding on the `application` "
        f"database, principal {RUNTIME_PRINCIPAL!r}) — that is the database "
        "0001_create_savings_signal_tables.sql creates its relations in"
    )


def test_savings_correlation_pool_does_not_read_the_omnibase_infra_dsn() -> None:
    """The regression guard, stated as the negative it actually is.

    ``OMNIBASE_INFRA_DB_URL`` is a DIFFERENT physical database
    (``omnibase_infra``). It holds neither signal relation, and reintroducing it
    here reinstates the exact UndefinedTableError this ticket closes.
    """
    code = _savings_block_code()
    assert WRONG_DSN_KEY not in code, (
        f"{WRONG_DSN_KEY!r} points at the `omnibase_infra` physical database, "
        "which contains neither omninode_internal.savings_injection_signals nor "
        "omninode_internal.savings_validator_catch_signals (OMN-16770)"
    )


def test_migration_grants_the_binding_principal_on_both_relations() -> None:
    """AC3: both relations, not one — they have identical exposure."""
    sql = NODE_MIGRATION.read_text(encoding="utf-8")
    for relation in (
        "omninode_internal.savings_injection_signals",
        "omninode_internal.savings_validator_catch_signals",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {relation} (" in sql
        assert (
            f"GRANT SELECT, INSERT, UPDATE ON {relation} TO {RUNTIME_PRINCIPAL};" in sql
        ), (
            f"{relation} must be granted to {RUNTIME_PRINCIPAL!r} — the principal "
            "the correlation pool now connects as"
        )


def test_manifest_declares_the_migration_in_the_internal_domain() -> None:
    """The migration is an application-stream artifact in the internal domain."""
    rows = [
        line.split("\t")
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    declared = [row for row in rows if row[0] == MIGRATION_ARTIFACT]
    assert len(declared) == 1, (
        f"expected exactly one manifest declaration for {MIGRATION_ARTIFACT}, "
        f"got {len(declared)}"
    )
    assert declared[0][3] == "omninode_internal"
