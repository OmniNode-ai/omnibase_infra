# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19514 vendor identity for the two ticket-join migrations.

omnimarket's node_projection_delegation migration 0047 adds a nullable
``delegation_events.ticket_id``, and node_projection_dod_verdict migration 0002
adds a nullable ``dod_verify_runs.delegation_correlation_id``, so a delegation
run joins to the ticket it worked and to the DoD verdict that judged it. Both
are vendored here FIRST, ahead of the omnimarket source, per the node-migration
vendor-parity ordering. The real-Postgres proofs of the columns and the writers
live beside the source in omnimarket.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"

# OMN-17887 retired the `tenant` schema and made `public` the TENANT domain's
# schema. That silently killed the positive control below, which retargeted a
# migration at `public.pg_class`: `public` became a KNOWN topology schema, so
# the retarget stopped being a violation, the control returned clean, and the
# test started failing on `assert () != ()`. That failure is the control doing
# its job -- the alternative was a gate whose every zero meant nothing.
#
# The live rule, measured against the shipped linter on all four profiles, is
# relation IDENTITY rather than schema: a relation the topology registry knows
# is accepted qualified, unqualified, or under the wrong schema, and a relation
# it does not know is refused. So the falsifier is an UNREGISTERED relation, and
# it stays one for as long as that is the rule.
_UNREGISTERED_RELATION = "omn19514_unregistered_relation"

#: (node, file, domain, sha256, the one column it adds, its type, the table
#: statement a broken copy retargets for the linter's positive control)
_VENDORED = (
    (
        "node_projection_delegation",
        "0047_delegation_events_ticket_id.sql",
        "tenant",
        "e7d8902e2e071da4c0fafcb1df139fc0b68c4e95f6f2dcc5262004235b267819",
        ("ticket_id", "TEXT"),
        "ALTER TABLE delegation_events",
    ),
    (
        "node_projection_dod_verdict",
        "0002_dod_verify_runs_delegation_correlation_id.sql",
        "omninode_internal",
        "f057edc26ff6498ce7c3156a2d51c9477703f859e19083f934198c7e4edc2e83",
        ("delegation_correlation_id", "UUID"),
        "ALTER TABLE omninode_internal.dod_verify_runs",
    ),
)
_IDS = [entry[1] for entry in _VENDORED]


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def _sql(node: str, filename: str) -> str:
    return (_FORWARD / "nodes" / node / filename).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("node", "filename", "domain", "sha256", "column", "table"), _VENDORED, ids=_IDS
)
def test_vendor_bytes_and_manifest_binding_are_exact(
    node: str,
    filename: str,
    domain: str,
    sha256: str,
    column: tuple[str, str],
    table: str,
) -> None:
    artifact_path = f"nodes/{node}/{filename}"
    vendored = _FORWARD / "nodes" / node / filename
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == sha256
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{node}",
        f"node:{node}",
        domain,
        f"node:{node}:{filename}",
        sha256,
    ]


@pytest.mark.parametrize(
    ("node", "filename", "domain", "sha256", "column", "table"), _VENDORED, ids=_IDS
)
def test_migration_is_declared_expand_only_and_is_purely_additive(
    node: str,
    filename: str,
    domain: str,
    sha256: str,
    column: tuple[str, str],
    table: str,
) -> None:
    """The class line matches the bytes: one nullable column and an index."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{node}/{filename}"] == "expand-only"
    statements = "\n".join(
        line
        for line in _sql(node, filename).splitlines()
        if not line.lstrip().startswith("--")
    )
    assert re.findall(r"ADD COLUMN IF NOT EXISTS (\w+) (\w+)", statements) == [column]
    # A NOT NULL constraint is forbidden; the partial index's IS NOT NULL
    # predicate is not a constraint.
    assert not re.search(r"(?<!IS )NOT NULL", statements.upper())
    for forbidden in (
        "DROP",
        "UPDATE",
        "DELETE",
        "CREATE OR REPLACE",
        "UNIQUE",
    ):
        assert forbidden not in statements.upper(), forbidden


@pytest.mark.parametrize(
    ("node", "filename", "domain", "sha256", "column", "table"), _VENDORED, ids=_IDS
)
def test_migration_passes_the_application_database_sql_gate(
    node: str,
    filename: str,
    domain: str,
    sha256: str,
    column: tuple[str, str],
    table: str,
) -> None:
    """Each clears the OMN-15361 gate on every shipped profile."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(node, filename)
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


@pytest.mark.parametrize(
    ("node", "filename", "domain", "sha256", "column", "table"), _VENDORED, ids=_IDS
)
def test_the_linter_is_live_positive_control(
    node: str,
    filename: str,
    domain: str,
    sha256: str,
    column: tuple[str, str],
    table: str,
) -> None:
    """A zero from the linter means something only if it can return non-zero.

    Asserted on every profile the gate above clears, not just ``local``: a
    control that is live on one profile and dead on three would leave the other
    three zeros unfalsifiable, which is the failure this test exists to catch.
    """
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(node, filename)
    broken = sql.replace(table, f"ALTER TABLE {_UNREGISTERED_RELATION}")
    assert broken != sql, f"the {table!r} anchor is no longer present in {filename}"
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(
            broken, load_topology_profile(profile)
        )
        assert violations != (), (
            f"{profile}: the linter returned clean on a migration retargeted at "
            f"{_UNREGISTERED_RELATION!r}, so its zero on the real file proves "
            "nothing. The control is dead -- find a falsifier the current rule "
            "set still refuses before trusting this gate."
        )
