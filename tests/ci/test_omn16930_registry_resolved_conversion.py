# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16930: the corpus-level invariants behind the 0031 -> 0032 supersession.

Two things could quietly undo this change, and neither is visible to any
existing gate:

1. Someone "fixes" 0031 in place. Its bytes are load-bearing — the .201 dev
   lane's ``platform_catalog.schema_migrations`` holds their sha256, and
   ``_ledger/bootstrap.sql`` raises ``conflicting migration checksum in
   canonical node history`` on any divergence, permanently bricking
   forward-migration on that lane (the OMN-16705 class). The manifest row
   would move with the edit, so the manifest gate stays green while the lane
   breaks. Only a pin against the value read from the live lane catches it.

2. Someone re-adds a literal slug map to the replacement. 0032 exists because
   a literal map is incomplete by construction: the registry gains tenants on
   every beta signup, and a migration is immutable once applied. A future
   "just add the one missing slug" edit is the exact regression, and it would
   pass every test in the corpus because a hardcoded CASE is perfectly valid
   SQL.

Both are pinned here as data, not prose.

Ticket: OMN-16930. Ruling of record, verbatim: "Hold + fix mechanism".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_NODES = (
    Path(__file__).resolve().parents[2] / "docker" / "migrations" / "forward" / "nodes"
)
_LEDGER = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
)

_SUPERSEDED = (
    _NODES
    / "node_projection_delegation"
    / ("0031_delegation_events_tenant_id_to_uuid.sql")
)
# OMN-17288 superseded 0032 with 0033. The corpus invariants below follow the
# OPERATIVE conversion, because they exist to stop a future edit from
# re-introducing a literal map -- and that edit would land in whichever file is
# live, not in a retired one.
_SUPERSEDED_0032 = (
    _NODES
    / "node_projection_delegation"
    / ("0032_delegation_events_tenant_id_uuid_via_registry.sql")
)
_CONVERSION = (
    _NODES
    / "node_projection_delegation"
    / ("0033_delegation_events_uuid_via_registry_single_transaction.sql")
)
_MIRROR = (
    _NODES
    / "node_projection_tenant_registry"
    / ("0000_create_tenant_registry_mirror.sql")
)

# Read from the live .201 dev lane on 2026-08-29, read-only, as `postgres`:
#   SELECT checksum, checksum_kind FROM platform_catalog.schema_migrations
#   WHERE version = 'node:node_projection_delegation:0031_...sql';
#   -> 79ee3b02..., content_sha256
# The column there is already `uuid`. This is the value that makes an in-place
# edit fatal, so it is pinned from the DATABASE, not from the file.
_LANE_RECORDED_0031_SHA256 = (
    "79ee3b021d0a04088b2f733fa0558ea110b2a6f75b4fb338abe9c5c123f74442"
)

# The live-census slugs and the two non-tenant literals the SEED fixtures were
# written under. None may appear in the conversion's resolution path.
#
# OMN-17288: the census also contained one EXTERNAL CUSTOMER's slug, and this
# repository is PUBLIC, so it is gone from this tuple and replaced by the
# synthetic stand-in the omnimarket fixtures now use. A blacklist that has lost
# an entry is a weaker blacklist, so the loss is paid for structurally rather
# than absorbed: `test_the_transform_expression_carries_no_literal` and
# `test_no_case_map_survives_in_the_conversion` below reject the regression
# SHAPE -- any quoted literal inside the transform expression, and any CASE map
# anywhere -- for every slug that could ever exist, named or not. That is
# strictly more coverage than the seven strings this tuple ever had.
_SLUG_LITERALS = (
    "beta-business-proof",
    "beta-gateway-canary-79afa7263852",
    "d5-e2e-0b5ae67c",
    "delegation-spotcheck-1786977419",
    "t-external-fixture-omn17288",
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_sql_comments(sql: str) -> str:
    """Drop ``--`` line comments.

    0032's header quotes the forbidden shapes at length in order to explain why
    it does not use them; matching that prose would be a self-inflicted false
    positive. Same masking posture as
    ``test_node_migration_shape_reconciliation.py``.
    """
    return "\n".join(re.sub(r"--.*$", "", line) for line in sql.splitlines())


class TestSupersededMigrationIsImmutable:
    def test_0031_bytes_match_the_checksum_the_dev_lane_recorded(self) -> None:
        assert _sha256(_SUPERSEDED) == _LANE_RECORDED_0031_SHA256, (
            "0031's bytes changed. The .201 dev lane recorded "
            f"{_LANE_RECORDED_0031_SHA256} as content_sha256 in "
            "platform_catalog.schema_migrations, and _ledger/bootstrap.sql "
            "raises 'conflicting migration checksum in canonical node history' "
            "on any divergence -- permanently bricking forward-migration on "
            "that lane (OMN-16705 class). 0031 is RETIRED, not editable: "
            "express the change additively in a new migration and record a "
            "supersession row instead."
        )

    def test_the_manifest_declares_that_same_checksum(self) -> None:
        """The two pins must agree, or one of them is stale.

        A drift here means the manifest was regenerated from edited bytes --
        the exact way an in-place edit slips past the manifest gate.
        """
        rows = (_LEDGER / "application-migrations.tsv").read_text().splitlines()
        declared = [
            row.split("\t")[-1]
            for row in rows
            if row.startswith("nodes/node_projection_delegation/0031_")
        ]
        assert declared == [_LANE_RECORDED_0031_SHA256]

    def test_the_supersession_is_recorded(self) -> None:
        rows = (_LEDGER / "migration-supersessions.tsv").read_text().splitlines()
        matching = [
            row
            for row in rows
            if row.startswith("nodes/node_projection_delegation/0031_")
            and "0032_delegation_events_tenant_id_uuid_via_registry.sql" in row
            and "OMN-16930" in row
        ]
        assert len(matching) == 1, (
            "0031 -> 0032 must carry exactly one supersession row citing "
            "OMN-16930; found " + str(len(matching))
        )

    def test_the_second_supersession_is_recorded(self) -> None:
        """OMN-17288: 0032 -> 0033.

        Without this row ``check_migration_append_only.py`` would have refused
        the change at all -- 0032 is manifest-declared, so its bytes may only
        move in the same diff that lands a higher-ordinal successor.
        """
        rows = (_LEDGER / "migration-supersessions.tsv").read_text().splitlines()
        matching = [
            row
            for row in rows
            if row.startswith("nodes/node_projection_delegation/0032_")
            and "0033_delegation_events_uuid_via_registry_single_transaction.sql" in row
            and "OMN-17288" in row
        ]
        assert len(matching) == 1, (
            "0032 -> 0033 must carry exactly one supersession row citing "
            "OMN-17288; found " + str(len(matching))
        )


class TestReplacementResolvesFromTheRegistry:
    def test_no_slug_literal_appears_in_the_conversion(self) -> None:
        """AC3. The whole point of the mechanism.

        A slug literal here means someone re-introduced a hardcoded map, which
        is incomplete by construction and immutable once applied.
        """
        body = _strip_sql_comments(_CONVERSION.read_text(encoding="utf-8"))
        found = [slug for slug in _SLUG_LITERALS if slug in body]
        assert not found, (
            f"0032 carries slug literal(s) {found} outside its comments. "
            "Identity must resolve by JOINing tenant_registry_mirror at apply "
            "time -- a literal map cannot track a registry that gains tenants "
            "on every signup, and a migration is immutable once applied "
            "(OMN-16930)."
        )

    def test_the_transform_expression_carries_no_literal(self) -> None:
        """OMN-17288. The regression SHAPE, not a list of names.

        0031's defect was a literal map in the transform expression. The
        blacklist above can only catch slugs someone already enumerated; the
        registry gains tenants on every signup, so the next one to be
        hardcoded is by definition not in it. A transform expression that
        resolves by JOIN needs no quoted literal at all, so any quoted literal
        there is the regression regardless of what it spells.
        """
        body = _strip_sql_comments(_CONVERSION.read_text(encoding="utf-8"))
        transforms = re.findall(
            r"ALTER\s+COLUMN\s+tenant_id\s+TYPE\s+UUID\s+USING\s*\((.*?)\)",
            body,
            re.S | re.I,
        )
        assert transforms, (
            "no `ALTER COLUMN tenant_id TYPE UUID USING (...)` found -- the "
            "conversion changed shape and this guard is no longer looking at "
            "the thing it guards"
        )
        offenders = [expr for expr in transforms if "'" in expr]
        assert not offenders, (
            f"the transform expression(s) {offenders} carry a quoted literal. "
            "Identity must resolve by JOINing tenant_registry_mirror at apply "
            "time; a literal there is a hardcoded map, which is incomplete by "
            "construction and immutable once applied (OMN-16930)."
        )

    def test_no_case_map_survives_in_the_conversion(self) -> None:
        """The other half of the shape. 0031 was `CASE tenant_id WHEN ... END`.

        A CASE map is perfectly valid SQL and passes every other test in this
        corpus, which is exactly why it is named here.
        """
        body = _strip_sql_comments(_CONVERSION.read_text(encoding="utf-8"))
        assert not re.search(r"\bCASE\b", body, re.I), (
            "the conversion grew a CASE expression. 0031's closed literal CASE "
            "with no ELSE is the defect this whole supersession chain exists "
            "to retire -- resolution is a JOIN against tenant_registry_mirror."
        )

    def test_the_conversion_joins_the_mirror(self) -> None:
        body = _strip_sql_comments(_CONVERSION.read_text(encoding="utf-8"))
        assert "tenant_registry_mirror" in body
        assert "FROM tenant_registry_mirror m" in body

    def test_the_abort_names_the_projection_and_the_cause(self) -> None:
        """AC5. 0031's failure cost a week because it named the symptom.

        The replacement must say which lever to pull: the projection, not the
        data.
        """
        text = _CONVERSION.read_text(encoding="utf-8")
        assert "HAS NOT CAUGHT UP" in text
        assert "node_projection_tenant_registry" in text
        assert "onex.tenant.events" in text

    def test_the_mirror_declares_no_rls(self) -> None:
        """AC2, as a corpus invariant rather than a runtime one.

        An RLS-covered mirror is invisible to the migrate identity
        (role_omnidash, rolbypassrls=f, app.tenant_id unset), which reproduces
        the OMN-16493 blindness in the one place it would be fatal: the
        conversion would resolve every row to NULL and abort with the same
        uninformative error it exists to replace.
        """
        body = _strip_sql_comments(_MIRROR.read_text(encoding="utf-8")).upper()
        assert "ROW LEVEL SECURITY" not in body
        assert "CREATE POLICY" not in body

    def test_the_mirror_is_classified_internal_not_tenant(self) -> None:
        rows = (_LEDGER / "application-migrations.tsv").read_text().splitlines()
        matching = [
            row
            for row in rows
            if row.startswith("nodes/node_projection_tenant_registry/0000_")
        ]
        assert len(matching) == 1
        assert matching[0].split("\t")[3] == "omninode_internal", (
            "tenant_registry_mirror is a cross-tenant registry index, not "
            "tenant-scoped data. Classifying it 'tenant' would put it behind "
            "the isolation posture that makes it unreadable by the identity "
            "that has to resolve against it."
        )


class TestBothMigrationsStayFenced:
    def test_the_fence_holds_0031_and_0032_and_0033(self) -> None:
        """The un-gate is an operator action gated on TWO independent things.

        0033 releasing itself the moment it merges would abort every deploy
        until the projection caught up -- correctly, and uselessly. 0031 and
        0032 stay fenced because leaving a retired id fenced is what keeps it
        retired: when the operator un-gates, it is 0033 and only 0033.
        """
        fence = (
            Path(__file__).resolve().parents[2]
            / "docker"
            / "migrations"
            / "forward"
            / "fenced-node-migrations.yaml"
        ).read_text(encoding="utf-8")
        ids = re.findall(r'^\s*-\s*id:\s*"([^"]*)"', fence, re.M)
        assert (
            "node:node_projection_delegation:"
            "0031_delegation_events_tenant_id_to_uuid.sql" in ids
        )
        assert (
            "node:node_projection_delegation:"
            "0032_delegation_events_tenant_id_uuid_via_registry.sql" in ids
        )
        assert (
            "node:node_projection_delegation:"
            "0033_delegation_events_uuid_via_registry_single_transaction.sql" in ids
        ), (
            "0033 is the OPERATIVE conversion (OMN-17288) and must arrive "
            "fenced on the same interlock 0032 carried -- the mirror has to be "
            "caught up before it can resolve anything"
        )
        # The mirror's create is deliberately NOT fenced: it is 0032's
        # precondition, so fencing it would guarantee the ordering violation
        # 0032 is written to detect.
        assert (
            "node:node_projection_tenant_registry:"
            "0000_create_tenant_registry_mirror.sql" not in ids
        )

    def test_the_ungate_ordering_is_recorded_in_the_fence_file(self) -> None:
        """Recorded where the person releasing the fence will actually read it.

        The ordering lives on three Linear tickets, which is where it gets
        re-derived from. This asserts it also lives in the file whose edit IS
        the release.
        """
        fence = (
            Path(__file__).resolve().parents[2]
            / "docker"
            / "migrations"
            / "forward"
            / "fenced-node-migrations.yaml"
        ).read_text(encoding="utf-8")
        assert "UN-GATE ORDERING" in fence
        assert "CAUGHT UP" in fence
        assert "OMN-16804" in fence
        assert "OPERATOR" in fence
