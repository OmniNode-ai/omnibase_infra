# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit proof for the PRIVILEGE leg of the delegation conversion readiness check.

OMN-15683. The leg exists because migration 0036 passed every other leg this
tool had -- and every leg its hermetic and lab proofs had -- and then aborted on
onex-dev with ``permission denied for table tenant_registry_mirror``.

The mechanism under test is a STATIC derivation: the migration is split at its
``set_config('role', ...)`` switch, and every relation it reads is attributed to
the role that would be current at that point. It is derived from the migration's
own bytes rather than declared by hand, so it cannot drift away from the file it
describes -- and it is exercised here against the REAL vendored bytes of 0034,
0036 and 0037, not against synthetic SQL, because the whole value of the leg is
what it says about those three files.

The end-to-end half (has_table_privilege against a database carrying the
onex-dev ownership and ACL split) lives in
tests/integration/migrations/test_omn15683_mixed_representation_conversion.py.
Both halves are needed: this one proves the plan is derived correctly, that one
proves the plan is asserted correctly.

Ticket: OMN-15683.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = (
    REPO_ROOT / "scripts" / "ci" / "check_delegation_tenant_conversion_readiness.py"
)
_MIGRATIONS = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation"
)
_0034 = _MIGRATIONS / "0034_delegation_events_uuid_via_registry_role_set_guard.sql"
_0036 = _MIGRATIONS / "0036_delegation_events_uuid_mixed_representation.sql"
_0037 = (
    _MIGRATIONS
    / "0037_delegation_events_uuid_mixed_representation_guard_before_set_role.sql"
)

MIRROR = "tenant_registry_mirror"
EVENTS = "delegation_events"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_omn15683_readiness", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def readiness() -> ModuleType:
    return _load()


def _plan(readiness: ModuleType, path: Path) -> dict[str, str]:
    return {
        item.relation: item.phase
        for item in readiness.derive_role_read_plan(path.read_text(encoding="utf-8"))
    }


def test_0036_reads_the_mirror_in_the_owner_role(readiness: ModuleType) -> None:
    """The defect, stated as a fact about the file rather than about a deploy.

    This is the leg that would have caught 0036 before staging deploy run
    34281092205 spent a red deploy discovering it.
    """
    plan = _plan(readiness, _0036)
    assert plan[MIRROR] == readiness.PHASE_OWNER, (
        "0036 no longer reads the mirror after its role switch -- if that is "
        "genuinely true the file changed, and it must not have"
    )
    assert plan[EVENTS] == readiness.PHASE_OWNER


def test_0034_carries_the_same_defect(readiness: ModuleType) -> None:
    """0036 inherited the ordering from 0034; the leg sees it in both.

    A leg that only ever fired on the one file it was written for would be a
    hardcoded assertion wearing a derivation's clothes.
    """
    assert _plan(readiness, _0034)[MIRROR] == readiness.PHASE_OWNER


def test_0037_reads_the_mirror_before_the_role_switch(readiness: ModuleType) -> None:
    """The repair, asserted on the bytes rather than on the header comment."""
    plan = _plan(readiness, _0037)
    assert plan[MIRROR] == readiness.PHASE_SESSION
    # And delegation_events is still read as its owner -- the repair moved ONE
    # read, it did not abandon the SET ROLE the blindness reconciliation needs.
    assert plan[EVENTS] == readiness.PHASE_OWNER


def test_the_plan_names_only_real_relations(readiness: ModuleType) -> None:
    """Prose inside exception messages is not SQL. Measured, not assumed.

    Without the string-literal blanking pass the derivation reports relations
    called ``omninode_cloud.public`` (from "a tenant that DOES exist in
    omninode_cloud.public.tenants") and ``the`` (from "genuinely absent FROM
    the registry") -- both of them read out of message text these files carry.
    """
    for path in (_0034, _0036, _0037):
        relations = set(_plan(readiness, path))
        assert relations <= {EVENTS, MIRROR}, (
            f"{path.name} derived spurious relations: "
            f"{sorted(relations - {EVENTS, MIRROR})}"
        )
        # Not vacuous: each file really does read both.
        assert relations == {EVENTS, MIRROR}


def test_a_set_returning_function_is_not_a_relation(readiness: ModuleType) -> None:
    """0037 resolves against ``jsonb_to_recordset(<variable>)``, not a table.

    A function call carries no table privilege, so reporting one as a
    (role, relation) pair would make the leg ask PostgreSQL a question it
    cannot answer -- and, worse, would report a finding on a name that is not
    a grantable object at all.
    """
    assert "jsonb_to_recordset" not in _plan(readiness, _0037)
    # Positive control: the same name spelled WITHOUT a call IS a relation.
    bare = {
        item.relation: item.phase
        for item in readiness.derive_role_read_plan(
            "SELECT * FROM jsonb_to_recordset;\n"
        )
    }
    assert bare == {"jsonb_to_recordset": readiness.PHASE_SESSION}


def test_catalog_relations_are_not_privilege_questions(readiness: ModuleType) -> None:
    """pg_catalog is readable by every role and is not part of a grant topology."""
    for path in (_0034, _0036, _0037):
        assert not [
            relation
            for relation in _plan(readiness, path)
            if relation.startswith("pg_")
        ]


def test_a_relation_the_file_creates_is_not_asserted(readiness: ModuleType) -> None:
    """A relation the migration CREATES is not a pre-existing privilege question.

    None of the three vendored files creates one today -- 0037 carries its
    mirror snapshot in a PL/pgSQL variable precisely so that no relation
    exists -- so the exclusion is exercised on synthetic SQL rather than left
    unproven. Asserting has_table_privilege about a relation that does not
    exist when a read-only probe runs would ask a question with no answer.
    """
    sql = (
        "CREATE TEMP TABLE scratch_thing AS SELECT 1;\n"
        "SELECT * FROM scratch_thing;\n"
        "SELECT * FROM tenant_registry_mirror;\n"
    )
    plan = {item.relation: item.phase for item in readiness.derive_role_read_plan(sql)}
    assert plan == {MIRROR: readiness.PHASE_SESSION}, (
        "the created-relation exclusion no longer holds"
    )
    # Positive control: without the CREATE, the same read IS reported.
    bare = {
        item.relation: item.phase
        for item in readiness.derive_role_read_plan("SELECT * FROM scratch_thing;\n")
    }
    assert bare == {"scratch_thing": readiness.PHASE_SESSION}


def test_comment_stripping_preserves_offsets(readiness: ModuleType) -> None:
    """The boundary is a character offset, so a shortening pass would misplace it."""
    for path in (_0034, _0036, _0037):
        raw = path.read_text(encoding="utf-8")
        assert len(readiness.strip_sql_comments(raw)) == len(raw)
        assert len(readiness.blank_sql_string_literals(raw)) == len(raw)


def test_a_commented_out_role_switch_does_not_move_the_boundary(
    readiness: ModuleType,
) -> None:
    """A file whose only ``set_config('role')`` is in a comment has no switch.

    Otherwise a header that merely DESCRIBES the switch -- which all three of
    these files do -- would split the plan at the wrong place and attribute
    every read to the owner role.
    """
    sql = (
        "-- describes set_config('role', v_owner, true) in prose only\n"
        "SELECT 1 FROM tenant_registry_mirror;\n"
    )
    plan = {item.relation: item.phase for item in readiness.derive_role_read_plan(sql)}
    assert plan == {MIRROR: readiness.PHASE_SESSION}


def test_evaluate_refuses_on_a_privilege_finding(readiness: ModuleType) -> None:
    """A denied pair is REFUSED, not INDETERMINATE, and names role and relation.

    REFUSED because it is a definite negative answer: the migration WILL abort
    on it. It is evaluated ahead of the visibility legs so a blinded table
    cannot mask it -- which is what happens in practice, since a migrate
    identity that cannot see the table is also the one whose reconstruction
    falls short.
    """
    visibility = readiness.Visibility(
        row_security_enabled=True,
        row_security_forced=True,
        row_security_active=False,
        n_live_tup=233,
    )
    finding = readiness.PrivilegeFinding(
        role="role_omninode_owner",
        relation=MIRROR,
        phase=readiness.PHASE_OWNER,
    )
    verdict, reasons = readiness.evaluate(
        visibility, readiness.MODE_DIRECT, [], [finding]
    )
    assert verdict == readiness.VERDICT_REFUSED
    assert len(reasons) == 1
    assert "role_omninode_owner" in reasons[0]
    assert MIRROR in reasons[0]
    assert "permission denied for table" in reasons[0]


def test_no_privilege_finding_leaves_the_other_legs_in_charge(
    readiness: ModuleType,
) -> None:
    """The positive control for the assertion above.

    Without it, ``verdict == REFUSED`` could hold for reasons that have nothing
    to do with the privilege leg, and an empty finding list would look like a
    pass it never earned.
    """
    visibility = readiness.Visibility(
        row_security_enabled=True,
        row_security_forced=True,
        row_security_active=False,
        n_live_tup=0,
    )
    verdict, reasons = readiness.evaluate(visibility, readiness.MODE_DIRECT, [], [])
    assert verdict == readiness.VERDICT_PASS
    assert reasons == []


def test_the_default_migration_is_the_operative_successor(
    readiness: ModuleType,
) -> None:
    """The leg checks the file that will actually run, with no flag to set.

    A default pointing at a retired ordinal would report a plan nobody is going
    to execute, which is worse than no leg at all.
    """
    default = (REPO_ROOT / readiness.DEFAULT_MIGRATION).resolve()
    assert default == _0037.resolve()
    assert default.is_file()
