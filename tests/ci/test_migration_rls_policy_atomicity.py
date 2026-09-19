# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first proof for the OMN-17298 RLS-policy ratchet.

Every control that matters here runs against REAL checked-in migration bytes,
not against a synthetic string that happens to look like the defect. The two
files that carry the defect (0032, superseded by OMN-17288; and
node_canary_score_reducer 0003, found by this very gate) and the two that fix
it (0033; and 0004, added by OMN-17298) are all in the tree, so the gate is
proven against the actual regression and the actual remedy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (
    _REPO_ROOT / "scripts" / "validation" / "check_migration_rls_policy_atomicity.py"
)
_FORWARD = _REPO_ROOT / "docker" / "migrations" / "forward"
_NODES = _FORWARD / "nodes"

_DELEGATION = _NODES / "node_projection_delegation"
_CANARY = _NODES / "node_canary_score_reducer"

# The four real files this gate is defined by.
_DEFECT_0032 = _DELEGATION / "0032_delegation_events_tenant_id_uuid_via_registry.sql"
_FIX_0033 = (
    _DELEGATION / "0033_delegation_events_uuid_via_registry_single_transaction.sql"
)
_DEFECT_0003 = _CANARY / "0003_capability_scores_tenant_id_to_uuid.sql"
_FIX_0004 = _CANARY / "0004_capability_scores_policy_atomic_restatement.sql"


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_migration_rls_policy_atomicity", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> object:
    return _load()


def _violations(gate: object, path: Path) -> list[str]:
    return gate.violations_for(  # type: ignore[attr-defined]
        path.relative_to(_REPO_ROOT), path.read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# RED controls -- real defective bytes, still in the tree.
# ---------------------------------------------------------------------------


def test_red_control_rule_b_catches_migration_0032(gate: object) -> None:
    """The OMN-17288 defect, replayed from the file's own bytes.

    0032 drops tenant_isolation inside its DO block and recreates it after
    `END$$`. Because the runner has no --single-transaction, `END$$` commits
    and the relation is briefly FORCE-RLS with no policy. If this assertion
    ever goes green without 0032 changing, the gate has stopped working.
    """
    assert _DEFECT_0032.is_file(), (
        "0032 must stay in the tree as this gate's RED control"
    )
    found = _violations(gate, _DEFECT_0032)
    assert any("RULE B" in v for v in found), found
    assert any("delegation_events" in v for v in found), found


def test_red_control_rule_b_catches_canary_0003(gate: object) -> None:
    """The third occurrence -- and the first that was actually applied.

    Found by running this gate over the checked-in tree, not by review. 0003
    is recorded in the .201 dev lane's platform_catalog.schema_migrations
    (applied 2026-08-17), so this window was traversed on a real database.
    """
    assert _DEFECT_0003.is_file()
    found = _violations(gate, _DEFECT_0003)
    assert any("RULE B" in v for v in found), found
    assert any("capability_scores" in v for v in found), found


def test_red_control_rule_a_catches_rls_with_no_policy_at_all(gate: object) -> None:
    """Enforcement switched on with no admitting rule anywhere in the file."""
    sql = """
    ALTER TABLE public.some_projection ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.some_projection FORCE ROW LEVEL SECURITY;
    GRANT SELECT ON public.some_projection TO app_dashboard;
    """
    found = gate.violations_for(Path("synthetic.sql"), sql)  # type: ignore[attr-defined]
    assert any("RULE A" in v for v in found), found
    assert any("some_projection" in v for v in found), found


# ---------------------------------------------------------------------------
# GREEN controls -- the shapes that are correct must not be flagged.
# ---------------------------------------------------------------------------


def test_migration_0033_the_canonical_fix_shape_passes(gate: object) -> None:
    """0033 is the shape the gate exists to require. It must be clean."""
    assert _FIX_0033.is_file()
    assert _violations(gate, _FIX_0033) == []


def test_migration_0004_added_by_this_ticket_passes(gate: object) -> None:
    """The successor OMN-17298 lands must satisfy the gate it ships with."""
    assert _FIX_0004.is_file()
    assert _violations(gate, _FIX_0004) == []


def test_prose_that_quotes_the_defect_is_not_read_as_code(gate: object) -> None:
    """0033's header quotes the exact statements that made 0032 defective.

    A regex over raw bytes reads that prose as code and reports the FIXED file
    as broken. This is the specific false positive that makes comment and
    string-literal scrubbing mandatory rather than tidy, and 0033 is the real
    file that proves it -- its header contains, verbatim:

        --   DROP POLICY IF EXISTS tenant_isolation ON delegation_events;
        --   CREATE POLICY tenant_isolation ON delegation_events ...;
    """
    raw = _FIX_0033.read_text(encoding="utf-8")
    assert "--   DROP POLICY IF EXISTS tenant_isolation ON delegation_events;" in raw
    assert _violations(gate, _FIX_0033) == []


def test_no_force_row_level_security_is_not_enforcement_on(gate: object) -> None:
    """`NO FORCE` relaxes enforcement; it can never strand a relation."""
    sql = "ALTER TABLE public.t NO FORCE ROW LEVEL SECURITY;"
    assert gate.violations_for(Path("synthetic.sql"), sql) == []  # type: ignore[attr-defined]


def test_drop_policy_outside_any_block_is_not_a_rule_b_violation(gate: object) -> None:
    """Rule B is about the DO-block commit boundary, nothing else.

    A top-level DROP + CREATE pair runs as two adjacent statements with no
    block commit between them, which is the pre-existing shape of most of
    this tree and is not what this gate is for.
    """
    sql = """
    DROP POLICY IF EXISTS tenant_isolation ON public.t;
    CREATE POLICY tenant_isolation ON public.t FOR ALL USING (true);
    """
    assert gate.violations_for(Path("synthetic.sql"), sql) == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Exemption semantics and the whole-tree gate.
# ---------------------------------------------------------------------------


def test_superseded_predecessors_are_the_only_exemption(gate: object) -> None:
    """Both defective files are exempt only because a successor landed.

    The exemption is not an allowlist: a row in migration-supersessions.tsv
    exists only once a strictly-higher-ordinal successor is in the tree, so it
    cannot be used to admit a NEW violation -- admitting one requires shipping
    its fix in the same change.
    """
    superseded = gate.load_superseded(_REPO_ROOT)  # type: ignore[attr-defined]
    assert (
        "nodes/node_projection_delegation/"
        "0032_delegation_events_tenant_id_uuid_via_registry.sql" in superseded
    )
    assert (
        "nodes/node_canary_score_reducer/"
        "0003_capability_scores_tenant_id_to_uuid.sql" in superseded
    )
    for predecessor in superseded:
        successor_rows = [
            line.split("\t")
            for line in (_FORWARD / "_ledger" / "migration-supersessions.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and line.split("\t")[0] == predecessor
        ]
        assert successor_rows, predecessor
        for row in successor_rows:
            assert (_FORWARD / row[1]).is_file(), row[1]


def test_the_whole_checked_in_forward_tree_is_clean(gate: object) -> None:
    """The gate is fail-closed over the real tree, not only over fixtures."""
    assert gate.check(_REPO_ROOT) == []  # type: ignore[attr-defined]


def test_the_gate_exits_non_zero_when_a_violation_exists(
    gate: object, tmp_path: Path
) -> None:
    """Exit-code contract: CI and pre-commit both key on it."""
    forward = tmp_path / "docker" / "migrations" / "forward" / "nodes" / "n"
    forward.mkdir(parents=True)
    (forward / "0001_bad.sql").write_text(
        "ALTER TABLE public.t ENABLE ROW LEVEL SECURITY;\n", encoding="utf-8"
    )
    assert gate.main(["--root", str(tmp_path)]) == 1  # type: ignore[attr-defined]
    (forward / "0001_bad.sql").write_text(
        "ALTER TABLE public.t ENABLE ROW LEVEL SECURITY;\n"
        "CREATE POLICY tenant_isolation ON public.t FOR ALL USING (true);\n",
        encoding="utf-8",
    )
    assert gate.main(["--root", str(tmp_path)]) == 0  # type: ignore[attr-defined]


class TestRuleBAdmitsTheDisablingForm:
    """OMN-18774: removing a posture is not the same as stranding one.

    RULE B's hazard is a COMMITTED state of "row-level security enforcing,
    zero policies", which denies every row to every non-owner principal. A
    block that drops the last policy and DISABLEs enforcement on the same
    relation in the same commit produces no such state.

    This arm exists because the recreate RULE B demanded is, for an
    omninode_internal relation, the very thing the operator ruled out: such a
    relation receives no tenant stamping and no row-level security at all
    (docs/tracking/ROLLING_WORK_LEDGER.md:654). Without the arm the gate would
    have no passing shape for a posture REMOVAL, and the only way past it
    would be an allowlist -- which is how a gate stops meaning anything.
    """

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        forward = tmp_path / "docker" / "migrations" / "forward" / "nodes" / "n"
        forward.mkdir(parents=True, exist_ok=True)
        path = forward / "0001_posture.sql"
        path.write_text(body, encoding="utf-8")
        return path

    def test_drop_then_disable_in_one_block_passes(
        self, gate: object, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path,
            "DO $b$ BEGIN\n"
            "  DROP POLICY IF EXISTS tenant_isolation ON public.t;\n"
            "  ALTER TABLE public.t DISABLE ROW LEVEL SECURITY;\n"
            "  ALTER TABLE public.t DROP COLUMN IF EXISTS tenant_id;\n"
            "END $b$;\n",
        )
        assert gate.main(["--root", str(tmp_path)]) == 0  # type: ignore[attr-defined]

    def test_a_bare_drop_in_a_block_still_fails(
        self, gate: object, tmp_path: Path
    ) -> None:
        """The negative control: the original hazard is still refused."""
        self._write(
            tmp_path,
            "DO $b$ BEGIN\n"
            "  DROP POLICY IF EXISTS tenant_isolation ON public.t;\n"
            "END $b$;\n",
        )
        assert gate.main(["--root", str(tmp_path)]) == 1  # type: ignore[attr-defined]

    def test_a_disable_placed_before_the_drop_still_fails(
        self, gate: object, tmp_path: Path
    ) -> None:
        """Order is load-bearing: the committed state is what the LAST statement leaves."""
        self._write(
            tmp_path,
            "DO $b$ BEGIN\n"
            "  ALTER TABLE public.t DISABLE ROW LEVEL SECURITY;\n"
            "  DROP POLICY IF EXISTS tenant_isolation ON public.t;\n"
            "END $b$;\n",
        )
        assert gate.main(["--root", str(tmp_path)]) == 1  # type: ignore[attr-defined]

    def test_a_reenable_after_the_disable_still_fails(
        self, gate: object, tmp_path: Path
    ) -> None:
        """Re-arming enforcement inside the block re-opens the window."""
        self._write(
            tmp_path,
            "DO $b$ BEGIN\n"
            "  DROP POLICY IF EXISTS tenant_isolation ON public.t;\n"
            "  ALTER TABLE public.t DISABLE ROW LEVEL SECURITY;\n"
            "  ALTER TABLE public.t ENABLE ROW LEVEL SECURITY;\n"
            "END $b$;\n",
        )
        assert gate.main(["--root", str(tmp_path)]) == 1  # type: ignore[attr-defined]

    def test_a_disable_on_a_DIFFERENT_relation_does_not_excuse_the_drop(
        self, gate: object, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path,
            "DO $b$ BEGIN\n"
            "  DROP POLICY IF EXISTS tenant_isolation ON public.t;\n"
            "  ALTER TABLE public.other DISABLE ROW LEVEL SECURITY;\n"
            "END $b$;\n",
        )
        assert gate.main(["--root", str(tmp_path)]) == 1  # type: ignore[attr-defined]

    def test_the_two_omn18774_migrations_pass_on_their_real_bytes(
        self, gate: object
    ) -> None:
        """The shape this arm admits is the shape that actually landed."""
        for relative in (
            "nodes/node_projection_delegation/"
            "0043_generation_events_drop_tenant_posture.sql",
            "nodes/node_projection_registration/"
            "0007_node_service_registry_drop_tenant_posture.sql",
        ):
            path = _FORWARD / relative
            assert path.is_file(), relative
            assert (
                gate.violations_for(  # type: ignore[attr-defined]
                    path, path.read_text(encoding="utf-8")
                )
                == []
            )
