# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18060 — static pins on the runner's least-privilege GRANT seam.

The live proofs are ``tests/scripts/test_login_only_role_grants_live_omn18060.py``;
they need a scratch Postgres and SKIP where ``initdb``/``pg_ctl`` are absent.
These assertions need no database, so they gate every PR — a seam whose only
proof skips on the CI image is a seam with no proof.

What they hold in place:

* the grant map may only express CONNECT + schema USAGE + column-scoped SELECT,
  never a write privilege, never ``ALL TABLES``, never a REVOKE;
* every principal it grants to is one whose LOGIN credential this deployment
  also owns (authorization without provenance is the thing being avoided);
* the relation-existence gate and the post-grant readback are both present;
* the OMN-16993 credential phase stays GRANT-free — the new seam is beside it,
  not inside it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

BEGIN_MARKER = "# ---- BEGIN login-only role grant seam (OMN-18060) ----"
END_MARKER = "# ---- END login-only role grant seam (OMN-18060) ----"

CANARY_ROLE = "chain_canary_reader"
CANARY_RELATION = "public.delegation_workflow_state"
CANARY_COLUMNS = ("correlation_id", "state")
# Present on the same relation and deliberately outside the grant: the
# delegation's own request/response material, and the tenant discriminator.
WITHHELD_COLUMNS = ("payload", "tenant_id")


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _executable_lines(text: str) -> list[str]:
    """Comment lines are prose, not behaviour, and this file's own rationale
    names every forbidden token it forbids."""
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _grant_seam() -> str:
    lines = _runner_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == BEGIN_MARKER]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == END_MARKER]
    assert len(starts) == 1 and len(ends) == 1 and ends[0] > starts[0], (
        "expected exactly one OMN-18060 grant seam delimited by its markers "
        f"(found {len(starts)} begin / {len(ends)} end). The markers are what "
        "the RED control in the live proofs strips; without them that control "
        "cannot be derived and silently stops being a control."
    )
    return "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"


def _grant_map_entries() -> list[str]:
    seam = _grant_seam()
    start = seam.index("for grant_role_entry in") + len("for grant_role_entry in")
    body = seam[start : seam.index("; do", start)]
    return re.findall(r'"([a-z0-9_]+:[a-z0-9_]+\.[a-z0-9_]+:[a-z0-9_,]+)"', body)


def _credential_map_entries() -> list[str]:
    text = _runner_text()
    start = text.index("for login_role_entry in") + len("for login_role_entry in")
    body = text[start : text.index("; do", start)]
    return re.findall(r'"([a-z_]+):([A-Z0-9_]+)"', body)


# ---------------------------------------------------------------------------
# What the map is allowed to say
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_grant_map_declares_the_canary_reader_column_scoped() -> None:
    entries = _grant_map_entries()
    assert entries, "LOGIN_ONLY_ROLE_GRANT_MAP parsed empty"

    expected = f"{CANARY_ROLE}:{CANARY_RELATION}:{','.join(CANARY_COLUMNS)}"
    assert expected in entries, (
        f"the grant map must declare {expected!r}; without it the chain "
        "canary's OMN-16025 link-2 readback has an identity and no authorization"
    )


@pytest.mark.unit
@pytest.mark.parametrize("column", WITHHELD_COLUMNS)
def test_grant_map_never_names_a_withheld_column(column: str) -> None:
    """``payload`` carries every tenant's delegation request and response.

    A scheduled CI probe closing a liveness gate needs one enum-ish string. A
    relation-wide grant would satisfy the canary identically and hand it all of
    this, which is why the map is column-scoped and why this pin exists.
    """
    for entry in _grant_map_entries():
        columns = entry.rsplit(":", 1)[1].split(",")
        assert column not in columns, (
            f"grant map entry {entry!r} names {column}, which is deliberately "
            "outside every declared grant on this relation"
        )


@pytest.mark.unit
def test_every_granted_role_is_a_role_this_deployment_also_provisions() -> None:
    """A grant to a principal whose credential we do not own has no provenance."""
    credential_roles = {role for role, _ in _credential_map_entries()}
    assert credential_roles, "runner LOGIN_ONLY_ROLE_MAP parsed empty"

    granted_roles = {entry.split(":", 1)[0] for entry in _grant_map_entries()}
    orphans = granted_roles - credential_roles
    assert not orphans, (
        f"{sorted(orphans)} receive grants from this seam but their LOGIN "
        "credential is not minted by LOGIN_ONLY_ROLE_MAP — this deployment "
        "would be authorizing an identity it does not provision"
    )


# ---------------------------------------------------------------------------
# What the seam is allowed to do
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "forbidden",
    [
        "INSERT",
        "UPDATE",
        "DELETE",
        "TRUNCATE",
        "ALL TABLES",
        "ALL PRIVILEGES",
        "ALL SEQUENCES",
        "WITH GRANT OPTION",
        "REVOKE",
        "CREATEROLE",
        "CREATEDB",
        "ALTER ROLE",
        "CREATE ROLE",
        "PASSWORD",
    ],
)
def test_grant_seam_cannot_express_a_write_or_a_credential(forbidden: str) -> None:
    """The seam issues read authorization and nothing else.

    ``CREATE``/``ALTER ROLE`` and ``PASSWORD`` are on this list for the mirror
    reason the credential seam refuses GRANT: the two halves stay separable, so
    neither can quietly grow into the other.
    """
    seam = "\n".join(_executable_lines(_grant_seam()))
    assert forbidden not in seam, (
        f"the OMN-18060 grant seam contains {forbidden!r}; it may issue only "
        "CONNECT, schema USAGE and column-scoped SELECT"
    )


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ["SUPERUSER", "BYPASSRLS"])
def test_seam_only_reads_and_reports_the_escalation_attributes(
    attribute: str,
) -> None:
    """These two are conferred only by CREATE/ALTER ROLE, both forbidden above.

    They still appear in the seam — in the ``pg_roles`` probe that detects an
    escalated role and in the message that refuses it. This pins that they
    appear ONLY there, so the refusal cannot be quietly replaced by a
    correcting mutation that needs role-administration privilege this seam
    deliberately does not hold.
    """
    for line in _executable_lines(_grant_seam()):
        if attribute not in line and attribute.lower() not in line:
            continue
        assert line.lstrip().startswith(("echo ", "-c ", "WHERE ")) or "rol" in line, (
            f"{line.strip()!r} mentions {attribute} outside a read probe or a "
            "diagnostic message"
        )


@pytest.mark.unit
def test_grant_seam_never_grants_a_relation_without_a_column_list() -> None:
    seam = "\n".join(_executable_lines(_grant_seam()))
    grants = re.findall(r"GRANT SELECT[^;]*", seam)
    assert grants, "the seam issues no SELECT grant at all"
    for grant in grants:
        assert re.search(r"GRANT SELECT\s*\(", grant), (
            f"{grant!r} is a relation-wide SELECT; the whole point of this seam "
            "is that the column list is not optional"
        )


@pytest.mark.unit
def test_grant_seam_gates_on_the_target_relation_existing() -> None:
    """The relation is created by the frozen flat stream, not by this seam.

    Without the gate a lane whose stream has not yet created it would fail the
    run — leaving ``migrations_complete`` FALSE and the migration gate
    UNHEALTHY — over a grant that is merely early.
    """
    seam = _grant_seam()
    assert "to_regclass(" in seam, (
        "the seam must probe for the relation rather than assume it; "
        "to_regclass() returns NULL instead of raising for an absent relation"
    )
    assert "does not exist yet" in seam, (
        "the skip must name its reason — an unexplained skip is "
        "indistinguishable from a seam that never ran"
    )


@pytest.mark.unit
def test_grant_seam_reads_the_outcome_back() -> None:
    """PostgreSQL does not raise for a GRANT issued without grant option.

    It emits ``WARNING: no privileges were granted`` and returns success. The
    readback is the only thing that separates a real grant from that no-op.
    """
    seam = _grant_seam()
    for probe in (
        "has_database_privilege(",
        "has_schema_privilege(",
        "has_column_privilege(",
        "has_table_privilege(",
    ):
        assert probe in seam, f"the seam must read its outcome back via {probe}"
    assert "rolbypassrls" in seam, (
        "the seam must refuse an escalated role: SUPERUSER/BYPASSRLS exempt a "
        "role from row-level security unconditionally, so a column-scoped grant "
        "to such a role constrains nothing"
    )
    assert "relation_wide_select" in seam, (
        "has_column_privilege() is true for a relation-level grant too, so the "
        "seam must separately refuse a role that holds relation-wide SELECT — "
        "otherwise the column scoping can be defeated and still read green"
    )


# ---------------------------------------------------------------------------
# The seam is beside the credential phase, not inside it
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_credential_phase_remains_free_of_authorization() -> None:
    """OMN-16993's invariant survives this change.

    ``test_warm_volume_login_credential_omn16993`` slices the credential phase
    from its function header to the ledger banner and asserts no GRANT appears
    in it. This pins the other half — that the grant seam is genuinely outside
    that slice — so the two cannot be merged by a later edit that keeps both
    modules green by accident.
    """
    text = _runner_text()
    credential_phase = text[
        text.index("reassert_login_only_role_credential() {") : text.index(
            "Ensuring service migration ledger exists"
        )
    ]
    assert BEGIN_MARKER not in credential_phase
    assert "GRANT " not in credential_phase
    assert text.index(BEGIN_MARKER) > text.index(
        "Ensuring service migration ledger exists"
    )


@pytest.mark.unit
def test_grant_seam_runs_after_the_flat_apply_loop_and_before_the_sentinel() -> None:
    """Ordering is the difference between converging on run 1 and on run 2.

    The relation is created by the flat apply loop, and a failure in the seam
    must still leave the sentinel FALSE.
    """
    text = _runner_text()
    apply_loop = text.index(
        'echo "[forward-migration] Complete: ${APPLIED} infra applied'
    )
    sentinel = text.index("All migrations complete. Setting sentinel TRUE")
    seam = text.index(BEGIN_MARKER)
    assert apply_loop < seam < sentinel, (
        "the grant seam must run after the flat migrations that create its "
        "target relation and before the sentinel that flips the migration gate "
        "HEALTHY"
    )


@pytest.mark.unit
def test_grant_seam_never_echoes_a_credential_variable() -> None:
    """The runner's stdout is captured by ``docker compose logs`` on every lane."""
    seam = _grant_seam()
    for secret_var in ("role_password", "escaped_password", "PASSWORD"):
        assert f"${secret_var}" not in seam, (
            f"the grant seam references ${secret_var}; it has no business "
            "touching credential material at all"
        )
