# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

r"""OMN-14899 re-proof — tenant isolation, asserted under the REAL app role.

WHY THIS EXISTS
---------------
OMN-14899's earlier isolation matrix went green while the session was
effectively exempt from row-level security. Postgres bypasses RLS
unconditionally for SUPERUSER, for BYPASSRLS, and for a table's OWNER, so an
isolation suite run under any of those returns exactly the same PASS whether
the policies are correct, wrong, or absent. That is not a weak proof — it is a
proof of nothing, and it is indistinguishable from a real one by looking at the
result.

Operator ruling 28 (2026-07-31) keeps OMN-14899 open on exactly this: merged is
not proven, and the acceptance criterion naming the *running application's*
connection was never satisfied.

HOW THE VACUITY IS MADE STRUCTURALLY IMPOSSIBLE
-----------------------------------------------
Every isolation assertion in this module depends on the ``proven_role``
fixture, and that fixture is an ADMISSION GATE, not a setup step. It refuses to
yield a connection until it has read, IN THIS SESSION, that:

1. ``current_user`` and ``session_user`` are both the expected read role. Not a
   role that can ``SET ROLE`` to it — ``session_user`` pins the authenticated
   identity, so a superuser that did ``SET ROLE app_dashboard`` is rejected.
   (``SET ROLE`` does not clear BYPASSRLS inherited from the session role.)
2. ``rolsuper`` and ``rolbypassrls`` are both false, read ``WHERE rolname =
   current_user`` — keyed on the live session identity, never on the literal
   role name, so the check cannot be satisfied by reading some *other* role's
   catalog row.
3. The connecting role owns zero tables in the schema under test. An owner is
   exempt from RLS with FORCE included.
4. The table under test actually carries ``relrowsecurity``,
   ``relforcerowsecurity`` and a policy named ``tenant_isolation``. A "denied"
   result against a table with no policy proves nothing about isolation.
5. Both tenants have rows the role could in principle see. A cross-tenant read
   returning zero is only evidence of DENIAL if the data exists to be denied
   (memory: prove RED against exists-but-wrong, not against empty).

A failure in the gate is a hard ``pytest.fail``, never a skip: a proof that
quietly degrades into a weaker proof is the failure mode this module exists to
remove.

HOW TO RUN IT
-------------
Read-only. It issues no ``INSERT``/``UPDATE``/``DELETE``/DDL and creates no
rows, so it is safe against a live lane — but it is NOT run automatically. It
skips unless a DSN is supplied, so CI never executes it:

    OMN14899_APP_DASHBOARD_DSN='postgresql://app_dashboard:...@host:5432/omnidash_analytics?sslmode=verify-full' \
    OMN14899_TENANT_A='<tenant with rows>' \
    OMN14899_TENANT_B='<different tenant with rows>' \
    uv run pytest tests/integration/db/test_omn14899_app_dashboard_reproof.py -v

The DSN must authenticate AS the read role. Pointing it at a superuser is not a
shortcut — the gate rejects it and names why.

Never paste the DSN into a PR body, ticket, or commit. The credential is
deployment-owned (AWS Secrets Manager ``omninode/staging/rds/app-dashboard``).

WHAT THIS DOES NOT PROVE
------------------------
It proves the ROLE and the POLICIES hold under a real pooled connection. It
does NOT by itself prove the *deployed application* connects as this role —
that is OMN-15358 (workload wiring), and the honest way to close it is to point
this harness at the DSN the running workload actually resolves, then record
which manifest/secret that DSN came from alongside the result.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from typing import Any

import psycopg2
import psycopg2.pool
import pytest

# The role whose identity every assertion below is conditioned on.
EXPECTED_READ_ROLE = "app_dashboard"

# The table OMN-14899's evidence trail uses. Pinned as a constant rather than
# read from an env var: an operator-supplied table name could silently point
# the proof at a table with no policy, and the gate's job is to make that
# class of substitution impossible rather than merely detectable.
TARGET_SCHEMA = "public"
TARGET_TABLE = "delegation_events"
TENANT_GUC = "app.tenant_id"

_DSN_ENV = "OMN14899_APP_DASHBOARD_DSN"
_TENANT_A_ENV = "OMN14899_TENANT_A"
_TENANT_B_ENV = "OMN14899_TENANT_B"


def _qualified() -> str:
    return f"{TARGET_SCHEMA}.{TARGET_TABLE}"


@pytest.fixture(scope="module")
def dsn() -> str:
    value = os.environ.get(_DSN_ENV)
    if not value:
        pytest.skip(
            f"{_DSN_ENV} not set — the OMN-14899 re-proof runs only against an "
            "explicitly supplied live DSN, never implicitly in CI"
        )
    return value


@pytest.fixture(scope="module")
def tenants() -> tuple[str, str]:
    """Two distinct tenant ids, both required.

    Deliberately no defaults. A silent default here would make a cross-tenant
    assertion pass against a tenant that does not exist, which is the same
    vacuity in a different costume.
    """
    tenant_a = os.environ.get(_TENANT_A_ENV)
    tenant_b = os.environ.get(_TENANT_B_ENV)
    missing = [
        name
        for name, value in ((_TENANT_A_ENV, tenant_a), (_TENANT_B_ENV, tenant_b))
        if not value
    ]
    if missing:
        pytest.fail(
            f"{_DSN_ENV} is set but {', '.join(missing)} is not. The isolation "
            "legs need two real, distinct tenants; running with one would "
            "reduce the cross-tenant check to a tautology."
        )
    assert tenant_a is not None and tenant_b is not None
    if tenant_a == tenant_b:
        pytest.fail(
            f"{_TENANT_A_ENV} and {_TENANT_B_ENV} are the same value "
            f"({tenant_a!r}) — a cross-tenant denial cannot be observed "
            "between a tenant and itself"
        )
    return tenant_a, tenant_b


def _scalar(cursor: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
    cursor.execute(sql, params)
    row = cursor.fetchone()
    assert row is not None, f"query returned no row: {sql}"
    return row[0]


def _assert_role_identity(cursor: Any) -> None:
    """Gate step 1+2 — the connecting identity, read in this very session."""
    cursor.execute("SELECT current_user, session_user")
    identity = cursor.fetchone()
    assert identity is not None
    current_user, session_user = identity

    if session_user != EXPECTED_READ_ROLE:
        pytest.fail(
            f"session_user is {session_user!r}, expected {EXPECTED_READ_ROLE!r}. "
            "The DSN must AUTHENTICATE as the read role. A privileged session "
            "that reached the role via SET ROLE is rejected on purpose: SET "
            "ROLE does not drop BYPASSRLS held by the session role, so every "
            "isolation result below would be vacuous."
        )
    if current_user != EXPECTED_READ_ROLE:
        pytest.fail(
            f"current_user is {current_user!r} but session_user is "
            f"{session_user!r} — the session has switched roles mid-flight; "
            "refusing to certify isolation under an ambiguous identity"
        )

    # Keyed on current_user, NOT on the literal role name: reading
    # `WHERE rolname = 'app_dashboard'` would report the flags of a role that
    # might not be the one this session is running as.
    cursor.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    )
    flags = cursor.fetchone()
    if flags is None:
        pytest.fail(
            "pg_roles has no row for current_user — cannot certify that this "
            "session is not RLS-exempt"
        )
    rolsuper, rolbypassrls = flags
    if rolsuper or rolbypassrls:
        pytest.fail(
            f"the connected role {current_user!r} has rolsuper={rolsuper}, "
            f"rolbypassrls={rolbypassrls}. Postgres bypasses row-level "
            "security unconditionally for such a role, so every isolation "
            "assertion below would pass regardless of whether the policies "
            "are correct. This is the exact vacuity that made the earlier "
            "OMN-14899 green worthless."
        )


def _assert_not_owner(cursor: Any) -> None:
    """Gate step 3 — ownership is an RLS exemption FORCE does not close."""
    cursor.execute(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = %s AND tableowner = current_user "
        "ORDER BY tablename",
        (TARGET_SCHEMA,),
    )
    owned = [row[0] for row in cursor.fetchall()]
    if owned:
        pytest.fail(
            f"the connected role owns table(s) {owned} in {TARGET_SCHEMA}. An "
            "owner is exempt from row-level security (FORCE included), so any "
            "isolation reading taken from them is a false clean."
        )


def _assert_table_is_actually_protected(cursor: Any) -> None:
    """Gate step 4 — a denial against an unprotected table proves nothing."""
    cursor.execute(
        "SELECT c.relrowsecurity, c.relforcerowsecurity, "
        "       EXISTS (SELECT 1 FROM pg_policy p "
        "                WHERE p.polrelid = c.oid AND p.polname = 'tenant_isolation') "
        "  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        " WHERE n.nspname = %s AND c.relname = %s AND c.relkind = 'r'",
        (TARGET_SCHEMA, TARGET_TABLE),
    )
    row = cursor.fetchone()
    if row is None:
        pytest.fail(
            f"{_qualified()} is not visible to the connected role — cannot "
            "certify isolation against a table this session cannot even "
            "resolve in the catalog"
        )
    enabled, forced, has_policy = row
    if not (enabled and has_policy):
        pytest.fail(
            f"{_qualified()} has relrowsecurity={enabled}, "
            f"tenant_isolation policy present={has_policy}. Zero-row results "
            "from an unprotected table are just an empty table, not isolation."
        )
    if not forced:
        # Not fatal for a NON-owner (which the gate already established this
        # role is), but recorded: FORCE is what protects the OWNER's own reads,
        # and its absence changes what a future writer-side proof would mean.
        warnings.warn(
            f"{_qualified()} has relforcerowsecurity=False. The connected role "
            "is a non-owner so its reads are still policed, but the owner's "
            "are not — do not cite this run as evidence about writer-side "
            "isolation.",
            stacklevel=2,
        )


def _assert_both_tenants_have_rows(cursor: Any, tenants: tuple[str, str]) -> None:
    """Gate step 5 — denial is only observable where data exists to deny."""
    for tenant in tenants:
        cursor.execute(f"SET {TENANT_GUC} = %s", (tenant,))
        visible = _scalar(cursor, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
        if visible == 0:
            pytest.fail(
                f"tenant {tenant!r} has zero visible rows in {_qualified()}. "
                "Every cross-tenant assertion below would then return zero for "
                "the trivial reason that the data is absent, not because the "
                "policy denied it. Point the harness at tenants that have rows."
            )
    cursor.execute(f"RESET {TENANT_GUC}")


@pytest.fixture(scope="module")
def proven_role(dsn: str, tenants: tuple[str, str]) -> Iterator[Any]:
    """A connection that has PROVEN it is the constrained role before yielding.

    The gate runs first and fails hard. No isolation assertion in this module
    can execute against an unproven identity, because none of them can obtain a
    connection except through here.
    """
    try:
        conn = psycopg2.connect(dsn)
    except psycopg2.OperationalError as exc:
        pytest.fail(
            f"could not open a session with {_DSN_ENV}: {exc}\n"
            "If this reads 'permission denied for database ... does not have "
            "CONNECT privilege', that is OMN-15297 — apply forward migration "
            "097_grant_app_dashboard_connect_omnidash_analytics.sql."
        )

    conn.autocommit = True
    with conn.cursor() as cur:
        _assert_role_identity(cur)
        _assert_not_owner(cur)
        _assert_table_is_actually_protected(cur)
        _assert_both_tenants_have_rows(cur, tenants)

    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Leg 0 — the gate itself, stated as a test so the evidence names it.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_leg0_connected_identity_is_the_constrained_read_role(
    proven_role: Any,
) -> None:
    """Records the identity the other legs are conditioned on.

    If this test is absent from a run's output, the other legs' results mean
    nothing — that is the point of stating it separately rather than leaving it
    implicit in a fixture.
    """
    with proven_role.cursor() as cur:
        cur.execute(
            "SELECT current_user, session_user, "
            "       (SELECT rolsuper FROM pg_roles WHERE rolname = current_user), "
            "       (SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user)"
        )
        row = cur.fetchone()

    assert row is not None
    current_user, session_user, rolsuper, rolbypassrls = row
    print(
        f"[OMN-14899] current_user={current_user} session_user={session_user} "
        f"rolsuper={rolsuper} rolbypassrls={rolbypassrls}"
    )
    assert current_user == EXPECTED_READ_ROLE
    assert session_user == EXPECTED_READ_ROLE
    assert rolsuper is False
    assert rolbypassrls is False


# =============================================================================
# Leg 1 — positive same-tenant access.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_leg1_same_tenant_rows_are_visible(
    proven_role: Any, tenants: tuple[str, str]
) -> None:
    """With the tenant context set, the role sees that tenant's rows — and ONLY
    that tenant's rows.

    The second half matters as much as the first: a policy that lets everything
    through also satisfies "sees its own rows".
    """
    tenant_a, _ = tenants
    with proven_role.cursor() as cur:
        cur.execute(f"SET {TENANT_GUC} = %s", (tenant_a,))
        visible = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
        foreign = _scalar(
            cur,
            f"SELECT count(*) FROM {_qualified()} WHERE tenant_id <> %s",  # noqa: S608 - identifier is a module constant, tenants are bound params
            (tenant_a,),
        )
        cur.execute(f"RESET {TENANT_GUC}")

    assert visible > 0, (
        f"tenant {tenant_a!r} sees zero rows with its own context set — the "
        "policy is denying the tenant its own data"
    )
    assert foreign == 0, (
        f"tenant {tenant_a!r} can see {foreign} row(s) belonging to other "
        "tenants — the tenant_isolation policy is not constraining reads"
    )


# =============================================================================
# Leg 2 — cross-tenant denial.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_leg2_other_tenants_rows_are_denied(
    proven_role: Any, tenants: tuple[str, str]
) -> None:
    """Tenant A's context must not reach tenant B's rows.

    The gate already proved tenant B HAS rows, so a zero here is denial rather
    than emptiness. Both directions are checked: a policy broken in one
    direction only is still broken.
    """
    tenant_a, tenant_b = tenants
    with proven_role.cursor() as cur:
        cur.execute(f"SET {TENANT_GUC} = %s", (tenant_a,))
        b_from_a = _scalar(
            cur,
            f"SELECT count(*) FROM {_qualified()} WHERE tenant_id = %s",  # noqa: S608 - identifier is a module constant, tenants are bound params
            (tenant_b,),
        )
        cur.execute(f"SET {TENANT_GUC} = %s", (tenant_b,))
        a_from_b = _scalar(
            cur,
            f"SELECT count(*) FROM {_qualified()} WHERE tenant_id = %s",  # noqa: S608 - identifier is a module constant, tenants are bound params
            (tenant_a,),
        )
        cur.execute(f"RESET {TENANT_GUC}")

    assert b_from_a == 0, (
        f"context={tenant_a!r} can read {b_from_a} row(s) of tenant "
        f"{tenant_b!r} — cross-tenant read"
    )
    assert a_from_b == 0, (
        f"context={tenant_b!r} can read {a_from_b} row(s) of tenant "
        f"{tenant_a!r} — cross-tenant read"
    )


# =============================================================================
# Leg 3 — unset context denies everything.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_leg3_unset_tenant_context_returns_zero_rows(proven_role: Any) -> None:
    """Fail-closed, with no default-tenant fallback.

    ``current_setting('app.tenant_id', true)`` is NULL when the GUC is unset,
    the policy predicate is NULL, and nothing is visible. This is OMN-14899's
    named negative proof, and it is the one that distinguishes a real policy
    from a permissive one.

    Both RESET and an explicit empty string are exercised: an application that
    "clears" the context by writing '' rather than resetting it must not
    thereby match a row whose tenant_id is ''.
    """
    with proven_role.cursor() as cur:
        cur.execute(f"RESET {TENANT_GUC}")
        after_reset = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
        raw = _scalar(cur, f"SELECT current_setting('{TENANT_GUC}', true)")

        cur.execute(f"SET {TENANT_GUC} = ''")
        after_empty = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
        cur.execute(f"RESET {TENANT_GUC}")

    assert after_reset == 0, (
        f"{after_reset} row(s) visible with {TENANT_GUC} unset "
        f"(current_setting returned {raw!r}) — the policy is not fail-closed "
        "and an un-scoped connection reads across every tenant"
    )
    assert after_empty == 0, (
        f"{after_empty} row(s) visible with {TENANT_GUC} set to the empty "
        "string — an application clearing context by writing '' would read "
        "another tenant's data"
    )


# =============================================================================
# Leg 4 — pooled-connection reuse.
# =============================================================================


@pytest.mark.integration
@pytest.mark.postgres
def test_leg4_pooled_connection_reuse_does_not_leak_tenant_context(
    proven_role: Any, dsn: str, tenants: tuple[str, str]
) -> None:
    """A recycled physical connection must not carry the previous checkout's
    tenant context.

    ``app.tenant_id`` is a session GUC. A pool hands the SAME backend to the
    next caller, so a session-level ``SET`` that is never reset survives the
    checkout boundary — and the next request, possibly for a different tenant
    or for no tenant at all, inherits it. That is a cross-tenant read produced
    entirely by connection management, with correct policies and a correct
    role.

    ``minconn=maxconn=1`` forces reuse, and ``pg_backend_pid()`` is compared
    across checkouts so REUSE ITSELF IS PROVEN rather than assumed — without
    that comparison a pool that quietly opened a second backend would make this
    test pass while testing nothing.

    ``proven_role`` is a parameter (not merely ordering) so the identity gate is
    a hard precondition of this leg too.
    """
    assert proven_role is not None
    tenant_a, tenant_b = tenants

    pool = psycopg2.pool.SimpleConnectionPool(1, 1, dsn)
    try:
        first = pool.getconn()
        first.autocommit = True
        with first.cursor() as cur:
            first_pid = _scalar(cur, "SELECT pg_backend_pid()")
            cur.execute(f"SET {TENANT_GUC} = %s", (tenant_a,))
            seen_a = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
        pool.putconn(first)

        assert seen_a > 0, "setup: tenant A must see its own rows on checkout 1"

        second = pool.getconn()
        second.autocommit = True
        with second.cursor() as cur:
            second_pid = _scalar(cur, "SELECT pg_backend_pid()")
            leaked_setting = _scalar(
                cur, f"SELECT current_setting('{TENANT_GUC}', true)"
            )
            unscoped_rows = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params

            cur.execute(f"SET {TENANT_GUC} = %s", (tenant_b,))
            b_rows = _scalar(cur, f"SELECT count(*) FROM {_qualified()}")  # noqa: S608 - identifier is a module constant, tenants are bound params
            a_rows_from_b = _scalar(
                cur,
                f"SELECT count(*) FROM {_qualified()} WHERE tenant_id = %s",  # noqa: S608 - identifier is a module constant, tenants are bound params
                (tenant_a,),
            )
            cur.execute(f"RESET {TENANT_GUC}")
        pool.putconn(second)
    finally:
        pool.closeall()

    assert first_pid == second_pid, (
        f"the pool handed out two different backends ({first_pid} then "
        f"{second_pid}) — connection reuse did not occur, so this leg proved "
        "nothing about pooled context leakage. Re-run against a pool that "
        "actually recycles."
    )
    assert unscoped_rows == 0, (
        f"a recycled connection returned {unscoped_rows} row(s) before the new "
        f"caller set any tenant context (current_setting={leaked_setting!r}) — "
        f"tenant {tenant_a!r}'s context leaked across the checkout boundary"
    )
    assert b_rows > 0, "tenant B must see its own rows on the recycled connection"
    assert a_rows_from_b == 0, (
        f"the recycled connection, scoped to tenant {tenant_b!r}, can read "
        f"{a_rows_from_b} row(s) of tenant {tenant_a!r}"
    )
