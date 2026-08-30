# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16993: the warm-volume migration runner must re-assert the LOGIN credential.

``docker/migrations/forward/000_create_multiple_databases.sh`` mints the LOGIN +
password for every principal in its ``LOGIN_ONLY_ROLE_MAP`` (Phase 2b, OMN-16843).
Postgres runs that file from ``/docker-entrypoint-initdb.d`` **only when the data
directory is empty** — a fresh volume. ``scripts/run-forward-migrations.sh``, the
one path that runs on every compose up, applied only ``*.sql``.

Between those two facts sat the defect: ``099_create_omninode_internal_live_events.sql``
creates ``omninode_runtime`` ``NOLOGIN`` on purpose (094's invariant keeps credential
material out of migrations), so on any warm volume the role existed with
``rolcanlogin = false`` and ``rolpassword IS NULL`` while ``OMNINODE_INTERNAL_DB_URL``
resolved perfectly and then failed at connect.

Observed on the stability lane 2026-08-29: its postgres container was recreated
2026-08-28T20:14Z on a pre-existing volume, ``pg_authid`` reported
``omninode_runtime | f | f``, and ``node_projection_session_replay`` DLQ'd 100% of
its topic on ``password authentication failed`` at ~26 failures/second while the
runtime reported ``healthy`` and offsets kept committing.

These assertions are static by design — they fire on hosts with no Docker and no
Postgres, which is exactly where a silent revert would otherwise survive until the
next lane rediscovers it by hand. The live readback is the ticket's own proof.

Ticket: OMN-16993. Compose-wiring prerequisite: OMN-16843. Identity epic: OMN-15426.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCKER_DIR = REPO_ROOT / "docker"
BASE_COMPOSE = DOCKER_DIR / "docker-compose.infra.yml"
JUDGE_COMPOSE = DOCKER_DIR / "docker-compose.judge.yml"
BOOTSTRAP = DOCKER_DIR / "migrations" / "forward" / "000_create_multiple_databases.sh"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

RUNTIME_PRINCIPAL = "omninode_runtime"
RUNTIME_PASSWORD_VAR = "OMNINODE_RUNTIME_PASSWORD"
MIGRATION_SERVICE = "forward-migration"

# Same coverage set as ``test_compose_lane_internal_dsn_omn16843.py``: the base
# file is merged first for dev, stability-test and prod, and judge redeclares the
# service wholesale so it inherits nothing from the base.
LANE_COMPOSE_FILES = (BASE_COMPOSE, JUDGE_COMPOSE)


def _load_compose(path: Path) -> dict[str, Any]:
    """Parse a compose file, tolerating compose's ``!override`` tag."""
    text = path.read_text(encoding="utf-8").replace("!override", "")
    doc = yaml.safe_load(text) or {}
    assert isinstance(doc, dict)
    return doc


def _migration_service_env(path: Path) -> dict[str, Any]:
    service = (_load_compose(path).get("services") or {})[MIGRATION_SERVICE]
    env = service.get("environment") or {}
    assert isinstance(env, dict)
    return env


def _executable_text(path: Path) -> str:
    """The script with ``#`` comment lines removed.

    Both files document the very map membership asserted below, so scanning raw
    text would make these assertions unfalsifiable.
    """
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _quoted_entries(text: str, marker: str, terminator: str) -> list[str]:
    """The double-quoted ``role:VAR`` entries between *marker* and *terminator*."""
    start = text.index(marker) + len(marker)
    body = text[start : text.index(terminator, start)]
    return re.findall(r'"([a-z_]+:[A-Z0-9_]+)"', body)


def _bootstrap_login_only_entries() -> list[str]:
    return _quoted_entries(_executable_text(BOOTSTRAP), "LOGIN_ONLY_ROLE_MAP=(", ")")


def _runner_login_only_entries() -> list[str]:
    return _quoted_entries(_executable_text(RUNNER), "for login_role_entry in", "; do")


# ---------------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_reasserts_the_login_credential_on_warm_volumes() -> None:
    """The runner must issue the LOGIN + PASSWORD attach, not merely read it.

    An ``ALTER ROLE ... LOGIN PASSWORD`` is the whole point: the role already
    exists on a warm volume (099 created it NOLOGIN), so a ``CREATE ROLE``-only
    seam would be a no-op on precisely the lanes that are broken.
    """
    text = _executable_text(RUNNER)

    assert RUNTIME_PASSWORD_VAR in text, (
        f"{RUNNER.name} must consume {RUNTIME_PASSWORD_VAR}; without it a warm "
        f"volume keeps {RUNTIME_PRINCIPAL} NOLOGIN forever"
    )
    assert re.search(r"ALTER ROLE\s+\"\$\{role_name\}\"\s+WITH LOGIN PASSWORD", text), (
        f"{RUNNER.name} must ALTER an existing role to LOGIN — the broken lanes "
        "already have the role, only the credential is missing"
    )
    assert "CREATE ROLE" in text, (
        f"{RUNNER.name} must also cover the never-provisioned case"
    )


@pytest.mark.unit
def test_runner_login_map_matches_the_bootstrap_map() -> None:
    """Drift guard: the fresh-volume and warm-volume seams provision the same set.

    Two lists in two languages (bash array / POSIX ``for``) is the cost of the
    runner being ``sh`` in a ``postgres:16-alpine`` sidecar. This pins them so a
    principal added to one is a test failure until it is added to the other —
    the alternative is a role that authenticates on fresh lanes and silently
    does not on warm ones, which is the class of defect this ticket exists for.
    """
    bootstrap_entries = _bootstrap_login_only_entries()
    runner_entries = _runner_login_only_entries()

    assert bootstrap_entries, "bootstrap LOGIN_ONLY_ROLE_MAP parsed empty"
    assert f"{RUNTIME_PRINCIPAL}:{RUNTIME_PASSWORD_VAR}" in bootstrap_entries
    assert sorted(runner_entries) == sorted(bootstrap_entries), (
        f"{RUNNER.name} provisions {sorted(runner_entries)} but "
        f"{BOOTSTRAP.name} provisions {sorted(bootstrap_entries)} — a principal "
        "in only one list authenticates on fresh volumes and not on warm ones"
    )


@pytest.mark.unit
def test_runner_never_widens_the_principal_beyond_login() -> None:
    """AC4 parity with the bootstrap: LOGIN + PASSWORD and nothing else.

    ``GRANT USAGE, CREATE ON SCHEMA public`` would let the role OWN tables, and
    Postgres exempts a table's owner from row-level security unconditionally —
    FORCE included. Authorization belongs to the topology instance and the
    topology-derived migrations; this seam must never issue it.
    """
    text = _executable_text(RUNNER)
    phase = text[
        text.index("reassert_login_only_role_credential() {") : text.index(
            "Ensuring service migration ledger exists"
        )
    ]

    for forbidden in ("GRANT ", "REVOKE ", "SUPERUSER", "BYPASSRLS", "CREATEROLE"):
        if forbidden in ("SUPERUSER", "BYPASSRLS", "CREATEROLE"):
            # Permitted only in their negated form on the CREATE branch.
            assert not re.search(rf"(?<!NO){forbidden}", phase), (
                f"credential seam must not grant {forbidden} to {RUNTIME_PRINCIPAL}"
            )
        else:
            assert forbidden not in phase, (
                f"credential seam must issue no {forbidden.strip()} — authorization "
                "is owned by the topology-derived migrations"
            )


@pytest.mark.unit
def test_runner_never_echoes_the_credential() -> None:
    """The value reaches psql on stdin only: never argv, never a log line.

    The runner's output is captured by ``docker compose logs`` on every lane,
    and a lane operator reading migration output must not be handed a password.
    """
    text = _executable_text(RUNNER)

    for secret_var in ("role_password", "escaped_password"):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("echo ", "printf ")):
                continue
            # `printf '%s' "$role_password" | sed` is the escaping pipeline, not
            # output: it writes to a pipe, never to stdout.
            if "| sed" in stripped:
                continue
            assert f"${secret_var}" not in stripped, (
                f"{RUNNER.name} would print the credential: {stripped!r}"
            )


# ---------------------------------------------------------------------------
# The compose half — the runner cannot assert a credential it is never handed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", LANE_COMPOSE_FILES, ids=lambda p: p.name)
def test_migration_service_receives_the_runtime_credential(
    compose_file: Path,
) -> None:
    """Guards the seam against vacuity.

    The runner's phase skips a role whose variable is unset, so a lane that
    never passes the variable gets the old silent behaviour back with the code
    still present and the test above still green.
    """
    env = _migration_service_env(compose_file)

    assert RUNTIME_PASSWORD_VAR in env, (
        f"{compose_file.name}: the {MIGRATION_SERVICE} service must receive "
        f"{RUNTIME_PASSWORD_VAR} or it cannot re-assert the credential on a warm "
        "volume — the runtime DSN resolves and then fails at connect"
    )


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", LANE_COMPOSE_FILES, ids=lambda p: p.name)
def test_migration_service_credential_is_a_reference_not_a_literal(
    compose_file: Path,
) -> None:
    """Ruling 1: the value lives in the secret store, compose carries a REF.

    ``:-`` rather than ``:?`` is deliberate here and mirrors the postgres
    service: an unprovisioned volume must skip the role, not wedge compose
    render for the whole lane. The fail-closed form lives on the CONSUMER
    (``OMNINODE_INTERNAL_DB_URL``), where an unset value is a real defect.
    """
    value = _migration_service_env(compose_file)[RUNTIME_PASSWORD_VAR]

    assert value == "${" + RUNTIME_PASSWORD_VAR + ":-}", (
        f"{compose_file.name}: expected the empty-means-skip reference form, "
        f"got {value!r}"
    )
