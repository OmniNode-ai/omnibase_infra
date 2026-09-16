# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18438: the warm-volume runner must provision ``role_omninode`` too.

Three prior tickets closed one defect class for one principal each -- OMN-16993
(``omninode_runtime``), OMN-17138 (``tenant_projection_writer``), OMN-18060
(``chain_canary_reader``). Each time the shape was identical: the principal is
minted by ``docker/migrations/forward/000_create_multiple_databases.sh``, which
Postgres runs from ``/docker-entrypoint-initdb.d`` **only when the data directory
is empty**, so on every warm volume the DSN resolved perfectly and then failed at
connect while the consuming container reported healthy.

``role_omninode`` is the fourth. It differs in one way that matters: it is a
``SERVICE_DB_MAP`` principal, not a ``LOGIN_ONLY_ROLE_MAP`` one, so the bootstrap
gives it a LOGIN credential **and** database access on ``omninode_cloud``, which
it owns the corpus of. The warm seam must therefore deliver both -- and must do
so from its own phase, because
``test_warm_volume_login_credential_omn16993.py::test_runner_never_widens_the_principal_beyond_login``
slices the login-only credential phase and asserts it contains no ``GRANT`` at
all. That assertion is correct and must keep holding; this seam lives after it.

Measured on the .201 compose dev lane 2026-09-16: ``pg_roles`` held no
``role_omninode`` row, ``omninode_cloud`` held 0 tables against 78 in
``omnibase_infra``, and ``docker logs onex-api --since 60m`` carried 112
``password authentication failed for user "role_omninode"`` lines while the
container reported ``Up (healthy)``.

These assertions are static by design -- they fire on hosts with no Docker and no
Postgres, which is where a silent revert would otherwise survive until the next
lane rediscovers it by hand.

Ticket: OMN-18438. Blocks: OMN-18421 AC1/AC2/AC5. Epic: OMN-17530.
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

PRINCIPAL = "role_omninode"
PASSWORD_VAR = "ROLE_OMNINODE_PASSWORD"
TARGET_DATABASE = "omninode_cloud"
MIGRATION_SERVICE = "forward-migration"

# The seam's own boundary markers. The credential phase that OMN-16993 pins ends
# before these, so a reader (and that test's slice) can tell the two apart.
SEAM_BEGIN = "# ---- BEGIN service-role database access seam (OMN-18438) ----"
SEAM_END = "# ---- END service-role database access seam (OMN-18438) ----"

# The base file is merged first for dev, stability-test and prod; judge
# redeclares the service wholesale so it inherits nothing from the base. Same
# coverage set as the three precedents' tests.
LANE_COMPOSE_FILES = (BASE_COMPOSE, JUDGE_COMPOSE)


def _load_compose(path: Path) -> dict[str, Any]:
    """Parse a compose file, tolerating compose's ``!override`` tag."""
    text = path.read_text(encoding="utf-8").replace("!override", "")
    return yaml.safe_load(text) or {}


def _executable_text(path: Path) -> str:
    """The script with comment-only lines dropped.

    A comment that quotes the forbidden shape must not read as the shape. Same
    helper contract as the OMN-16993 test's.
    """
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _seam_slice() -> str:
    """The runner text between this seam's own markers.

    Fails loudly rather than returning the whole file when a marker is missing:
    a slice that silently widens to the whole script would make every assertion
    below pass for the wrong reason.
    """
    text = RUNNER.read_text(encoding="utf-8")
    assert SEAM_BEGIN in text, (
        f"{RUNNER.name} carries no {SEAM_BEGIN!r} marker -- the OMN-18438 seam is "
        "absent, so no warm volume provisions role_omninode"
    )
    assert SEAM_END in text, f"{RUNNER.name} carries no {SEAM_END!r} marker"
    return text[text.index(SEAM_BEGIN) : text.index(SEAM_END)]


# ---------------------------------------------------------------------------
# AC1 -- the runner seam exists, names the principal, and is its own phase
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_provisions_the_service_role_on_a_warm_volume() -> None:
    """AC1: the seam names the principal, its password variable and its database."""
    seam = _seam_slice()

    for token in (PRINCIPAL, PASSWORD_VAR, TARGET_DATABASE):
        assert token in seam, (
            f"{RUNNER.name}'s OMN-18438 seam does not name {token!r} -- a warm "
            f"volume would leave {PRINCIPAL} absent and onex-api unable to connect"
        )


@pytest.mark.unit
def test_seam_is_separate_from_the_login_only_credential_phase() -> None:
    """AC1: this seam must not sit inside the phase OMN-16993 forbids GRANT in.

    ``test_warm_volume_login_credential_omn16993.py`` slices from
    ``reassert_login_only_role_credential() {`` to ``Ensuring service migration
    ledger exists`` and asserts no ``GRANT`` appears there. This seam issues
    GRANTs by design, so it must begin after that slice ends.
    """
    text = RUNNER.read_text(encoding="utf-8")
    login_phase_end = text.index("Ensuring service migration ledger exists")

    assert text.index(SEAM_BEGIN) > login_phase_end, (
        "the OMN-18438 seam starts inside the OMN-16993 credential phase, whose "
        "test asserts that phase contains no GRANT -- move it after that phase"
    )


@pytest.mark.unit
def test_service_role_is_not_added_to_the_login_only_map() -> None:
    """AC1: the login-only map is pinned against the bootstrap's own map.

    ``test_warm_volume_login_credential_omn16993.py`` asserts the runner's
    ``LOGIN_ONLY_ROLE_MAP`` equals the bootstrap's. ``role_omninode`` lives in
    the bootstrap's ``SERVICE_DB_MAP`` instead, so adding it to the login-only
    loop would fail that parity assertion and would also under-provision it --
    the login-only path issues no grants at all.
    """
    text = _executable_text(RUNNER)
    login_map = text[
        text.index('"omninode_runtime:OMNINODE_RUNTIME_PASSWORD"') : text.index(
            "Ensuring service migration ledger exists"
        )
    ]

    assert f"{PRINCIPAL}:{PASSWORD_VAR}" not in login_map, (
        f"{PRINCIPAL} was added to LOGIN_ONLY_ROLE_MAP -- that map is pinned "
        "equal to the bootstrap's, and the login-only path issues no grants"
    )


@pytest.mark.unit
def test_bootstrap_still_owns_the_principal_on_a_fresh_volume() -> None:
    """AC1: the warm seam mirrors the fresh-volume seam; it does not replace it.

    A principal provisioned on only one of the two paths authenticates on fresh
    volumes and not on warm ones, or the reverse. Both must carry it.
    """
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")

    assert f'"{TARGET_DATABASE}:{PRINCIPAL}:{PASSWORD_VAR}"' in bootstrap, (
        f"{BOOTSTRAP.name}'s SERVICE_DB_MAP no longer carries {PRINCIPAL} -- the "
        "warm seam mirrors that entry and must not become its only home"
    )


# ---------------------------------------------------------------------------
# AC1 -- database access, not merely a LOGIN
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_seam_grants_connect_and_schema_access_on_the_target_database() -> None:
    """AC1: a LOGIN alone is not enough for this principal.

    The bootstrap revokes ``CONNECT`` from ``PUBLIC`` on every managed database,
    and Postgres 15+ no longer grants ``CREATE ON SCHEMA public`` to ``PUBLIC``.
    So a role with only a password can authenticate and then fail at connect, or
    connect and fail to create -- which is the corpus apply failing, one layer
    further in.
    """
    seam = _seam_slice()

    assert re.search(r'GRANT\s+CONNECT\s+ON\s+DATABASE\s+"?\$?\{?\w*\}?"?', seam), (
        f"the OMN-18438 seam issues no GRANT CONNECT on {TARGET_DATABASE}"
    )
    assert "USAGE, CREATE ON SCHEMA public" in seam, (
        "the OMN-18438 seam issues no USAGE, CREATE on schema public -- the "
        "corpus is applied by this principal, so it must be able to create"
    )


@pytest.mark.unit
def test_seam_reads_its_grants_back() -> None:
    """AC1: a GRANT that did not take must fail the run, not log success.

    A ``GRANT`` issued without grant option on the object warns and returns
    success rather than raising -- the same trap OMN-18060's seam documents.
    """
    seam = _seam_slice()

    assert "has_database_privilege" in seam, (
        "the OMN-18438 seam does not read its CONNECT grant back"
    )
    assert "has_schema_privilege" in seam, (
        "the OMN-18438 seam does not read its schema grant back"
    )


@pytest.mark.unit
def test_seam_skips_named_when_the_database_is_absent() -> None:
    """AC1: a lane without the database is a legitimate state, not a failure.

    Every lane's bootstrap creates ``omninode_cloud``, but a lane whose volume
    predates that entry has not. A named skip re-asserts on the next run; an
    unnamed failure would turn the migration gate UNHEALTHY on a lane that is
    fine.
    """
    seam = _seam_slice()

    assert "pg_database" in seam, (
        "the OMN-18438 seam does not check that the target database exists "
        "before granting on it"
    )
    assert "skip" in seam, "the OMN-18438 seam has no named-skip branch"


# ---------------------------------------------------------------------------
# AC4 -- the credential never leaks, and a malformed one fails the run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_seam_refuses_a_non_hex_credential() -> None:
    """AC4: skipping a MALFORMED credential is how a lane looks provisioned.

    Deliberately stricter than the bootstrap, which counts an invalid password
    as a skip. Same contract the OMN-16993 seam states: absent is a skip,
    malformed is a failure that leaves ``migrations_complete`` FALSE.
    """
    seam = _seam_slice()

    assert "[!0-9a-fA-F]" in seam, (
        "the OMN-18438 seam does not enforce the hex-only credential contract "
        "(openssl rand -hex 32) that both other seams enforce"
    )
    assert "return 1" in seam, (
        "the OMN-18438 seam has no non-zero refusal branch -- a malformed "
        "credential would be skipped and the lane would look provisioned"
    )


@pytest.mark.unit
def test_seam_never_echoes_or_argv_passes_the_credential() -> None:
    """AC4: the value reaches psql on stdin only -- never argv, never a log line.

    The runner's output is captured by ``docker compose logs`` on every lane,
    and ``ps`` exposes argv to every process on the host.
    """
    seam = "\n".join(
        line for line in _seam_slice().splitlines() if not line.lstrip().startswith("#")
    )

    for secret_var in ("service_role_password", "service_escaped_password"):
        for line in seam.splitlines():
            stripped = line.strip()
            if stripped.startswith(("echo ", "printf ")):
                # `printf '%s' "$var" | sed` is the escaping pipeline, not
                # output: it writes to a pipe, never to stdout.
                if "| sed" in stripped:
                    continue
                assert f"${secret_var}" not in stripped, (
                    f"{RUNNER.name} would print the credential: {stripped!r}"
                )
            if stripped.startswith("psql ") or " psql " in stripped:
                assert f"${secret_var}" not in stripped, (
                    f"{RUNNER.name} would pass the credential in argv, visible "
                    f"to ps for every process on the host: {stripped!r}"
                )


# ---------------------------------------------------------------------------
# AC2 -- the compose half: the runner cannot assert what it is never handed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", LANE_COMPOSE_FILES, ids=lambda p: p.name)
def test_migration_service_is_handed_the_password(compose_file: Path) -> None:
    """AC2: every lane's forward-migration service receives the variable.

    This is the half OMN-17138 found missing for ``tenant_projection_writer``:
    the map entry and the fail-closed consumer both landed, the compose line did
    not, and every lane logged a named skip for a year.
    """
    compose = _load_compose(compose_file)
    service = compose.get("services", {}).get(MIGRATION_SERVICE)
    assert service is not None, (
        f"{compose_file.name} declares no {MIGRATION_SERVICE} service"
    )

    environment = service.get("environment", {})
    assert PASSWORD_VAR in environment, (
        f"{compose_file.name}'s {MIGRATION_SERVICE} is not handed {PASSWORD_VAR} "
        f"-- the runner would log a named skip and {PRINCIPAL} would stay absent"
    )


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", LANE_COMPOSE_FILES, ids=lambda p: p.name)
def test_password_is_empty_means_skip_not_fail_closed(compose_file: Path) -> None:
    """AC2: empty means skip, exactly as for the three principals before it.

    A ``${VAR:?}`` form here would fail compose RENDER on every lane that has
    not provisioned the credential -- turning a provisioning gap into an outage
    of the whole project, including lanes that need nothing from this seam.
    """
    compose = _load_compose(compose_file)
    environment = compose["services"][MIGRATION_SERVICE]["environment"]
    declared = str(environment[PASSWORD_VAR])

    assert declared == f"${{{PASSWORD_VAR}:-}}", (
        f"{compose_file.name} declares {PASSWORD_VAR} as {declared!r}; the "
        f"empty-means-skip form ${{{PASSWORD_VAR}:-}} is what the three "
        "precedents use and what keeps an unprovisioned lane rendering"
    )
