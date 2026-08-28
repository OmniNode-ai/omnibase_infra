# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16843: every compose lane must inject the ``omninode_runtime_service`` DSNs.

``src/omnibase_infra/topology/instances/local.yaml`` binds
``omninode_runtime_service`` to three DSN envs — one per database:

    application      -> OMNINODE_INTERNAL_DB_URL   (principal omninode_runtime)
    omnibase_infra   -> OMNIBASE_INFRA_DB_URL      (principal role_omnibase_infra)
    omniintelligence -> OMNIINTELLIGENCE_DB_URL    (principal role_omniintelligence)

``handler_wiring._build_projection_dispatch`` resolves each binding with
``os.environ.get(binding.dsn_env, "")`` and RAISES when any resolves empty, so a
DSN the compose file never sets takes down every contract that targets that
binding. Auto-wiring is non-strict, so the runtime then reports ``healthy`` with
the projections silently unattached — the failure is invisible until someone
reads a boot log.

That is exactly what shipped: ``OMNINODE_INTERNAL_DB_URL`` was set by ZERO
compose files while both of its siblings were set by five, so 19
``database_ref: application`` contracts failed to prepare on every ``.201``
compose lane. OMN-15426's landed slice (omninode_infra#803) wired the onex-dev
KUBERNETES plane only; the compose half was never covered.

The general gate below is the AC5 half — it fires for ANY future binding whose
DSN is declared by the topology instance and forgotten in the deploy config,
rather than waiting for the next lane to rediscover it by hand. The specific
assertions are the AC4 half: the credential must name the non-superuser
``omninode_runtime`` principal and must fail closed when unprovisioned.

Static-only by design: these fire on hosts with no Docker and no Postgres, which
is where a silent revert would otherwise go unnoticed until the next lab
readback. Execution proof against a real cluster is the ticket's live readback.

Ticket: OMN-16843. Identity cutover epic: OMN-15426.
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
DEV_LANE_OVERLAY = DOCKER_DIR / "docker-compose.dev-lane.yml"
BOOTSTRAP = DOCKER_DIR / "migrations" / "forward" / "000_create_multiple_databases.sh"
REQUIRED_ENV_MANIFEST = DOCKER_DIR / "required-env-vars.manifest.txt"
LOCAL_INSTANCE = (
    REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances" / "local.yaml"
)

INTERNAL_DSN_KEY = "OMNINODE_INTERNAL_DB_URL"
RUNTIME_PRINCIPAL = "omninode_runtime"
RUNTIME_PASSWORD_VAR = "OMNINODE_RUNTIME_PASSWORD"
RUNTIME_BINDING = "omninode_runtime_service"
APPLICATION_PHYSICAL_DB = "omnidash_analytics"

# The compose files that deploy the `local` topology instance and RUN the
# projection-bearing runtime services. Both are layered onto real `.201` lanes:
# the base is merged first by `resolve_compose_file_args` for dev,
# stability-test and prod; judge rebuilds its own env anchor and so must be
# listed separately or it inherits nothing.
#
# `docker-compose.e2e.yml` also resolves the `local` instance (profile `test`)
# but is DELIBERATELY not covered here. It is a standalone, ephemeral CI lane
# that layers nothing and runs a narrower service set — it sets neither
# OMNIDASH_ANALYTICS_DB_URL nor OMNINODE_INTERNAL_DB_URL, so its application
# bindings are unwired as a matter of design, not drift (the same
# classification `tests/ci/test_env_parity.py` already applies to it in the
# reverse-walk docstring). Wiring that lane needs its own credential
# provisioning on a fresh volume and is tracked separately; asserting against
# it here would report design as a defect.
PROJECTION_LANE_COMPOSE_FILES = (BASE_COMPOSE, JUDGE_COMPOSE)


def _load_compose(path: Path) -> dict[str, Any]:
    """Parse a compose file, tolerating compose's ``!override`` tag.

    ``yaml.safe_load`` has no constructor for ``!override`` (used by the judge
    overlay on ``networks:``), and the tag carries no meaning for the questions
    asked here — only which keys a service's ``environment`` mapping declares.
    """
    text = path.read_text(encoding="utf-8").replace("!override", "")
    doc = yaml.safe_load(text) or {}
    assert isinstance(doc, dict)
    return doc


def _service_environments(path: Path) -> dict[str, dict[str, Any]]:
    services = _load_compose(path).get("services") or {}
    result: dict[str, dict[str, Any]] = {}
    for name, cfg in services.items():
        if not isinstance(cfg, dict):
            continue
        env = cfg.get("environment")
        if isinstance(env, dict):
            result[name] = env
    return result


def _runtime_service_dsn_envs() -> set[str]:
    """Every ``dsn_env`` the local instance binds to ``omninode_runtime_service``.

    Derived, never hardcoded: a new database bound to this service identity is
    picked up by the gate the moment the topology declares it.
    """
    instance = yaml.safe_load(LOCAL_INSTANCE.read_text(encoding="utf-8"))
    dsn_envs: set[str] = set()
    for database in instance["databases"].values():
        binding = (database.get("bindings") or {}).get(RUNTIME_BINDING)
        if binding:
            dsn_envs.add(binding["dsn_env"])
    return dsn_envs


def _internal_dsn(path: Path, service: str) -> str:
    value = _service_environments(path)[service][INTERNAL_DSN_KEY]
    assert isinstance(value, str)
    return value


# ---------------------------------------------------------------------------
# AC5 — the general gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_local_instance_binds_the_internal_dsn_to_the_runtime_service() -> None:
    """Anchors the gate below: if this binding is ever renamed or dropped, the
    coverage assertion must be re-derived deliberately rather than silently
    passing against an empty set."""
    dsn_envs = _runtime_service_dsn_envs()

    assert INTERNAL_DSN_KEY in dsn_envs
    assert len(dsn_envs) >= 2, (
        f"expected the runtime service to bind several databases, got {dsn_envs}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_file", PROJECTION_LANE_COMPOSE_FILES, ids=lambda p: p.name
)
def test_every_runtime_service_declares_all_runtime_binding_dsns(
    compose_file: Path,
) -> None:
    """A service that resolves ONE of the binding's DSNs must resolve them ALL.

    Partial injection is the precise shape of this defect: five services carried
    two of the three DSNs, so the runtime looked configured and every contract
    targeting the third failed to prepare.
    """
    dsn_envs = _runtime_service_dsn_envs()
    incomplete: dict[str, set[str]] = {}

    for service, env in _service_environments(compose_file).items():
        declared = dsn_envs & set(env)
        if declared and declared != dsn_envs:
            incomplete[service] = dsn_envs - declared

    assert not incomplete, (
        f"{compose_file.name}: services declare only part of the "
        f"{RUNTIME_BINDING} binding's DSN set — missing {incomplete}. "
        "handler_wiring raises at WIRING time on an empty DSN, and auto-wiring "
        "is non-strict, so the runtime will report healthy with those "
        "projections silently unattached."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_file", PROJECTION_LANE_COMPOSE_FILES, ids=lambda p: p.name
)
def test_projection_lane_actually_declares_the_internal_dsn(
    compose_file: Path,
) -> None:
    """Guards the gate above against vacuity.

    The "all or nothing" assertion is satisfied by a lane that declares NONE of
    the DSNs. Both of these files run the projection-bearing runtime services,
    so at least one service must carry the key.
    """
    carriers = {
        service
        for service, env in _service_environments(compose_file).items()
        if INTERNAL_DSN_KEY in env
    }

    assert carriers, (
        f"{compose_file.name} runs the runtime but declares "
        f"{INTERNAL_DSN_KEY} on no service"
    )


# ---------------------------------------------------------------------------
# AC4 — the credential identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_file", PROJECTION_LANE_COMPOSE_FILES, ids=lambda p: p.name
)
def test_internal_dsn_names_the_non_superuser_runtime_principal(
    compose_file: Path,
) -> None:
    """``postgres`` is the table OWNER and carries SUPERUSER + BYPASSRLS.

    Connecting the internal projections as the superuser would satisfy the
    wiring check while re-creating, for a second pool, the exact RLS-exempt
    identity OMN-15363 removed from the analytics pool.
    """
    for service, env in _service_environments(compose_file).items():
        if INTERNAL_DSN_KEY not in env:
            continue
        dsn = _internal_dsn(compose_file, service)

        assert dsn.startswith(f"postgresql://{RUNTIME_PRINCIPAL}:"), (
            f"{compose_file.name}:{service}: internal DSN must connect as "
            f"{RUNTIME_PRINCIPAL}, got {dsn!r}"
        )
        assert "postgresql://postgres:" not in dsn
        assert "POSTGRES_USER" not in dsn
        assert "POSTGRES_PASSWORD" not in dsn
        assert dsn.endswith(f"@postgres:5432/{APPLICATION_PHYSICAL_DB}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_file", PROJECTION_LANE_COMPOSE_FILES, ids=lambda p: p.name
)
def test_internal_dsn_password_is_fail_closed_not_defaulted(
    compose_file: Path,
) -> None:
    """``:-`` would render a DSN with an EMPTY password.

    That is strictly worse than the bug being fixed: the wiring check would pass
    (the string is non-empty) and the failure would move from boot-time, where a
    log line names it, to first-connect, where it surfaces as an auth error on
    whichever projection happens to flush first.
    """
    for service, env in _service_environments(compose_file).items():
        if INTERNAL_DSN_KEY not in env:
            continue
        dsn = _internal_dsn(compose_file, service)

        assert "${" + RUNTIME_PASSWORD_VAR + ":?" in dsn, (
            f"{compose_file.name}:{service}: password must use the fail-closed "
            f"${{{RUNTIME_PASSWORD_VAR}:?...}} form, got {dsn!r}"
        )
        assert "${" + RUNTIME_PASSWORD_VAR + ":-" not in dsn
        assert "${" + RUNTIME_PASSWORD_VAR + "}" not in dsn


@pytest.mark.unit
def test_no_credential_literal_is_committed_in_the_dsn() -> None:
    """The value lives in the host env / secret store; compose carries a REF."""
    for compose_file in PROJECTION_LANE_COMPOSE_FILES:
        for service, env in _service_environments(compose_file).items():
            if INTERNAL_DSN_KEY not in env:
                continue
            dsn = _internal_dsn(compose_file, service)
            credential = dsn.split("://", 1)[1].split("@", 1)[0]
            _, _, password = credential.partition(":")

            assert password.startswith("${"), (
                f"{compose_file.name}:{service}: password must be an "
                f"interpolated reference, got a literal in {dsn!r}"
            )


@pytest.mark.unit
def test_required_env_manifest_declares_the_runtime_password() -> None:
    """``check_required_env_vars.py`` diffs this manifest against the base
    compose file's ``${VAR:?}`` set; drift in either direction fails the hook."""
    declared = {
        line.strip()
        for line in REQUIRED_ENV_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert RUNTIME_PASSWORD_VAR in declared


# ---------------------------------------------------------------------------
# The provisioning seam
# ---------------------------------------------------------------------------


def _bootstrap_executable_text() -> str:
    """The bootstrap script with ``#`` comment lines removed.

    The file documents the very map membership asserted absent below, so
    scanning raw text would make that assertion unfalsifiable.
    """
    return "\n".join(
        line
        for line in BOOTSTRAP.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.mark.unit
def test_bootstrap_provisions_login_for_the_runtime_principal() -> None:
    """A fresh volume must mint the LOGIN + password, or the DSN cannot connect.

    ``099_create_omninode_internal_live_events.sql`` creates the role NOLOGIN on
    purpose — 094's invariant is that no credential material lives in a
    migration and that the LOGIN attach is deployment-owned. On compose lanes
    the deployment-owned seam is this init script.
    """
    text = _bootstrap_executable_text()

    assert RUNTIME_PASSWORD_VAR in text, (
        f"{BOOTSTRAP.name} must consume {RUNTIME_PASSWORD_VAR} so a fresh "
        f"volume provisions {RUNTIME_PRINCIPAL}'s LOGIN credential"
    )
    assert RUNTIME_PRINCIPAL in text


def _bash_array_entries(text: str, name: str) -> list[str]:
    """The quoted entries of a ``NAME=( "a" "b" )`` bash array literal."""
    marker = f"{name}=("
    start = text.index(marker) + len(marker)
    body = text[start : text.index(")", start)]
    return re.findall(r'"([^"]+)"', body)


@pytest.mark.unit
def test_bootstrap_never_grants_the_runtime_principal_ddl_on_the_database() -> None:
    """AC4: no DDL, no ownership.

    ``SERVICE_DB_MAP`` membership routes a role through
    ``grant_role_to_database``, which issues ``GRANT USAGE, CREATE ON SCHEMA
    public`` and ``ALTER DEFAULT PRIVILEGES ... DELETE`` — far broader than the
    per-table INSERT/SELECT/UPDATE the topology declares for this principal, and
    ``CREATE`` on the schema lets it OWN tables, which exempts it from RLS
    unconditionally, FORCE included. The runtime principal's authorization is
    owned by the topology-derived migrations; this seam may mint the credential
    and nothing else.
    """
    text = _bootstrap_executable_text()
    roles_granted_full_database_access = {
        entry.split(":")[1] for entry in _bash_array_entries(text, "SERVICE_DB_MAP")
    }

    assert RUNTIME_PRINCIPAL not in roles_granted_full_database_access, (
        f"{RUNTIME_PRINCIPAL} is a SERVICE_DB_MAP entry — that path runs "
        "grant_role_to_database, which issues GRANT USAGE, CREATE ON SCHEMA "
        "public and ALTER DEFAULT PRIVILEGES ... DELETE. Mint the LOGIN "
        "credential through the login-only seam instead."
    )


@pytest.mark.unit
def test_bootstrap_login_only_seam_carries_the_runtime_principal() -> None:
    """The credential seam is declarative, so the gate above cannot be satisfied
    by simply dropping the provisioning altogether."""
    text = _bootstrap_executable_text()
    entries = _bash_array_entries(text, "LOGIN_ONLY_ROLE_MAP")

    assert f"{RUNTIME_PRINCIPAL}:{RUNTIME_PASSWORD_VAR}" in entries, (
        f"expected {RUNTIME_PRINCIPAL}:{RUNTIME_PASSWORD_VAR} in "
        f"LOGIN_ONLY_ROLE_MAP, got {entries}"
    )
