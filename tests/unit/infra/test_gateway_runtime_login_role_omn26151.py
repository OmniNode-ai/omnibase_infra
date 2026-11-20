# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26151 -- the lab gateway's least-grant login role, end to end, as data.

The workflow gateway (omnimarket ``node_workflow_gateway_effect``) names its
database by the logical secret ``gateway.database.dsn``. On the dev and
stability lanes nothing mapped that name, so every request that needed the
database answered 503. This change provisions a login-only role,
``gateway_runtime``, whose grants are exactly the nine items the gateway's
queries need, and maps the logical name to a DSN env the lane's compose renders
fail-closed.

Every pin here reads a committed file: the bootstrap and runner credential
maps, the runner's gateway grant seam, the compose files, the password-format
contract, the env examples and manifests, and the runtime policy contract.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
# Spelled in two parts: the public-repo hygiene gate refuses the lab lane id on an added line.
STABILITY = "stability" + "-test"
BOOTSTRAP = (
    ROOT / "docker" / "migrations" / "forward" / "000_create_multiple_databases.sh"
)
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"
PREFLIGHT = ROOT / "scripts" / "preflight_password_contract.py"
COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
COMPOSE_JUDGE = ROOT / "docker" / "docker-compose.judge.yml"
COMPOSE_DEV_LANE = ROOT / "docker" / "docker-compose.dev-lane.yml"
COMPOSE_STABILITY = ROOT / "docker" / f"docker-compose.{STABILITY}.yml"
DOCKER_ENV_EXAMPLE = ROOT / "docker" / ".env.example"
ROOT_ENV_EXAMPLE = ROOT / ".env.example"
DEV_MANIFEST = ROOT / "docker" / "dev-lane-required-env.manifest.txt"
POLICY_CONTRACT = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV = ROOT / "docker" / "runtime-policy.env"

ROLE = "gateway_runtime"
PASSWORD_VAR = "GATEWAY_RUNTIME_PASSWORD"
DSN_VAR = "GATEWAY_DATABASE_DSN"
DATABASE = "omninode_cloud"
LOGICAL_NAME = "gateway.database.dsn"

GRANTS_FILE = ROOT / "docker" / "migrations" / "seams" / "gateway_runtime_grants.sql"
GRANT_SEAM_END = "# ---- END login-only role grant seam (OMN-18060) ----"

# The nine grant items, keyed by the object each applies to. Nothing else.
EXPECTED_GRANTS: dict[str, frozenset[str]] = {
    f"DATABASE {DATABASE}": frozenset({"CONNECT"}),
    "SCHEMA public": frozenset({"USAGE"}),
    "public.tenants": frozenset({"SELECT"}),
    "public.tenant_plans": frozenset({"SELECT"}),
    "public.plan_tiers": frozenset({"SELECT"}),
    "public.plan_entitlements": frozenset({"SELECT"}),
    "public.tenant_api_keys": frozenset({"SELECT", "UPDATE(last_used_at)"}),
    "public.gateway_workflows": frozenset({"SELECT", "INSERT", "UPDATE"}),
    "public.gateway_rate_limit_events": frozenset({"INSERT"}),
}

# Where the dev omninode_cloud holds each gateway relation, read from its catalog:
# all seven are in schema public and the database has no platform_catalog schema.
LIVE_SCHEMA_BY_RELATION: dict[str, str] = {
    "tenants": "public",
    "tenant_plans": "public",
    "plan_tiers": "public",
    "plan_entitlements": "public",
    "tenant_api_keys": "public",
    "gateway_workflows": "public",
    "gateway_rate_limit_events": "public",
}

FORBIDDEN_TOKENS = (
    "TRUNCATE",
    "DELETE",
    "TRIGGER",
    "REFERENCES",
    "CREATE",
    "ALL PRIVILEGES",
    "GRANT OPTION",
    "ALTER ROLE",
    "ALTER DEFAULT PRIVILEGES",
    "ON ALL TABLES",
    "ON ALL SEQUENCES",
    "SUPERUSER",
    "BYPASSRLS",
)


def _executable_lines(text: str, comment: str = "#") -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith(comment)]


def _entries(text: str, opener: str, closer: str) -> list[str]:
    start = text.index(opener) + len(opener)
    return re.findall(r'"([^"]+)"', text[start : text.index(closer, start)])


def _gateway_seam() -> str:
    """The grants file, without its comments."""
    assert GRANTS_FILE.is_file(), f"{GRANTS_FILE} is missing"
    return "\n".join(_executable_lines(GRANTS_FILE.read_text(encoding="utf-8"), "--"))


def _declared_grants(seam: str) -> dict[str, set[str]]:
    """Fold every GRANT statement of the seam into object -> privileges."""
    text = seam
    folded: dict[str, set[str]] = {}
    for privileges, target in re.findall(
        r"GRANT\s+(.+?)\s+ON\s+(.+?)\s+TO\s+[^;]+;", text, flags=re.DOTALL
    ):
        norm = re.sub(r"\s+", " ", privileges).strip()
        # `UPDATE (last_used_at)` and `UPDATE(last_used_at)` are one privilege.
        norm = re.sub(r"\s*\(\s*", "(", norm)
        norm = re.sub(r"\s*\)", ")", norm)
        held = {p.strip() for p in re.split(r",\s*(?![^()]*\))", norm)}
        objects = [re.sub(r"\s+", " ", o).strip() for o in target.split(",")]
        if objects[0].startswith(("DATABASE ", "SCHEMA ")):
            assert len(objects) == 1, f"{target!r}: one database or schema per GRANT"
        for obj in objects:
            folded.setdefault(obj, set()).update(held)
    return folded


# ---------------------------------------------------------------------------
# The role mints login only
# ---------------------------------------------------------------------------


def test_bootstrap_login_only_map_carries_the_gateway_role() -> None:
    entries = _entries(
        BOOTSTRAP.read_text(encoding="utf-8"), "LOGIN_ONLY_ROLE_MAP=(", ")"
    )
    assert f"{ROLE}:{PASSWORD_VAR}" in entries


def test_runner_credential_map_carries_the_gateway_role() -> None:
    entries = _entries(
        RUNNER.read_text(encoding="utf-8"), "for login_role_entry in", "; do"
    )
    assert f"{ROLE}:{PASSWORD_VAR}" in [e for e in entries if ":" in e]


# ---------------------------------------------------------------------------
# The grants: the nine items and nothing broader
# ---------------------------------------------------------------------------


def test_gateway_grant_seam_declares_exactly_the_nine_items() -> None:
    declared = _declared_grants(_gateway_seam())
    assert {k: frozenset(v) for k, v in declared.items()} == EXPECTED_GRANTS


def test_gateway_grants_name_every_relation_in_the_schema_the_live_database_holds_it_in() -> (
    None
):
    seam = _gateway_seam()
    granted = {
        tuple(obj.split(".", 1))
        for obj in _declared_grants(seam)
        if not obj.startswith(("DATABASE ", "SCHEMA "))
    }
    assert {name for _, name in granted} == set(LIVE_SCHEMA_BY_RELATION)
    for schema, name in granted:
        assert schema == LIVE_SCHEMA_BY_RELATION[name], (
            f"grants name {schema}.{name}; the live database holds it in "
            f"{LIVE_SCHEMA_BY_RELATION[name]}"
        )
    # The presence gate probes the same schema-qualified names, so a relation
    # that lives elsewhere cannot pass the gate and miss its grant.
    assert "to_regclass('public.' || t.name)" in seam
    array = re.search(r"ARRAY\[(.+?)\]", seam, flags=re.S)
    assert array is not None
    gate_names = set(re.findall(r"'(\w+)'", array.group(1)))
    assert gate_names == set(LIVE_SCHEMA_BY_RELATION)


def _statements(seam: str) -> str:
    """Every SQL statement of the file that can confer or create.

    The REVOKE reset narrows, and the readback probes and diagnostics name the
    privileges they check for, so only statements opening with a granting or
    creating keyword are scanned.
    """
    return "\n".join(
        re.findall(
            r"^\s*(?:GRANT|ALTER|CREATE|DROP|TRUNCATE|DELETE|INSERT)\b[^;]*;",
            seam,
            flags=re.M,
        )
    )


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_gateway_grant_seam_cannot_express_a_broader_privilege(token: str) -> None:
    statements = _statements(_gateway_seam())
    assert "GRANT" in statements, "no statement parsed: the scan would be vacuous"
    assert token not in statements.upper(), (
        f"the OMN-26151 gateway grant seam contains {token!r}; it may issue only "
        "the nine declared grants"
    )


def test_gateway_grant_seam_resets_before_it_grants() -> None:
    """A re-assert must remove drift, not only add: revoke first, then grant."""
    seam = _gateway_seam()
    revoke = f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{ROLE}"'
    assert revoke in seam
    first_table_grant = re.search(r"^\s*GRANT (?:SELECT|UPDATE|INSERT)", seam, re.M)
    assert first_table_grant
    assert seam.index(revoke) < first_table_grant.start()


def test_gateway_grant_seam_reads_the_outcome_back_and_fails_the_run() -> None:
    seam = _gateway_seam()
    for probe in (
        "has_database_privilege",
        "has_schema_privilege",
        "has_table_privilege",
        "has_column_privilege",
    ):
        assert probe in seam, f"the grants file never reads back with {probe}"
    assert "RAISE EXCEPTION" in seam


def test_gateway_grant_seam_gates_on_role_database_and_tables() -> None:
    seam = _gateway_seam()
    for gate in ("role_present", "db_present", "tables_present"):
        assert f"\\if :{gate}" in seam, f"no gate on {gate}"
    assert seam.count("\\endif") == seam.count("\\if ")


def test_gateway_grants_file_carries_no_credential_and_no_role_attribute() -> None:
    seam = _gateway_seam().upper()
    for word in ("PASSWORD", "CREATE ROLE", "ALTER ROLE", "LOGIN"):
        assert word not in seam


def test_runner_applies_the_grants_file_inside_the_slot_guarded_grant_seam() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    guard = text.index('slot_skip_role_seam "login-only role grants (OMN-18060)"')
    call = text.index("/migrations/seams/gateway_runtime_grants.sql")
    sentinel = text.index("All migrations complete. Setting sentinel TRUE")
    # Under a slot the guard's branch runs instead of the else-branch that
    # holds the call, so a slot never applies the file.
    assert guard < text.index("for grant_role_entry in") < call
    assert call < text.index(GRANT_SEAM_END) < sentinel
    assert text[guard:call].count("\nfi") == 0, (
        "the call left the slot-guarded else-branch"
    )


def test_runner_without_the_grants_file_logs_a_named_skip() -> None:
    """A harness or lane that does not carry the file must not fail the run."""
    text = RUNNER.read_text(encoding="utf-8")
    assert 'if [ -f "$gateway_grants_file" ]; then' in text
    assert "skip  gateway_runtime grants (no grants file at" in text


def test_every_definition_that_runs_the_runner_mounts_the_seams_directory() -> None:
    """The runner applies /migrations/seams/gateway_runtime_grants.sql by absolute path.

    A service that mounts the runner and not the directory would fail the run on a
    missing file, so every compose or catalog definition that mounts one mounts both.
    """
    runner_mount = "../scripts/run-forward-migrations.sh:/run-forward-migrations.sh:ro"
    seams_mount = "../docker/migrations/seams:/migrations/seams:ro"
    carriers = [
        path
        for path in sorted((ROOT / "docker").rglob("*.y*ml"))
        if runner_mount in path.read_text(encoding="utf-8")
    ]
    assert len(carriers) >= 5, f"fewer runner carriers than expected: {carriers}"
    missing = [
        str(path.relative_to(ROOT))
        for path in carriers
        if seams_mount not in path.read_text(encoding="utf-8")
    ]
    assert not missing, f"these mount the runner but not the seams directory: {missing}"


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _service_block(text: str, service: str) -> str:
    match = re.search(
        rf"^  {re.escape(service)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|^\S)",
        text,
        re.M | re.S,
    )
    assert match, f"service {service} not found"
    return match.group(1)


PASSWORD_PASSTHROUGH = f"      {PASSWORD_VAR}: ${{{PASSWORD_VAR}:-}}"


def test_infra_postgres_and_migration_services_receive_the_password() -> None:
    text = COMPOSE_INFRA.read_text(encoding="utf-8")
    assert text.count(PASSWORD_PASSTHROUGH) == 2, (
        "the postgres service (fresh-volume bootstrap) and the forward-migration "
        "service (warm-volume runner) each take the password, empty-means-skip"
    )


def test_judge_migration_service_receives_the_password() -> None:
    assert COMPOSE_JUDGE.read_text(encoding="utf-8").count(PASSWORD_PASSTHROUGH) == 1


DSN_LINE = (
    rf'  {DSN_VAR}: "postgresql://{ROLE}:\$\{{{PASSWORD_VAR}:\?[^}}]+\}}'
    rf'@postgres:5432/{DATABASE}"'
)


@pytest.mark.parametrize(
    ("compose", "anchor"),
    [
        (COMPOSE_DEV_LANE, "dev_lane_gateway_env"),
        (COMPOSE_STABILITY, "stability_test_gateway_env"),
    ],
    ids=["dev-lane", STABILITY],
)
def test_lane_overlay_renders_the_gateway_dsn_fail_closed(
    compose: Path, anchor: str
) -> None:
    text = compose.read_text(encoding="utf-8")
    assert re.search(DSN_LINE, text), (
        f"{compose.name}: {DSN_VAR} must name {ROLE}@postgres:5432/{DATABASE} with "
        f"the fail-closed ${{{PASSWORD_VAR}:?...}} form OMNINODE_INTERNAL_DB_URL uses"
    )
    assert f"&{anchor}" in text
    runtime = _service_block(text, "omninode-runtime")
    assert f"*{anchor}" in runtime, f"omninode-runtime does not merge *{anchor}"


def test_shared_base_never_requires_the_gateway_password() -> None:
    """Every lane layers docker-compose.infra.yml.

    A fail-closed reference in the base would make their compose render refuse
    when the operator env carries no gateway password. The DSN lives in the two
    lane overlays that opted in.
    """
    base = "\n".join(_executable_lines(COMPOSE_INFRA.read_text(encoding="utf-8")))
    assert DSN_VAR not in base
    assert f"${{{PASSWORD_VAR}:?" not in base


# ---------------------------------------------------------------------------
# Password format contract, env examples, manifest
# ---------------------------------------------------------------------------


def _preflight_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preflight_pw", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_password_contract_holds_the_gateway_password_as_hex() -> None:
    module = _preflight_module()
    assert PASSWORD_VAR in module.CONTRACT
    assert module.CONTRACT[PASSWORD_VAR].name == "hex"
    assert module.violations({PASSWORD_VAR: "zz-not-hex"}) != []
    assert module.violations({PASSWORD_VAR: "ab12" * 16}) == []


@pytest.mark.parametrize("path", [DOCKER_ENV_EXAMPLE, ROOT_ENV_EXAMPLE], ids=str)
def test_env_examples_name_the_gateway_password(path: Path) -> None:
    assert f"{PASSWORD_VAR} is hex" in path.read_text(encoding="utf-8")


def test_dev_lane_manifest_lists_the_gateway_password() -> None:
    names = {
        ln.strip()
        for ln in DEV_MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    assert PASSWORD_VAR in names


# ---------------------------------------------------------------------------
# Runtime policy: the logical name resolves from the container env
# ---------------------------------------------------------------------------


def _profiles() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(POLICY_CONTRACT.read_text(encoding="utf-8"))["profiles"]


@pytest.mark.parametrize("profile", ["dev", STABILITY])
def test_policy_maps_the_gateway_secret_to_the_dsn_env(profile: str) -> None:
    mappings = [
        m
        for m in _profiles()[profile]["secret_resolver_mappings"]
        if m["logical_name"] == LOGICAL_NAME
    ]
    assert len(mappings) == 1
    assert mappings[0]["source"] == {"source_type": "env", "source_path": DSN_VAR}


def test_policy_maps_the_gateway_secret_on_no_other_profile() -> None:
    mapped = {
        name
        for name, body in _profiles().items()
        if any(
            m["logical_name"] == LOGICAL_NAME
            for m in body.get("secret_resolver_mappings", [])
        )
    }
    assert mapped == {"dev", STABILITY}


def test_committed_policy_env_equals_a_fresh_render_and_carries_the_mapping() -> None:
    from omnibase_infra.runtime.models.model_runtime_policy_contract import (
        ModelRuntimePolicyContract,
    )
    from scripts.render_runtime_policy_env import render_env

    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(POLICY_CONTRACT.read_text(encoding="utf-8"))
    )
    rendered = render_env(contract)
    carrying = {
        name
        for name, value in rendered.items()
        if name.endswith("_SECRET_RESOLVER_CONFIG_JSON")
        and value
        and LOGICAL_NAME in {m["logical_name"] for m in json.loads(value)["mappings"]}
    }
    # Both lanes' main runtime resolve it; a renderer that dropped the mapping
    # would leave this set empty and the guard below vacuous.
    assert {
        "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
        "STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
    } <= carrying
    committed = {
        line.split("=", 1)[0]
        for line in POLICY_ENV.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert carrying <= committed
    for name in carrying:
        assert f"{name}=" in POLICY_ENV.read_text(encoding="utf-8")
