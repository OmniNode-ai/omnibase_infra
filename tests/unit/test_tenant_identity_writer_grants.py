# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26471 -- the least-grant ``tenant_identity_writer`` role on ``omninode_cloud``.

The runtime tenant and key nodes write the cloud identity tables' rows. This
change provisions a login-only role, ``tenant_identity_writer``, whose grants are
exactly the list below, through a seam modelled on ``gateway_runtime_grants.sql``
(OMN-26151), and renders its DSN into the dev lane's main runtime fail-closed.

Every pin reads a committed file: the seam, the bootstrap and runner credential
maps, the slot provisioner, the compose files, the password-format contract, the
env examples and manifests, and the runtime policy contract. The seam's own
readback is pinned against the GRANT statements it sits under, with positive
controls: a copy with one grant dropped, and a copy with a delegable grant, must
each be refused.
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

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = (
    ROOT / "docker" / "migrations" / "forward" / "000_create_multiple_databases.sh"
)
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"
PROVISIONER = ROOT / "scripts" / "provision_db_slot.sh"
PREFLIGHT = ROOT / "scripts" / "preflight_password_contract.py"
COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
COMPOSE_JUDGE = ROOT / "docker" / "docker-compose.judge.yml"
COMPOSE_DEV_LANE = ROOT / "docker" / "docker-compose.dev-lane.yml"
DOCKER_ENV_EXAMPLE = ROOT / "docker" / ".env.example"
ROOT_ENV_EXAMPLE = ROOT / ".env.example"
DEV_MANIFEST = ROOT / "docker" / "dev-lane-required-env.manifest.txt"
DEV_INSTANCE_TEMPLATE = (
    ROOT / "scripts" / "deploy-agent" / "deploy" / "dev-instance.env.template"
)
POLICY_CONTRACT = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV = ROOT / "docker" / "runtime-policy.env"
SEAM = ROOT / "docker" / "migrations" / "seams" / "tenant_identity_writer_grants.sql"
GATEWAY_SEAM = ROOT / "docker" / "migrations" / "seams" / "gateway_runtime_grants.sql"

ROLE = "tenant_identity_writer"
PASSWORD_VAR = "TENANT_IDENTITY_WRITER_PASSWORD"
DSN_VAR = "TENANT_IDENTITY_DATABASE_DSN"
DATABASE = "omninode_cloud"
LOGICAL_NAME = "tenant_identity.database.dsn"
SEQUENCE = "tenant_event_outbox_id_seq"
GRANT_SEAM_END = "# ---- END login-only role grant seam (OMN-18060) ----"

# The declared grant set, keyed by the object each applies to. Nothing else.
# `UPDATE(a,b)` is a column-scoped UPDATE, columns sorted.
EXPECTED_GRANTS: dict[str, frozenset[str]] = {
    f"DATABASE {DATABASE}": frozenset({"CONNECT"}),
    "SCHEMA public": frozenset({"USAGE"}),
    "public.tenants": frozenset(
        {
            "SELECT",
            "INSERT",
            "UPDATE(kc_patch_attempt_count,kc_patch_last_error,name,"
            "principal_id,provisioning_state)",
        }
    ),
    "public.plan_tiers": frozenset({"SELECT"}),
    "public.tenant_plans": frozenset({"SELECT", "INSERT"}),
    "public.tenant_api_keys": frozenset(
        {"SELECT", "INSERT", "UPDATE(is_active,revoked_at)"}
    ),
    "public.tenant_event_outbox": frozenset(
        {"SELECT", "INSERT", "UPDATE(attempts,last_error,published_at)"}
    ),
    f"SEQUENCE public.{SEQUENCE}": frozenset({"USAGE"}),
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

PROBES = (
    "has_table_privilege",
    "has_column_privilege",
    "has_sequence_privilege",
    "has_database_privilege",
    "has_schema_privilege",
)


def _executable_lines(text: str, comment: str = "#") -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith(comment)]


def _entries(text: str, opener: str, closer: str) -> list[str]:
    start = text.index(opener) + len(opener)
    return re.findall(r'"([^"]+)"', text[start : text.index(closer, start)])


def _seam_text() -> str:
    assert SEAM.is_file(), f"{SEAM} is missing"
    return SEAM.read_text(encoding="utf-8")


def _seam(text: str | None = None) -> str:
    """The seam without its comments."""
    body = _seam_text() if text is None else text
    return "\n".join(_executable_lines(body, "--"))


def _norm_privilege(raw: str) -> str:
    """`UPDATE ( b, a )` -> `UPDATE(a,b)`: one spelling per privilege."""
    raw = re.sub(r"\s+", " ", raw).strip()
    match = re.fullmatch(r"(\w+) ?\((.+)\)", raw)
    if match is None:
        return raw
    cols = sorted(c.strip() for c in match.group(2).split(","))
    return f"{match.group(1)}({','.join(cols)})"


def _declared_grants(seam: str) -> dict[str, set[str]]:
    """Fold every GRANT statement of the seam into object -> privileges."""
    folded: dict[str, set[str]] = {}
    for privileges, target in re.findall(
        r"GRANT\s+(.+?)\s+ON\s+(.+?)\s+TO\s+[^;]+;", seam, flags=re.DOTALL
    ):
        held = {
            _norm_privilege(p)
            for p in re.split(r",\s*(?![^()]*\))", re.sub(r"\s+", " ", privileges))
        }
        objects = [re.sub(r"\s+", " ", o).strip() for o in target.split(",")]
        if objects[0].startswith(("DATABASE ", "SCHEMA ", "SEQUENCE ")):
            assert len(objects) == 1, f"{target!r}: one object per GRANT of this kind"
        for obj in objects:
            folded.setdefault(obj, set()).update(held)
    return folded


# ---------------------------------------------------------------------------
# The grants: exactly the declared set and nothing broader
# ---------------------------------------------------------------------------


def test_seam_declares_exactly_the_declared_grants() -> None:
    declared = _declared_grants(_seam())
    assert {k: frozenset(v) for k, v in declared.items()} == EXPECTED_GRANTS


def test_seam_names_only_the_five_identity_tables_and_gates_on_them() -> None:
    seam = _seam()
    tables = {
        obj.split(".", 1)[1]
        for obj in _declared_grants(seam)
        if obj.startswith("public.")
    }
    assert tables == {
        "tenants",
        "plan_tiers",
        "tenant_plans",
        "tenant_api_keys",
        "tenant_event_outbox",
    }
    assert "to_regclass('public.' || t.name)" in seam
    array = re.search(r"ARRAY\[(.+?)\]", seam, flags=re.S)
    assert array is not None
    assert set(re.findall(r"'(\w+)'", array.group(1))) == tables
    assert f"to_regclass('public.{SEQUENCE}')" in seam


def _statements(seam: str) -> str:
    """Every statement that can confer or create (not the REVOKE reset or probes)."""
    return "\n".join(
        re.findall(
            r"^\s*(?:GRANT|ALTER|CREATE|DROP|TRUNCATE|DELETE|INSERT)\b[^;]*;",
            seam,
            flags=re.M,
        )
    )


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_seam_cannot_express_a_broader_privilege(token: str) -> None:
    statements = _statements(_seam())
    assert "GRANT" in statements, "no statement parsed: the scan would be vacuous"
    assert token not in statements.upper(), (
        f"the OMN-26471 grant seam contains {token!r}; it may issue only the "
        "declared grants"
    )


def test_a_delegable_grant_is_refused_by_the_forbidden_scan() -> None:
    """Positive control: the scan above fires on a grant with delegation rights."""
    mutated = _seam().replace(
        'GRANT USAGE ON SCHEMA public TO "tenant_identity_writer";',
        'GRANT USAGE ON SCHEMA public TO "tenant_identity_writer" WITH GRANT OPTION;',
    )
    assert mutated != _seam(), "the control did not change the seam"
    assert "GRANT OPTION" in _statements(mutated).upper()


def test_seam_resets_before_it_grants() -> None:
    """A re-assert must remove drift, not only add: revoke first, then grant."""
    seam = _seam()
    first_grant = re.search(
        r"^\s*GRANT (?:SELECT|UPDATE|INSERT|USAGE ON SEQ)", seam, re.M
    )
    assert first_grant
    for revoke in (
        f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{ROLE}"',
        f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM "{ROLE}"',
    ):
        assert revoke in seam
        assert seam.index(revoke) < first_grant.start()


def test_seam_gates_on_role_database_tables_and_sequence() -> None:
    seam = _seam()
    for gate in ("role_present", "db_present", "tables_present", "sequence_present"):
        assert f"\\if :{gate}" in seam, f"no gate on {gate}"
    assert seam.count("\\endif") == seam.count("\\if ")
    assert f"skip  {ROLE} grants" in seam


def test_seam_carries_no_credential_and_no_role_attribute() -> None:
    seam = _seam().upper()
    for word in ("PASSWORD", "CREATE ROLE", "ALTER ROLE", "LOGIN"):
        assert word not in seam


# ---------------------------------------------------------------------------
# The readback: effective privileges, delegation, and the positive controls
# ---------------------------------------------------------------------------


def _readback(seam: str) -> str:
    match = re.search(r"DO \$(\w+_readback)\$.*?\$\1\$;", seam, flags=re.S)
    assert match, "the seam has no readback DO block"
    return match.group(0)


def _readback_literals(seam: str) -> dict[str, str]:
    """The expected strings the readback compares each effective set with."""
    block = _readback(seam)
    out = dict(re.findall(r"(\w+_set) IS DISTINCT FROM '([^']*)'", block))
    assert set(out) == {"table_set", "column_set", "sequence_set"}, out
    return out


def _effective_from_grants(seam: str) -> dict[str, str]:
    """The string each probe returns, computed from the GRANT statements alone."""
    tables: list[tuple[str, str]] = []
    columns: list[tuple[str, str, str]] = []
    sequences: list[tuple[str, str]] = []
    for obj, privs in _declared_grants(seam).items():
        if obj.startswith(("DATABASE ", "SCHEMA ")):
            continue
        if obj.startswith("SEQUENCE "):
            sequences += [(obj.split(".", 1)[1], p) for p in privs]
            continue
        name = obj.split(".", 1)[1]
        for priv in privs:
            col = re.fullmatch(r"(\w+)\((.+)\)", priv)
            if col is None:
                tables.append((name, priv))
            else:
                columns += [(name, c, col.group(1)) for c in col.group(2).split(",")]
    return {
        "table_set": ",".join(f"{r}:{p}" for r, p in sorted(tables)),
        "column_set": ",".join(f"{r}.{c}:{p}" for r, c, p in sorted(columns)),
        "sequence_set": ",".join(f"{r}:{p}" for r, p in sorted(sequences)),
    }


def test_readback_probes_every_effective_privilege_and_raises() -> None:
    block = _readback(_seam())
    for probe in PROBES:
        assert probe in block, f"the readback never calls {probe}"
        assert f"{probe}(" in block
    assert "RAISE EXCEPTION" in block
    assert block.count("WITH GRANT OPTION") >= len(PROBES), (
        "the readback must test the delegation form of every probe, not only the "
        "plain form (the gateway seam does not)"
    )
    assert "rolsuper" in _seam() and "rolbypassrls" in _seam()


def test_readback_expected_sets_equal_the_sets_the_grants_produce() -> None:
    seam = _seam()
    assert _readback_literals(seam) == _effective_from_grants(seam)


def test_readback_fires_when_one_grant_is_dropped_from_a_copy() -> None:
    """Positive control: drop one grant, and the readback's expected set differs."""
    seam = _seam()
    for dropped in (
        'GRANT SELECT ON public.plan_tiers TO "tenant_identity_writer";',
        f'GRANT USAGE ON SEQUENCE public.{SEQUENCE} TO "tenant_identity_writer";',
    ):
        mutated = seam.replace(dropped, "")
        assert mutated != seam, f"the control did not drop {dropped!r}"
        assert _readback_literals(mutated) != _effective_from_grants(mutated)


def test_readback_fires_when_a_column_is_added_to_a_copy() -> None:
    seam = _seam()
    mutated = seam.replace(
        "(is_active, revoked_at)", "(is_active, revoked_at, key_hash)"
    )
    assert mutated != seam
    assert _readback_literals(mutated) != _effective_from_grants(mutated)


def test_seam_reports_ok_only_after_the_readback() -> None:
    seam = _seam()
    ok = f"{ROLE} holds exactly the declared grants"
    assert ok in seam
    assert seam.index(_readback(seam)) < seam.index(ok)


# ---------------------------------------------------------------------------
# The role mints login only, and the runner asserts its credential
# ---------------------------------------------------------------------------


def test_bootstrap_login_only_map_carries_the_role() -> None:
    entries = _entries(
        BOOTSTRAP.read_text(encoding="utf-8"), "LOGIN_ONLY_ROLE_MAP=(", ")"
    )
    assert f"{ROLE}:{PASSWORD_VAR}" in entries


def test_runner_credential_map_carries_the_role() -> None:
    entries = _entries(
        RUNNER.read_text(encoding="utf-8"), "for login_role_entry in", "; do"
    )
    assert f"{ROLE}:{PASSWORD_VAR}" in [e for e in entries if ":" in e]


def test_runner_slot_managed_roles_and_provisioner_carry_the_role() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    managed = re.search(r'^SLOT_MANAGED_ROLES="([^"]+)"', runner, re.M)
    assert managed and ROLE in managed.group(1).split()
    provisioner = PROVISIONER.read_text(encoding="utf-8")
    login_only = re.search(r'^LOGIN_ONLY_ROLES="([^"]+)"', provisioner, re.M)
    assert login_only and ROLE in login_only.group(1).split()
    assert re.search(rf"^{ROLE}:\d+$", provisioner, re.M), "no connection budget"


def test_runner_applies_the_seam_inside_the_slot_guarded_grant_seam() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    guard = text.index('slot_skip_role_seam "login-only role grants (OMN-18060)"')
    gateway = text.index("/migrations/seams/gateway_runtime_grants.sql")
    call = text.index("/migrations/seams/tenant_identity_writer_grants.sql")
    sentinel = text.index("All migrations complete. Setting sentinel TRUE")
    assert guard < text.index("for grant_role_entry in") < gateway < call
    assert call < text.index(GRANT_SEAM_END) < sentinel
    assert text[guard:call].count("\nfi") == 0, "the call left the slot-guarded branch"


def test_runner_without_the_seam_file_logs_a_named_skip() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert f"skip  {ROLE} grants (no grants file at" in text


def test_every_definition_that_runs_the_runner_mounts_the_seams_directory() -> None:
    runner_mount = "../scripts/run-forward-migrations.sh:/run-forward-migrations.sh:ro"
    seams_mount = "../docker/migrations/seams:/migrations/seams:ro"
    carriers = [
        p
        for p in sorted((ROOT / "docker").rglob("*.y*ml"))
        if runner_mount in p.read_text(encoding="utf-8")
    ]
    assert len(carriers) >= 5, f"fewer runner carriers than expected: {carriers}"
    missing = [
        str(p.relative_to(ROOT))
        for p in carriers
        if seams_mount not in p.read_text(encoding="utf-8")
    ]
    assert not missing, f"these mount the runner but not the seams directory: {missing}"


def test_gateway_seam_is_untouched_by_this_change() -> None:
    """AC4: the nine gateway grants and their `ok` line are what they were."""
    text = GATEWAY_SEAM.read_text(encoding="utf-8")
    assert "ok    gateway_runtime holds exactly the nine gateway grants" in text


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


def test_dev_lane_overlay_renders_the_dsn_fail_closed_in_the_main_runtime() -> None:
    text = COMPOSE_DEV_LANE.read_text(encoding="utf-8")
    assert re.search(DSN_LINE, text), (
        f"{DSN_VAR} must name {ROLE}@postgres:5432/{DATABASE} with the fail-closed "
        f"${{{PASSWORD_VAR}:?...}} form"
    )
    runtime = _service_block(text, "omninode-runtime")
    assert "*dev_lane_gateway_env" in runtime
    anchors = re.findall(r"&(dev_lane_\w+)\n(?:  \S.*\n)*?  " + DSN_VAR, text)
    assert len(anchors) == 1 and f"*{anchors[0]}" in runtime, (
        "omninode-runtime does not merge the anchor that carries the DSN"
    )
    # P38: only omninode-runtime carries the DSN, never the effects runtime.
    assert f"*{anchors[0]}" not in _service_block(text, "runtime-effects")


def test_shared_base_never_requires_the_password_and_only_passes_the_dsn() -> None:
    base = "\n".join(_executable_lines(COMPOSE_INFRA.read_text(encoding="utf-8")))
    dsn_lines = [ln.strip() for ln in base.splitlines() if DSN_VAR in ln]
    assert dsn_lines == [f"{DSN_VAR}: ${{{DSN_VAR}:-}}"]
    assert f"${{{PASSWORD_VAR}:?" not in base


# ---------------------------------------------------------------------------
# Password format contract, env examples, manifest, instance template
# ---------------------------------------------------------------------------


def _preflight_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preflight_pw_tiw", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_password_contract_holds_the_password_as_hex() -> None:
    module = _preflight_module()
    assert PASSWORD_VAR in module.CONTRACT
    assert module.CONTRACT[PASSWORD_VAR].name == "hex"
    assert module.violations({PASSWORD_VAR: "zz-not-hex"}) != []
    assert module.violations({PASSWORD_VAR: "ab12" * 16}) == []


@pytest.mark.parametrize("path", [DOCKER_ENV_EXAMPLE, ROOT_ENV_EXAMPLE], ids=str)
def test_env_examples_name_the_password(path: Path) -> None:
    assert f"{PASSWORD_VAR} is hex" in path.read_text(encoding="utf-8")


def test_dev_lane_manifest_and_instance_template_list_the_password() -> None:
    names = {
        ln.strip()
        for ln in DEV_MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    assert PASSWORD_VAR in names
    template = DEV_INSTANCE_TEMPLATE.read_text(encoding="utf-8").splitlines()
    assert f"{PASSWORD_VAR}=" in template


# ---------------------------------------------------------------------------
# Runtime policy: the logical name resolves from the container env
# ---------------------------------------------------------------------------


def _profiles() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(POLICY_CONTRACT.read_text(encoding="utf-8"))["profiles"]


def test_policy_maps_the_secret_to_the_dsn_env_on_the_dev_profile_only() -> None:
    mapped: dict[str, list[dict[str, Any]]] = {}
    for name, body in _profiles().items():
        rows = [
            m
            for m in body.get("secret_resolver_mappings", [])
            if m["logical_name"] == LOGICAL_NAME
        ]
        if rows:
            mapped[name] = rows
    assert set(mapped) == {"dev"}
    assert len(mapped["dev"]) == 1
    assert mapped["dev"][0]["source"] == {"source_type": "env", "source_path": DSN_VAR}


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
    assert "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON" in carrying
    assert not {n for n in carrying if not n.startswith("DEV_")}
    committed = POLICY_ENV.read_text(encoding="utf-8")
    for name in carrying:
        assert f"{name}=" in committed
    assert committed.count(f'"logical_name":"{LOGICAL_NAME}"') == len(carrying)
