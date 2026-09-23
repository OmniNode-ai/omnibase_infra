# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19087: one stated format contract for the five compose passwords.

Before this ticket, three of the five passwords a lane deploy consumes had a
format rule, enforced only by the forward migration, which runs after Postgres,
Redpanda and Valkey are up. A dogfood cold bring-up on 2026-09-21 used a mixed
alphanumeric generator, all infrastructure came up healthy, and the deploy died
minutes later in a different script naming a third variable.

AC-1: a malformed value for any of the five is refused before any container
starts, naming the variable and the contract. The refusal runs in
``scripts/deploy-runtime.sh`` (``guard_password_contract``, called in ``main``
before attribution, build, sync or bring-up) and in
``scripts/preflight_required_compose_env.py`` (the deploy agent's and
``refresh_dev_lane.sh``'s pre-compose check).

AC-2: the documented contract matches what is enforced. The contract is stated
once, in ``scripts/preflight_password_contract.py``. Every tracked env example
carries the block that module renders, verbatim; the migration runner and the
bootstrap keep the same login-role list and hex pattern; no example value and no
compose default for these five sits outside the contract.

Every scanner below has a positive control: a fixture it must flag.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_CHECKER = _SCRIPTS / "preflight_password_contract.py"
_COMPOSE_PREFLIGHT = _SCRIPTS / "preflight_required_compose_env.py"
_DEPLOY_RUNTIME = _SCRIPTS / "deploy-runtime.sh"
_RUNNER = _SCRIPTS / "run-forward-migrations.sh"
_BOOTSTRAP = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "000_create_multiple_databases.sh"
)

_ENV_EXAMPLES = (
    _REPO_ROOT / ".env.example",
    _REPO_ROOT / "docker" / ".env.example",
    _REPO_ROOT / "docker" / "env-example-full.txt",
    _REPO_ROOT / "docker" / "dogfood.env.example",
    _REPO_ROOT / "docker" / "judge.env.example",
    _REPO_ROOT / "docker" / "lakshman.env.example",
)


def _load_contract() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "preflight_password_contract", _CHECKER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its module through sys.modules while it is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CONTRACT = _load_contract()
_ALL_VARS: tuple[str, ...] = tuple(_CONTRACT.CONTRACT)

# Test values, never real credentials. The hex value satisfies both formats.
_HEX_VALUE = "0123456789abcdef" * 4
# The live lab shape for the two infrastructure passwords (alphanumeric, not hex).
_ALNUM_VALUE = "Zq9Kx7Wm" * 5
# Breaks a postgresql:// or redis:// URL.
_URL_BREAKING_VALUE = "ab@cd:ef/gh"
# The bootstrap's placeholder form.
_PLACEHOLDER_VALUE = "__REPLACE_WITH_SECURE_PASSWORD__"


def _valid_env() -> dict[str, str]:
    return dict.fromkeys(_ALL_VARS, _HEX_VALUE)


def _malformed_value(var: str) -> str:
    """A value outside ``var``'s own format.

    For a hex variable this is the incident's shape: mixed alphanumeric, which
    is fine for the other two variables and wrong here.
    """
    if _CONTRACT.CONTRACT[var] is _CONTRACT.HEX:
        return _ALNUM_VALUE
    return _URL_BREAKING_VALUE


def _run_checker(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_CHECKER), "--lane", "test"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
        check=False,
    )


# --------------------------------------------------------------------------
# The contract itself
# --------------------------------------------------------------------------


def test_contract_covers_exactly_the_five_compose_passwords() -> None:
    assert set(_ALL_VARS) == {
        "POSTGRES_PASSWORD",
        "VALKEY_PASSWORD",
        "OMNINODE_RUNTIME_PASSWORD",
        "TENANT_PROJECTION_WRITER_PASSWORD",
        "CHAIN_CANARY_READER_PASSWORD",
    }
    for var in _CONTRACT.LOGIN_ROLE_VARS:
        assert _CONTRACT.CONTRACT[var] is _CONTRACT.HEX
    for var in _CONTRACT.INFRA_VARS:
        assert _CONTRACT.CONTRACT[var] is _CONTRACT.URL_SAFE


# --------------------------------------------------------------------------
# AC-1: each of the five, malformed in turn, is refused and named
# --------------------------------------------------------------------------


@pytest.mark.parametrize("var", _ALL_VARS)
def test_each_malformed_password_is_refused_by_name(var: str) -> None:
    env = _valid_env()
    bad = _malformed_value(var)
    env[var] = bad
    result = _run_checker(env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "PASSWORD_CONTRACT_VIOLATION" in result.stderr
    assert f"    {var}\n" in result.stderr
    assert f"contract : {_CONTRACT.CONTRACT[var].name}" in result.stderr
    others = set(_ALL_VARS) - {var}
    for other in others:
        assert f"    {other}\n" not in result.stderr, f"{other} named but valid"
    assert bad not in result.stdout + result.stderr, "a value was printed"


@pytest.mark.parametrize("var", _ALL_VARS)
def test_the_placeholder_is_refused_for_every_password(var: str) -> None:
    env = _valid_env()
    env[var] = _PLACEHOLDER_VALUE
    result = _run_checker(env)
    assert result.returncode == 1
    assert f"    {var}\n" in result.stderr


def test_negative_control_all_five_valid_hex_passes() -> None:
    result = _run_checker(_valid_env())
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_negative_control_live_lab_shape_passes() -> None:
    """Alphanumeric infrastructure passwords and hex login roles: every lab host."""
    env = _valid_env()
    for var in _CONTRACT.INFRA_VARS:
        env[var] = _ALNUM_VALUE
    assert _run_checker(env).returncode == 0


def test_unset_and_empty_are_not_judged_here() -> None:
    """Presence belongs to the compose guards and the required-env preflight."""
    assert _run_checker({}).returncode == 0
    assert _run_checker(dict.fromkeys(_ALL_VARS, "")).returncode == 0


def test_every_violation_is_named_in_one_message() -> None:
    env = {var: _malformed_value(var) for var in _ALL_VARS}
    result = _run_checker(env)
    assert result.returncode == 1
    assert f"5 of {len(_ALL_VARS)}" in result.stderr
    for var in _ALL_VARS:
        assert f"    {var}\n" in result.stderr


def test_postgres_volume_warning_only_when_postgres_is_refused() -> None:
    env = _valid_env()
    env["VALKEY_PASSWORD"] = _URL_BREAKING_VALUE
    assert "data volume initialises" not in _run_checker(env).stderr

    env["POSTGRES_PASSWORD"] = _URL_BREAKING_VALUE
    assert "data volume initialises" in _run_checker(env).stderr


# --------------------------------------------------------------------------
# AC-1: the refusal is wired where a deploy begins, before any container starts
# --------------------------------------------------------------------------


def _function_body(text: str, name: str) -> str:
    start = text.index(f"\n{name}() {{\n")
    end = text.index("\n}\n", start)
    return text[start : end + 3]


def _run_deploy_guard(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    text = _DEPLOY_RUNTIME.read_text(encoding="utf-8")
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            "resolve_lane_name() { printf 'dogfood\\n'; }",
            'OMNIBASE_OPERATOR_ENV_FILE="operator.env"',
            _function_body(text, "guard_password_contract"),
            f'guard_password_contract "{_REPO_ROOT}" omnibase-infra-dogfood',
            "echo GUARD_PASSED",
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", **env},
        check=False,
    )


@pytest.mark.parametrize("var", _ALL_VARS)
def test_deploy_runtime_guard_refuses_each_malformed_password(var: str) -> None:
    env = _valid_env()
    env[var] = _malformed_value(var)
    result = _run_deploy_guard(env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "GUARD_PASSED" not in result.stdout
    assert f"    {var}\n" in result.stderr
    assert "REFUSED" in result.stderr


def test_deploy_runtime_guard_passes_valid_passwords() -> None:
    result = _run_deploy_guard(_valid_env())
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


def test_deploy_runtime_guard_runs_before_anything_is_deployed() -> None:
    text = _DEPLOY_RUNTIME.read_text(encoding="utf-8")
    main_body = text[text.index("\nmain() {") :]
    call_at = main_body.index('guard_password_contract "${repo_root}"')
    assert main_body.index('guard_dogfood_deploy_root "${compose_project}"') < call_at
    for later in (
        "guard_lane_deploy_attribution ",
        "sync_files ",
        "build_images ",
        "restart_services ",
        "bringup_full_stack ",
        "write_registry ",
    ):
        assert call_at < main_body.index(later), (
            f"{later.strip()} must run AFTER the password contract guard"
        )


def test_compose_env_preflight_refuses_a_malformed_password(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text(
        "services:\n"
        "  postgres:\n"
        "    environment:\n"
        "      X: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}\n",
        encoding="utf-8",
    )
    policy = tmp_path / "runtime-policy.env"
    policy.write_text("", encoding="utf-8")

    def run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(_COMPOSE_PREFLIGHT),
                "--lane",
                "dev",
                "--compose-file",
                str(compose),
                "--runtime-policy-env",
                str(policy),
            ],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", **env},
            check=False,
        )

    good = run(_valid_env())
    assert good.returncode == 0, good.stderr

    env = _valid_env()
    env["OMNINODE_RUNTIME_PASSWORD"] = _ALNUM_VALUE
    bad = run(env)
    assert bad.returncode == 1
    assert "    OMNINODE_RUNTIME_PASSWORD\n" in bad.stderr
    assert _ALNUM_VALUE not in bad.stdout + bad.stderr


# --------------------------------------------------------------------------
# AC-2: the consumers and the documentation agree with the contract
# --------------------------------------------------------------------------


def test_migration_runner_login_roles_and_pattern_match_the_contract() -> None:
    text = _RUNNER.read_text(encoding="utf-8")
    loop = text[text.index("for login_role_entry in \\") :]
    loop = loop[: loop.index("; do")]
    names = tuple(re.findall(r'"[a-z_]+:([A-Z_]+)"', loop))
    assert names == _CONTRACT.LOGIN_ROLE_VARS
    body = _function_body(text, "reassert_login_only_role_credential")
    assert "*[!0-9a-fA-F]*)" in body
    assert "preflight_password_contract.py" in body


def test_bootstrap_login_roles_and_patterns_match_the_contract() -> None:
    text = _BOOTSTRAP.read_text(encoding="utf-8")
    role_map = text[text.index("LOGIN_ONLY_ROLE_MAP=(") :]
    role_map = role_map[: role_map.index("\n)")]
    names = tuple(re.findall(r'"[a-z_]+:([A-Z_]+)"', role_map))
    assert names == _CONTRACT.LOGIN_ROLE_VARS
    assert "grep -qE '^[0-9a-fA-F]+$'" in text
    assert "grep -qE '^__REPLACE_WITH_.*__$'" in text
    assert _CONTRACT.HEX.pattern.pattern == "[0-9a-fA-F]+"
    assert _CONTRACT.PLACEHOLDER.pattern == "__REPLACE_WITH_.*__"


_ASSIGNMENT = re.compile(r"^#?\s*([A-Z_]+)=(.*)$")


def _contract_assignments(text: str) -> list[tuple[str, str]]:
    """Every ``VAR=value`` line (commented or not) for a contract variable."""
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line.strip())
        if match and match.group(1) in _CONTRACT.CONTRACT:
            found.append((match.group(1), match.group(2).strip()))
    return found


def _example_value_problems(text: str) -> list[str]:
    problems: list[str] = []
    for var, value in _contract_assignments(text):
        if not value or _CONTRACT.PLACEHOLDER.fullmatch(value):
            continue  # empty or the placeholder: both refused at deploy, by design
        if _CONTRACT.CONTRACT[var].pattern.fullmatch(value) is None:
            problems.append(var)
    return problems


def test_example_value_scanner_positive_control() -> None:
    assert _example_value_problems(
        "OMNINODE_RUNTIME_PASSWORD=judge-omninode-runtime-password\n"
    ) == ["OMNINODE_RUNTIME_PASSWORD"]
    assert _example_value_problems("# VALKEY_PASSWORD=a@b\n") == ["VALKEY_PASSWORD"]
    assert _example_value_problems("POSTGRES_PASSWORD=\n") == []


def _base64_password_advice(text: str) -> list[int]:
    """Line numbers where a base64 generator is advised for a password.

    A base64 generator is fine for a non-password secret (Infisical's auth
    secret), so the line and the one before it must mention a password.
    """
    lines = text.splitlines()
    hits: list[int] = []
    for number, line in enumerate(lines):
        if "rand -base64" not in line:
            continue
        window = (lines[number - 1] if number else "") + line
        if "password" in window.lower():
            hits.append(number + 1)
    return hits


def test_base64_advice_scanner_positive_control() -> None:
    lakshman_before_this_ticket = (
        "# Generate the four infrastructure passwords FRESH for this lane (for example\n"
        "# `openssl rand -base64 32`). Never reuse another lane's values.\n"
    )
    assert _base64_password_advice(lakshman_before_this_ticket) == [2]
    assert (
        _base64_password_advice(
            "# Generate INFISICAL_AUTH_SECRET with:    openssl rand -base64 32\n"
        )
        == []
    )


@pytest.mark.parametrize("path", _ENV_EXAMPLES, ids=lambda p: p.name)
def test_env_example_states_the_enforced_contract(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert _contract_assignments(text), f"{path.name} names no contract variable"
    assert _CONTRACT.render_env_example_block() in text, (
        f"{path.name} does not carry the contract block rendered by "
        "scripts/preflight_password_contract.py"
    )
    assert _example_value_problems(text) == []
    assert _base64_password_advice(text) == [], (
        f"{path.name} advises a base64 generator for a password; its output "
        "('+', '/', '=') breaks the contract"
    )


_COMPOSE_DEFAULT = re.compile(r"\$\{([A-Z_]+):?-([^}]*)\}")


def _compose_default_problems(text: str) -> list[str]:
    problems: list[str] = []
    for var, default in _COMPOSE_DEFAULT.findall(text):
        if var not in _CONTRACT.CONTRACT or not default:
            continue
        if _CONTRACT.CONTRACT[var].pattern.fullmatch(default) is None:
            problems.append(var)
    return problems


def test_compose_default_scanner_positive_control() -> None:
    assert _compose_default_problems("X: ${OMNINODE_RUNTIME_PASSWORD:-not-hex}\n") == [
        "OMNINODE_RUNTIME_PASSWORD"
    ]
    assert _compose_default_problems("X: ${VALKEY_PASSWORD:-}\n") == []


def test_no_compose_default_sits_outside_the_contract() -> None:
    compose_files = sorted((_REPO_ROOT / "docker").rglob("*.yml")) + sorted(
        (_REPO_ROOT / "docker").rglob("*.yaml")
    )
    assert compose_files
    offenders = {
        str(path.relative_to(_REPO_ROOT)): problems
        for path in compose_files
        if (problems := _compose_default_problems(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
