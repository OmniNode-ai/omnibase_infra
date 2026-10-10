# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the slot's existing gateway adapter and teardown ordering."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import runpy
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "scripts/smoke/smoke_delegation.sh"
VERIFY = ROOT / "scripts/runtime_build/prepr_verify_lane.sh"
TEARDOWN = ROOT / "scripts/runtime_build/prepr_teardown_slot.py"
SLOT_POLICY = ROOT / "scripts/runtime_build/prepr_slot_policy.py"

pytestmark = pytest.mark.unit


def slot_one_project() -> str:
    spec = importlib.util.spec_from_file_location("prepr_slot_policy", SLOT_POLICY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    project: str = module.resolve_slot(1).compose_project
    return project


class GatewayResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int) -> None:
        super().__init__(body)
        self.status = status


def revoke_program() -> str:
    match = re.search(r"<<'PYREVOKE'\n(.*?)\nPYREVOKE", SMOKE.read_text(), re.S)
    assert match, "the gateway adapter must expose an offboard proof"
    return match.group(1)


def test_revocation_precedes_container_and_database_removal() -> None:
    source = TEARDOWN.read_text()
    destroy = source.split("    def destroy(", 1)[1].split("    def readback(", 1)[0]
    assert destroy.index("revoke_tenant(") < destroy.index('planned["containers"]')
    assert destroy.index("revoke_tenant(") < destroy.index('"--drop"')
    assert (
        "return"
        in destroy.split("revoke_tenant(", 1)[1].split('planned["containers"]', 1)[0]
    ), "an unproven revoke must preserve the gateway and credential for retry"


def test_bringup_mints_through_existing_adapter_after_gateway_readiness() -> None:
    source = VERIFY.read_text()
    assert "--tenant-lifecycle mint" in source
    mint = source.index("--tenant-lifecycle mint")
    assert mint > source.index('compose "${PROFILES[@]}" up')
    assert mint < source.index("# 12. THE SLOT DESCRIPTOR")
    assert 'LAB_TENANT_SLUG="onex-${DB_SLOT}"' in source


def test_gateway_gets_private_admin_sentinels() -> None:
    source = VERIFY.read_text()
    assert "TENANT_BOOTSTRAP_ADMIN_SECRET" in source
    assert "TENANT_OFFBOARD_ADMIN_SECRET" in source
    assert source.index("secrets.token_hex(") > source.index(
        'source "${OMNIBASE_OPERATOR_ENV_FILE}"'
    )


def test_slot_cloud_migration_uses_its_own_database_and_volume() -> None:
    source = (ROOT / "docker/docker-compose.prepr.yml").read_text()
    assert 'DB_NAME: "omninode_cloud_${ONEX_DB_SLOT' in source
    assert 'DB_USER: "role_omninode_${ONEX_DB_SLOT' in source
    assert "prepr_cloud_migrations:/work" in source


@pytest.mark.parametrize("after", [401, 403])
def test_revoke_requires_live_success_then_authentication_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    after: int,
) -> None:
    calls = run_revoke(monkeypatch, tmp_path, after=after)
    assert [call.get_method() for call in calls] == ["GET", "POST", "GET"]
    assert calls[1].full_url.endswith(
        "/admin/tenants/00000000-0000-4000-8000-000000000001/revoke"
    )
    assert calls[1].get_header("X-admin-secret") == "test-admin"
    assert calls[0].get_header("X-api-key") == "test-key"
    output = capsys.readouterr().out
    assert '"revoked": true' in output
    assert "test-key" not in output
    assert "test-admin" not in output


@pytest.mark.parametrize("after", [200, 429, 500, None])
def test_revoke_does_not_accept_success_outage_or_rate_limit_as_revocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, after: int | None
) -> None:
    with pytest.raises((SystemExit, urllib.error.URLError)):
        run_revoke(monkeypatch, tmp_path, after=after)


def test_revoke_does_not_accept_an_already_invalid_key_as_positive_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        run_revoke(monkeypatch, tmp_path, after=401, before=401)


def run_revoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    after: int | None,
    before: int = 200,
) -> list[Any]:
    state = tmp_path / "credential.env"
    state.write_text(
        "TENANT_ID=00000000-0000-4000-8000-000000000001\n"
        "TENANT_SLUG=onex-prepr1\nONEX_API_KEY=test-key\n"
    )
    monkeypatch.setenv("SMOKE_TENANT_STATE_FILE", str(state))
    monkeypatch.setenv("SMOKE_TENANT_SLUG", "onex-prepr1")
    monkeypatch.setenv("TENANT_OFFBOARD_ADMIN_SECRET", "test-admin")
    calls: list[Any] = []

    def urlopen(request: Any, timeout: int) -> Any:
        calls.append(request)
        status = before if len(calls) == 1 else (200 if len(calls) == 2 else after)
        if status is None:
            raise urllib.error.URLError("unreachable")
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "refused", {}, None)
        return GatewayResponse(json.dumps({"api_keys_revoked": 1}).encode(), status)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    program = tmp_path / "gateway_revoke.py"
    program.write_text(revoke_program())
    runpy.run_path(str(program))
    return calls


def test_revoke_retry_uses_bound_proof_and_still_requires_live_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_revoke(monkeypatch, tmp_path, after=401)
    with pytest.raises(SystemExit) as stopped:
        run_revoke(monkeypatch, tmp_path, after=401, before=401)
    assert stopped.value.code == 0
    proof = json.loads((tmp_path / "credential.env.revoked.json").read_text())
    assert proof["key_sha256"] == hashlib.sha256(b"test-key").hexdigest()
    with pytest.raises(SystemExit) as refused:
        run_revoke(monkeypatch, tmp_path, after=401, before=200)
    assert refused.value.code != 0


@pytest.mark.parametrize("revoke_ok", [True, False])
def test_teardown_adapter_preserves_resources_when_offboard_fails(
    tmp_path: Path, revoke_ok: bool
) -> None:
    spec = importlib.util.spec_from_file_location("prepr_teardown_slot", TEARDOWN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = tmp_path / "tenant-state"
    state.mkdir()
    (state / "credential.env").write_text("placeholder")
    commands: list[list[str]] = []

    def runner(
        argv: list[str], env: dict[str, str] | None
    ) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        is_revoke = "revoke" in argv
        if is_revoke:
            assert env and env["COMPOSE_PROJECT"] == slot_one_project()
        proof = (
            json.dumps(
                {
                    "tenant_id": "slot-tenant",
                    "revoked": True,
                    "auth_before": 200,
                    "auth_after": 401,
                }
            )
            if is_revoke
            else ""
        )
        return subprocess.CompletedProcess(
            argv, 0 if revoke_ok or not is_revoke else 1, proof, "offboard refused"
        )

    td = module._Teardown(module.selectors_for(module.policy.SLOTS[1]), runner, {})
    planned = {
        "containers": ["slot-api"],
        "groups": [],
        "topics": [],
        "volumes": [],
        "images": [],
    }
    td.destroy(planned, tmp_path)
    assert commands[0][-2:] == ["--tenant-lifecycle", "revoke"]
    if revoke_ok:
        assert commands[1] == ["docker", "rm", "-f", "slot-api"]
        assert not tmp_path.exists()
    else:
        assert len(commands) == 1
        assert tmp_path.exists()
        assert td.errors


@pytest.mark.parametrize(
    ("control_after", "ok"),
    [("1:" + "a" * 32, True), ("1:" + "b" * 32, False), ("0:" + "a" * 32, False)],
)
def test_slot_mint_adapter_reads_shared_postgres_and_refuses_control_drift(
    tmp_path: Path, control_after: str, ok: bool
) -> None:
    credential = tmp_path / "credential.env"
    credential.write_text(
        "TENANT_ID=00000000-0000-4000-8000-000000000001\n"
        "TENANT_SLUG=onex-prepr1\nONEX_API_KEY=test-key\n"
    )
    shim = tmp_path / "docker"
    shim.write_text(
        f"#!{sys.executable}\n"
        """
import io, json, os, pathlib, sys, urllib.request
args = sys.argv[1:]
with open(os.environ["TEST_DOCKER_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "ps":
    service = next(arg.split("=", 2)[-1] for arg in args if "compose.service=" in arg)
    print("shared-db" if service == "postgres" else "slot-" + service)
elif "psql" in args:
    counter = pathlib.Path(os.environ["TEST_CONTROL_COUNTER"])
    seen = counter.exists()
    counter.write_text("seen")
    print(os.environ["TEST_CONTROL_AFTER"] if seen else "1:" + "a" * 32)
else:
    def urlopen(request, timeout):
        response = io.BytesIO(b"{}")
        response.status = 200
        return response
    urllib.request.urlopen = urlopen
    program = pathlib.Path(os.environ["TEST_API_PROGRAM"])
    program.write_text(sys.stdin.read())
    import runpy
    runpy.run_path(str(program))
"""
    )
    shim.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        COMPOSE_PROJECT=slot_one_project(),
        TENANT_STATE_FILE=str(credential),
        TEST_DOCKER_LOG=str(log),
        TEST_CONTROL_COUNTER=str(tmp_path / "control"),
        TEST_CONTROL_AFTER=control_after,
        TEST_API_PROGRAM=str(tmp_path / "api.py"),
    )
    result = subprocess.run(
        ["bash", str(SMOKE), "--target", "compose", "--tenant-lifecycle", "mint"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) is ok, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(
        "label=com.docker.compose.project=omnibase-infra" in call
        and "label=com.docker.compose.service=postgres" in call
        for call in calls
    )
    assert len([call for call in calls if "psql" in call]) == 2
    assert "test-key" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("status", "rows", "ok"),
    [
        (200, [{"correlation_id": "probe-correlation"}], True),
        (200, [], False),
        (200, [{"comment": "probe-correlation"}], False),
        (200, [{"correlation_id": "not-probe-correlation"}], False),
        (500, [{"correlation_id": "probe-correlation"}], False),
    ],
)
def test_slot_canary_reader_requires_the_exact_correlated_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    rows: list[dict[str, str]],
    ok: bool,
) -> None:
    match = re.search(
        r"api_python_env [^\n]*SMOKE_REQUIRE_SLOT_PROOF <<'PY'\n(.*?)\nPY",
        SMOKE.read_text(),
        re.S,
    )
    assert match
    program = tmp_path / "reader.py"
    program.write_text(match.group(1))
    monkeypatch.setenv("ONEX_API_KEY", "test-key")
    monkeypatch.setenv("SMOKE_REQUIRE_SLOT_PROOF", "1")
    monkeypatch.setenv("SMOKE_CORRELATION_ID", "probe-correlation")

    def urlopen(request: Any, timeout: int) -> Any:
        response = GatewayResponse(json.dumps({"delegations": rows}).encode(), status)
        if status != 200:
            raise urllib.error.HTTPError(
                request.full_url, status, "failed", {}, response
            )
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if ok:
        runpy.run_path(str(program))
    else:
        with pytest.raises(SystemExit):
            runpy.run_path(str(program))


def test_lifecycle_falsifiers_are_wired_to_ci_and_precommit() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    path = "tests/unit/scripts/test_prepr_tenant_lifecycle_omn18894.py"
    assert any(
        path in step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    )
    hook = next(
        hook
        for repo in hooks["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "prepr-tenant-lifecycle"
    )
    assert path in hook["entry"]
    assert hook["pass_filenames"] is False
