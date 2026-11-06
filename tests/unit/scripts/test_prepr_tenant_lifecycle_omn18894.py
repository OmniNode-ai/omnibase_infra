# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the slot's existing gateway adapter and teardown ordering."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import runpy
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_prepr_teardown_slot import offboard_proof_digest

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
    assert proof["key_hmac_sha256"] == offboard_proof_digest(
        "test-key", "00000000-0000-4000-8000-000000000001"
    )
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
        if argv[:3] == ["docker", "ps", "-a"]:
            return subprocess.CompletedProcess(argv, 0, "slot-api\n", "")
        if argv[:2] == ["docker", "ps"]:
            # The dev-lane server lookup by compose label (OMN-17427).
            service = argv[argv.index("--filter") + 3].rsplit("=", 1)[1]
            return subprocess.CompletedProcess(argv, 0, f"omnibase-infra-{service}", "")
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


@pytest.fixture
def recorded_reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(TEARDOWN.parent))
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "locks"))
    spec = importlib.util.spec_from_file_location("recorded_slot_reaper", TEARDOWN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STAGING_PARENT", tmp_path / "staging")
    return module


def saved_offboard(staging: Path) -> Path:
    credential = staging / "tenant-state" / "credential.env"
    credential.parent.mkdir(parents=True)
    credential.write_text(
        "TENANT_ID=00000000-0000-4000-8000-000000000001\n"
        "TENANT_SLUG=onex-prepr1\nONEX_API_KEY=test-key\n"
    )
    proof = credential.with_suffix(".env.revoked.json")
    proof.write_text(
        json.dumps(
            {
                "tenant_id": "00000000-0000-4000-8000-000000000001",
                "key_hmac_sha256": offboard_proof_digest(
                    "test-key", "00000000-0000-4000-8000-000000000001"
                ),
                "auth_before": 200,
                "auth_after": 401,
            }
        )
    )
    return proof


@pytest.mark.parametrize("installed", [False, True])
def test_recorded_standalone_reaper_finishes_after_gateway_removal(
    recorded_reaper: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
) -> None:
    # Recorded case: expired standalone-1695065, containers already removed,
    # nine databases, ten roles and staging left after the first teardown.
    from tests.unit.scripts import test_prepr_teardown_slot_omn18896 as teardown_tests

    reaper = recorded_reaper
    if installed:
        info = reaper.install_reaper(
            tmp_path / "installed", tmp_path / "units", enable=False
        )
        helper = Path(info["helper"])
        assert helper.read_bytes() == TEARDOWN.read_bytes()
        spec = importlib.util.spec_from_file_location(
            "installed_recorded_reaper", helper
        )
        assert spec and spec.loader
        reaper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reaper)
        monkeypatch.setattr(reaper, "STAGING_PARENT", tmp_path / "staging")
        monkeypatch.setattr(
            teardown_tests.teardown, "PROVISION_DB", reaper.PROVISION_DB
        )
    staging = reaper.STAGING_PARENT / "slot-1"
    saved_offboard(staging)
    host = teardown_tests.FakeHost()
    host.containers[slot_one_project()] = set()
    host.databases = {"control"} | {f"db{i}_prepr1" for i in range(9)}
    host.roles = {"postgres"} | {f"role{i}_prepr1" for i in range(10)}
    host.topics = {name for name in host.topics if not name.startswith("prepr1.")}
    host.groups = {name for name in host.groups if not name.startswith("prepr1.")}
    host.valkey[1] = 0
    host.volumes[slot_one_project()] = set()
    host.images[slot_one_project()] = set()
    original_host = host

    def runner(argv: list[str], env: Any = None) -> Any:
        if argv[:1] == ["bash"] and "revoke" in argv:
            return subprocess.CompletedProcess(
                argv,
                1,
                "",
                "FATAL: no running onex-api container in compose project "
                + slot_one_project(),
            )
        return original_host(argv, env)

    record = reaper.claim(
        tmp_path, 1, "recorded-run", "standalone-1695065", str(staging)
    )
    record.update(expires=time.time() - 1, owner_pid=99999999, owner_start="gone")
    reaper.write_record(tmp_path, 1, record)

    def shell_adapter(argv: list[str], **kwargs: Any) -> Any:
        assert Path(argv[1]).name == "prepr_teardown_slot.sh"
        report = Path(argv[argv.index("--report-out") + 1])
        code = reaper.run_teardown(
            reaper.policy.SLOTS[1],
            runner=runner,
            staging_root=staging,
            base_env={"POSTGRES_USER": "postgres"},
            report_path=report,
        )
        return subprocess.CompletedProcess(argv, code)

    monkeypatch.setattr(reaper.subprocess, "run", shell_adapter)
    result = reaper.reap_slot(tmp_path, 1)
    assert result["verdict"] == "REAPED", result
    report = json.loads((tmp_path / "slot-1.teardown.json").read_text())
    assert report["verdict"] == "CLEAN"
    assert report["tenant_authentication"]["auth_before"] == 200
    assert report["tenant_authentication"]["auth_after"] == 401
    assert all(value == 0 for value in report["residue"].values())
    assert all(value > 0 for value in report["controls"].values())
    assert host.databases == {"control"} and host.roles == {"postgres"}
    assert host.containers["omnibase-infra"] == {"d1", "d2"}
    assert host.valkey[0] == 5 and host.valkey[2] == 1
    assert not staging.exists() and not reaper.record_path(tmp_path, 1).exists()


@pytest.mark.parametrize(
    "fault", ["missing", "key", "tenant", "before", "after", "malformed"]
)
def test_removed_gateway_without_bound_offboard_evidence_retains_resources(
    recorded_reaper: Any, tmp_path: Path, fault: str
) -> None:
    from tests.unit.scripts.test_prepr_teardown_slot_omn18896 import FakeHost

    reaper = recorded_reaper
    staging = reaper.STAGING_PARENT / "slot-1"
    proof = saved_offboard(staging)
    if fault == "missing":
        proof.unlink()
    elif fault == "malformed":
        proof.write_text("[]")
    else:
        data = json.loads(proof.read_text())
        field, value = {
            "key": ("key_hmac_sha256", "other-key"),
            "tenant": ("tenant_id", "another-tenant"),
            "before": ("auth_before", 401),
            "after": ("auth_after", 200),
        }[fault]
        data[field] = value
        proof.write_text(json.dumps(data))
    host = FakeHost()
    host.containers[slot_one_project()] = set()
    td = reaper._Teardown(reaper.selectors_for(reaper.policy.SLOTS[1]), host, {})
    assert td.revoke_tenant(staging) is False
    assert td.errors
    assert staging.exists()
    assert not any("--drop" in call or "FLUSHDB" in call for call in host.calls)


@pytest.mark.parametrize("listing", ["stopped-api", None])
def test_saved_offboard_proof_never_replaces_live_check_with_containers_or_unreadable_listing(
    recorded_reaper: Any, listing: str | None, tmp_path: Path
) -> None:
    reaper = recorded_reaper
    staging = reaper.STAGING_PARENT / "slot-1"
    saved_offboard(staging)
    calls: list[list[str]] = []

    def runner(argv: list[str], env: Any = None) -> Any:
        calls.append(argv)
        if argv[:3] == ["docker", "ps", "-a"]:
            return subprocess.CompletedProcess(
                argv, int(listing is None), listing or "", "unreadable"
            )
        assert "revoke" in argv
        return subprocess.CompletedProcess(argv, 1, "", "no running gateway")

    td = reaper._Teardown(reaper.selectors_for(reaper.policy.SLOTS[1]), runner, {})
    assert td.revoke_tenant(staging) is False
    assert td.tenant_authentication is None
    assert len(calls) == (1 if listing is None else 2)


@pytest.mark.parametrize(
    "report_kind", ["recorded", "nonzero-clean", "malformed", "missing", "list"]
)
def test_reaper_reports_safe_teardown_cause_and_never_releases_incomplete_lease(
    recorded_reaper: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_kind: str,
) -> None:
    reaper = recorded_reaper
    staging = reaper.STAGING_PARENT / "slot-1"
    record = reaper.claim(
        tmp_path, 1, "recorded-run", "standalone-1695065", str(staging)
    )
    record.update(expires=time.time() - 1, owner_pid=99999999, owner_start="gone")
    reaper.write_record(tmp_path, 1, record)
    report_path = tmp_path / "slot-1.teardown.json"
    report_path.write_text('{"verdict": "CLEAN"}')
    raw_secret = "DO-NOT-PUBLISH-THIS-STDERR"

    def shell_adapter(argv: list[str], **kwargs: Any) -> Any:
        assert not report_path.exists(), "stale CLEAN report survived into retry"
        if report_kind == "recorded":
            report_path.write_text(
                json.dumps(
                    {
                        "verdict": "INCOMPLETE",
                        "errors": [
                            "revoke slot tenant and prove authentication refusal: exit 1: "
                            "FATAL: no running onex-api container in compose project "
                            + slot_one_project(),
                            raw_secret,
                        ],
                        "residue": {"databases": 9, "roles": 10, "staging_root": 1},
                        "controls": {"dev_containers": 2},
                    }
                )
            )
        elif report_kind == "nonzero-clean":
            report_path.write_text('{"verdict": "CLEAN"}')
        elif report_kind == "malformed":
            report_path.write_text("{")
        elif report_kind == "list":
            report_path.write_text("[]")
        return subprocess.CompletedProcess(argv, 13)

    monkeypatch.setattr(reaper.subprocess, "run", shell_adapter)
    result = reaper.reap_slot(tmp_path, 1)
    assert result["verdict"] == "INCOMPLETE"
    assert reaper.read_record(tmp_path, 1) == record
    diagnostic = result["teardown"]
    assert diagnostic["report_path"] == str(report_path)
    assert diagnostic["returncode"] == 13
    assert raw_secret not in json.dumps(result)
    if report_kind == "recorded":
        assert diagnostic["reason"] == "slot gateway is not running"
        assert diagnostic["residue"] == {"databases": 9, "roles": 10, "staging_root": 1}
        assert diagnostic["controls"] == {"dev_containers": 2}


def test_dependency_install_failure_preserves_the_previous_reaper_unit(
    recorded_reaper: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = tmp_path / "units"
    units.mkdir()
    service = units / "onex-prepr-slot-reaper.service"
    service.write_text("previous installed interpreter\n")
    calls: list[list[str]] = []

    def failed_sync(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        assert argv[:3] == ["uv", "sync", "--frozen"]
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(Path(argv[-1]) / ".venv")
        raise subprocess.CalledProcessError(17, argv)

    monkeypatch.setattr(recorded_reaper.subprocess, "run", failed_sync)
    with pytest.raises(subprocess.CalledProcessError):
        recorded_reaper.install_reaper(tmp_path / "installed", units, enable=True)
    assert len(calls) == 1
    assert service.read_text() == "previous installed interpreter\n"
    assert not (units / "onex-prepr-slot-reaper.timer").exists()


def test_contract_handler_never_writes_clean_for_missing_readback_axes(
    tmp_path: Path,
    recorded_reaper: Any,
) -> None:
    from omnibase_infra.handlers.handler_prepr_teardown_slot import (
        HandlerPreprTeardownSlot,
    )
    from omnibase_infra.models.model_prepr_teardown_request import (
        ModelPreprTeardownRequest,
    )

    report = tmp_path / "report.json"
    from omnibase_infra.handlers.handler_prepr_teardown_slot import (
        create_container,
    )

    container = create_container(
        lambda *_: pytest.fail("report rendering ran a command"),
        {},
        recorded_reaper._files(),
    )
    result = container.get_service(HandlerPreprTeardownSlot).handle(
        ModelPreprTeardownRequest(
            operation="write_report",
            compose_project=slot_one_project(),
            slot_token="prepr1",
            staging_root=tmp_path,
            repo_root=ROOT,
            report_path=report,
        )
    )
    assert result.success is False and result.exit_code == 13
    assert json.loads(report.read_text())["verdict"] == "INCOMPLETE"


def _boundary_report(source: str) -> Any:
    from omnibase_core.models.nodes.node_boundary_import_check.model_boundary_import_check_input import (
        ModelBoundaryImportCheckInput,
    )
    from omnibase_core.models.nodes.node_boundary_import_check.model_boundary_import_source_file import (
        ModelBoundaryImportSourceFile,
    )
    from omnibase_core.nodes.node_boundary_import_check_compute.handler import (
        NodeBoundaryImportCheckCompute,
    )

    return NodeBoundaryImportCheckCompute().handle(
        ModelBoundaryImportCheckInput(
            repo_packages=("omnibase_infra",),
            node_packages=("omnibase_infra.nodes.node_prepr_teardown_slot",),
            files=(
                ModelBoundaryImportSourceFile(
                    path="scripts/runtime_build/prepr_teardown_slot.py", source=source
                ),
            ),
        )
    )


def test_cli_teardown_composition_has_no_node_boundary_imports() -> None:
    report = _boundary_report(TEARDOWN.read_text())
    assert report.overall_status == "PASS", report


def test_recorded_private_registry_node_boundary_import_still_refuses() -> None:
    # The recorded push refusal was a true alarm, including a dynamic import.
    report = _boundary_report(
        "def _node():\n"
        "    return importlib.import_module(\n"
        '        "omnibase_infra.nodes.node_prepr_teardown_slot.registry.registry_prepr_teardown_slot"\n'
        "    )\n"
    )
    assert report.overall_status == "FAIL", report
    assert len(report.findings) == 1
    assert "outside->node" in report.findings[0].message
