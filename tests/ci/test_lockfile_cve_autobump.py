# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Base-only CVE remediation, single-PR refresh, and workflow trust boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import lockfile_cve_autobump as bump

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/lockfile-cve-autobump.yml"
BRANCH = "bot/lockfile-cve-autobump"
RUNNER_LOCK = "docker/runners/runner-image.lock.json"


def advisory(
    name: str,
    version: str,
    fixes: list[str],
    *,
    vuln_id: str = "PYSEC-2026-4011",
    severity: str = "HIGH",
) -> dict[str, Any]:
    return {
        "package": {"name": name, "version": version, "ecosystem": "PyPI"},
        "vulnerabilities": [
            {
                "id": vuln_id,
                "summary": "Sandbox escape in virtualenv",
                "database_specific": {"severity": severity},
                "affected": [
                    {
                        "package": {"name": name, "ecosystem": "PyPI"},
                        "ranges": [
                            {
                                "type": "ECOSYSTEM",
                                "events": [
                                    {"introduced": "0"},
                                    *({"fixed": v} for v in fixes),
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def payload(*packages: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": [
            {
                "source": {"path": "uv.lock", "type": "lockfile"},
                "packages": list(packages),
            }
        ]
    }


def write_lock(root: Path, version: str) -> None:
    (root / "uv.lock").write_text(
        f'version = 1\nrevision = 1\nrequires-python = ">=3.12"\n\n[[package]]\nname = "virtualenv"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
    )


class ModelFakeRunner:
    def __init__(
        self,
        *,
        open_pr: bool = False,
        same_tree: bool = False,
        upgrade: bool = True,
        remote: bool = False,
        fail: tuple[str, ...] = (),
    ) -> None:
        self.calls: list[list[str]] = []
        self.open_pr = open_pr
        self.same_tree = same_tree
        self.upgrade = upgrade
        self.remote = remote
        self.fail = fail

    def __call__(
        self, cmd: Sequence[str], cwd: Path | None
    ) -> subprocess.CompletedProcess[str]:
        args = list(cmd)
        self.calls.append(args)
        code, out = 0, ""
        if self.fail and tuple(args[: len(self.fail)]) == self.fail:
            return subprocess.CompletedProcess(args, 1, "", "simulated failure")
        if args[:3] == ["gh", "pr", "list"]:
            out = json.dumps(
                [
                    {
                        "number": 42,
                        "url": "https://github.com/org/repo/pull/42",
                        "headRefOid": "old-oid",
                    }
                ]
                if self.open_pr
                else []
            )
        elif args[:3] == ["gh", "label", "list"]:
            # Real gh prints nothing (not "[]") when no label matches.
            out = ""
        elif args[:3] == ["uv", "lock", "--upgrade-package"] and self.upgrade:
            assert cwd is not None
            write_lock(cwd, "20.36.2")
        elif args[:3] == ["git", "diff", "--quiet"]:
            code = 1
        elif args[:2] == ["git", "rev-parse"]:
            out = (
                "new-tree"
                if self.same_tree or args[-1] == "HEAD^{tree}"
                else "old-tree"
            )
        elif args[:2] == ["git", "ls-remote"]:
            out = "old-oid\trefs/heads/" + BRANCH if self.remote else ""
        elif args[:3] == ["gh", "pr", "create"]:
            out = "https://github.com/org/repo/pull/43\n"
        return subprocess.CompletedProcess(args, code, out, "")


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bump.ModelAutobumpConfig:
    monkeypatch.setenv("GH_TOKEN", "writer-installation-token")
    write_lock(tmp_path, "20.36.1")
    for relative in (
        "pyproject.toml",
        RUNNER_LOCK,
        ".github/actions/setup-python-uv/action.yml",
        "scripts/ci/ci_env_digest.py",
        "scripts/ci/ensure_ci_env.sh",
    ):
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    scan = tmp_path / "osv.json"
    scan.write_text(json.dumps(payload(advisory("virtualenv", "20.36.1", ["20.36.2"]))))
    return bump.ModelAutobumpConfig(
        osv_json=scan,
        repo="org/repo",
        event_name="schedule",
        ref="refs/heads/dev",
        default_branch="dev",
        repo_root=tmp_path,
    )


def test_plan_smallest_fix_then_max_across_advisories() -> None:
    first = advisory("Virtual_Env", "20.36.1", ["20.40", "20.36.2", "20.36.1", "bad"])
    first["vulnerabilities"][0]["affected"][0]["package"]["name"] = "virtual-env"
    extra = advisory(
        "virtual-env", "20.36.1", ["20.38", "20.37"], vuln_id="GHSA-second"
    )
    low = advisory("urllib3", "2.7.0", ["2.7.1"], severity="LOW")
    plan = bump.compute_bump_plan(payload(first, extra, low))
    assert plan == (
        bump.ModelBumpItem(
            "virtual-env", "20.36.1", "20.37", ("GHSA-second", "PYSEC-2026-4011")
        ),
    )


@pytest.mark.parametrize("fixes", [["20.36.1"], ["invalid"]])
def test_no_determinable_fix_raises(fixes: list[str]) -> None:
    with pytest.raises(bump.ModelPlanError, match="virtualenv"):
        bump.compute_bump_plan(payload(advisory("virtualenv", "20.36.1", fixes)))


@pytest.mark.parametrize(
    ("field", "value"), [("name", "another-package"), ("ecosystem", "npm")]
)
def test_unrelated_affected_entry_is_not_a_fix(field: str, value: str) -> None:
    item = advisory("virtualenv", "20.36.1", ["20.36.2"])
    item["vulnerabilities"][0]["affected"][0]["package"][field] = value
    with pytest.raises(bump.ModelPlanError, match="virtualenv"):
        bump.compute_bump_plan(payload(item))


def test_opens_one_pr_and_regenerates_runner_lock(
    cfg: bump.ModelAutobumpConfig,
) -> None:
    before = (cfg.repo_root / RUNNER_LOCK).read_bytes()
    runner = ModelFakeRunner()
    assert bump.run_autobump(cfg, runner) == 0
    creates = [c for c in runner.calls if c[:3] == ["gh", "pr", "create"]]
    assert len(creates) == 1
    create = creates[0]
    assert (
        create[create.index("--title") + 1]
        == "fix(OMN-20174): bump virtualenv to fixed versions for lockfile CVE advisories"
    )
    body = create[create.index("--body") + 1]
    assert body.startswith("Ticket: OMN-20174.\n")
    assert "| virtualenv | 20.36.1 | 20.36.2 | PYSEC-2026-4011 |" in body
    assert "base-inherited" in body and "Lockfile CVE Scan is unchanged" in body
    assert ["git", "add", "uv.lock", RUNNER_LOCK] in runner.calls
    assert ["git", "push", "origin", f"HEAD:refs/heads/{BRANCH}"] in runner.calls
    commits = [c for c in runner.calls if "commit" in c]
    assert len(commits) == 1 and "user.name=onexbot-occ-writer[bot]" in commits[0]
    assert (
        "fix(OMN-20174): bump virtualenv for lockfile CVE advisories [bot]"
        in commits[0]
    )
    edits = [c for c in runner.calls if c[:3] == ["gh", "pr", "edit"]]
    assert "priority:landing" in edits[0] and "drain:priority" in edits[0]
    assert (cfg.repo_root / RUNNER_LOCK).read_bytes() != before
    assert any(
        "scripts.ci.check_lockfile_registry_allowlist" in c for c in runner.calls
    )


@pytest.mark.parametrize("same_tree", [False, True])
def test_refresh_never_duplicates(
    cfg: bump.ModelAutobumpConfig, same_tree: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ModelFakeRunner(open_pr=True, same_tree=same_tree)
    assert bump.run_autobump(cfg, runner) == 0
    assert not any(c[:3] == ["gh", "pr", "create"] for c in runner.calls)
    pushes = [c for c in runner.calls if c[:2] == ["git", "push"]]
    if same_tree:
        assert not pushes
        assert "already current" in capsys.readouterr().out
    else:
        assert pushes == [
            [
                "git",
                "push",
                f"--force-with-lease={BRANCH}:old-oid",
                "origin",
                f"HEAD:refs/heads/{BRANCH}",
            ]
        ]
        assert any(
            c[:3] == ["gh", "pr", "edit"] and "--body" in c and "priority:landing" in c
            for c in runner.calls
        )


@pytest.mark.parametrize(
    ("event", "ref"),
    [
        ("pull_request", "refs/heads/dev"),
        ("merge_group", "refs/heads/dev"),
        ("schedule", "refs/heads/feature/x"),
        ("push", "refs/tags/v1"),
    ],
)
def test_non_base_run_has_no_runner_calls(
    cfg: bump.ModelAutobumpConfig,
    event: str,
    ref: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not bump.is_base_inherited_run(event, ref, "dev")
    runner = ModelFakeRunner()
    assert bump.run_autobump(replace(cfg, event_name=event, ref=ref), runner) == 0
    assert runner.calls == []
    assert "not a base-inherited run" in capsys.readouterr().out


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_default_branch_guard_accepts(event: str) -> None:
    assert bump.is_base_inherited_run(event, "refs/heads/dev", "dev")


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_writer_token_fails_closed(
    cfg: bump.ModelAutobumpConfig,
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if token is None:
        monkeypatch.delenv("GH_TOKEN")
    else:
        monkeypatch.setenv("GH_TOKEN", token)
    runner = ModelFakeRunner()
    assert bump.run_autobump(cfg, runner) == 2
    assert runner.calls == []
    assert "::error::" in capsys.readouterr().out


def test_report_only_never_touches_open_pr(
    cfg: bump.ModelAutobumpConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg.osv_json.write_text(
        json.dumps(payload(advisory("urllib3", "2.7.0", ["2.7.1"], severity="LOW")))
    )
    runner = ModelFakeRunner(open_pr=True)
    assert bump.run_autobump(cfg, runner) == 0
    assert runner.calls == []
    assert "no blocking findings" in capsys.readouterr().out


@pytest.mark.parametrize("open_pr", [False, True])
def test_dry_run_is_read_only(
    cfg: bump.ModelAutobumpConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    open_pr: bool,
) -> None:
    monkeypatch.delenv("GH_TOKEN")
    before = (cfg.repo_root / "uv.lock").read_bytes()
    runner = ModelFakeRunner(open_pr=open_pr)
    assert bump.run_autobump(replace(cfg, dry_run=True), runner) == 0
    assert len(runner.calls) == 1 and runner.calls[0][:3] == ["gh", "pr", "list"]
    assert (cfg.repo_root / "uv.lock").read_bytes() == before
    out = capsys.readouterr().out
    assert "PLAN: virtualenv 20.36.1 -> 20.36.2 (PYSEC-2026-4011)" in out
    assert "uv lock --upgrade-package virtualenv" in out and BRANCH in out
    assert "fix(OMN-20174):" in out and ("42" if open_pr else "none") in out


def test_plan_cli_has_no_runner_calls(
    cfg: bump.ModelAutobumpConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bump.main(["plan", "--osv-json", str(cfg.osv_json)]) == 0
    assert "PLAN: virtualenv" in capsys.readouterr().out


def test_upgrade_below_target_fails(
    cfg: bump.ModelAutobumpConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ModelFakeRunner(upgrade=False)
    assert bump.run_autobump(cfg, runner) == 1
    assert "virtualenv" in capsys.readouterr().out
    assert not any(c[:2] == ["git", "push"] or "commit" in c for c in runner.calls)


@pytest.mark.parametrize(
    "failure",
    [
        ("gh", "label", "create"),
        ("uv", "run", "python", "-m", "scripts.ci.check_lockfile_registry_allowlist"),
    ],
)
def test_label_or_registry_failure_is_fatal(
    cfg: bump.ModelAutobumpConfig, failure: tuple[str, ...]
) -> None:
    runner = ModelFakeRunner(fail=failure)
    assert bump.run_autobump(cfg, runner) == 1
    assert not any(c[:2] == ["git", "push"] or "commit" in c for c in runner.calls)


def test_remote_without_pr_uses_lease(cfg: bump.ModelAutobumpConfig) -> None:
    runner = ModelFakeRunner(remote=True)
    assert bump.run_autobump(cfg, runner) == 0
    assert any(c[:3] == ["git", "push", "--force-with-lease"] for c in runner.calls)


def test_non_default_base_requires_explicit_test_mode(
    cfg: bump.ModelAutobumpConfig,
) -> None:
    runner = ModelFakeRunner()
    test_cfg = replace(cfg, base_branch="Proof/OMN-20174")
    assert bump.run_autobump(test_cfg, runner) == 2
    assert runner.calls == []
    assert (
        bump.run_autobump(replace(test_cfg, allow_non_default_base=True), runner) == 0
    )
    assert any(
        "HEAD:refs/heads/bot/lockfile-cve-autobump-proof-omn-20174" in c
        for c in runner.calls
    )


def test_workflow_trust_boundary_and_pins() -> None:
    text = WORKFLOW.read_text()
    doc = yaml.safe_load(text)
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "23,53 * * * *"}]
    assert triggers["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    assert doc["permissions"] == {"contents": "read"}
    assert doc["concurrency"] == {
        "group": "lockfile-cve-autobump",
        "cancel-in-progress": False,
    }
    assert len(doc["jobs"]) == 1
    job = next(iter(doc["jobs"].values()))
    assert (
        job["if"]
        == "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
    )
    assert job["timeout-minutes"] == 20
    steps = job["steps"]
    mint = steps[0]
    assert mint["id"] == "app-token" and mint["uses"].startswith(
        "actions/create-github-app-token@"
    )
    assert "continue-on-error" not in mint
    assert mint["with"]["permission-contents"] == "write"
    assert mint["with"]["permission-pull-requests"] == "write"
    assert "github.token" not in text and "secrets.GITHUB_TOKEN" not in text
    assert "--allow-non-default-base" not in text
    for step in steps:
        token = step.get("with", {}).get("token") or step.get("env", {}).get("GH_TOKEN")
        if token:
            assert token == "${{ steps.app-token.outputs.token }}"
        uses = step.get("uses", "")
        if uses and not uses.startswith("./"):
            assert len(uses.split("@")[1]) == 40
    assert steps[1]["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert (
        steps[1]["with"]["persist-credentials"] is True
        and steps[1]["with"]["fetch-depth"] == 0
    )
    sibling = yaml.safe_load(
        (
            ROOT / ".github/workflows/dependabot-runner-image-lock-refresh.yml"
        ).read_text()
    )
    assert job["runs-on"] == sibling["jobs"]["refresh"]["runs-on"]
    assert mint["uses"] == sibling["jobs"]["refresh"]["steps"][0]["uses"]
    ci_text = (ROOT / ".github/workflows/ci.yml").read_text()
    ci = yaml.safe_load(ci_text)
    assert "check_lockfile_cve evaluate" in ci_text
    scan = next(
        s
        for s in ci["jobs"]["lockfile-cve-scan"]["steps"]
        if s["name"] == "Run osv-scanner"
    )
    new_scan = next(s for s in steps if s["name"] == "Run osv-scanner")
    assert new_scan["env"] == scan["env"]
    assert "set -euo pipefail" in new_scan["run"]
    assert new_scan["run"] == scan["run"].replace(
        "python3 -c", "uv run python -c"
    ).replace("set -uo pipefail", "set -euo pipefail")
    for key in ("PYTHON_VERSION", "UV_VERSION", "CACHE_VERSION"):
        assert doc["env"][key] == ci["env"][key]
    setup = next(
        s
        for s in ci["jobs"]["lockfile-cve-scan"]["steps"]
        if s.get("uses") == "./.github/actions/setup-python-uv"
    )
    assert steps[2] == setup
    assert "github.event_name == 'workflow_dispatch' && inputs.dry_run" in text
