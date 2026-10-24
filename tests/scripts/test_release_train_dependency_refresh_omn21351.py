# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The release PR owns one fenced third-party lock refresh (OMN-21351)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = _ROOT / "config" / "release_train_policy.yaml"
_WORKFLOW = _ROOT / ".github" / "workflows" / "release-train-nightly.yml"


def _load(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load(_ROOT / "scripts" / "ci" / "release_train.py", "release_train_under_test")
classifier = _load(
    _ROOT / "scripts" / "runtime_change_classifier.py",
    "_omnibase_infra_runtime_change_classifier",
)


def _policy() -> Any:
    return rt.load_policy(_POLICY)["omnibase_infra"]


def _lock() -> str:
    entries = [
        ("omnibase-infra", "0.38.80", 'editable = "."'),
        ("OmniBase_Core", "1.0", 'registry = "https://pypi.org/simple"'),
        ("ONEX.Tools", "1.0", 'registry = "https://pypi.org/simple"'),
        ("Zed.Tools", "1.0", 'registry = "https://pypi.org/simple"'),
        ("Alpha__Client", "2.0", 'registry = "https://pypi.org/simple"'),
        ("editable-dep", "1.0", 'editable = "../editable-dep"'),
        ("virtual-dep", "1.0", 'virtual = "../virtual-dep"'),
        ("git-dep", "1.0", 'git = "https://github.com/example/dependency"'),
        ("path-dep", "1.0", 'path = "../path-dep"'),
    ]
    return "version = 1\n" + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ {source} }}\n'
        for name, version, source in entries
    )


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    (tmp_path / "uv.lock").write_bytes(_lock().replace("\n", "\r\n").encode())
    (tmp_path / "pyproject.toml").write_bytes(
        b'[project]\nname = "omnibase-infra"\nversion = "0.38.80"\n'
    )
    return tmp_path


def _never_run(_argv: Sequence[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
    pytest.fail("refresh must not invoke uv")


def test_release_train_dependency_refresh_real_policy() -> None:
    raw = yaml.safe_load(_POLICY.read_text())
    assert raw["dependency_refresh_first_party_prefixes"] == ["omni", "onex"]
    assert len(raw["repos"]) == 7
    policies = rt.load_policy(_POLICY)
    for repo, entry in raw["repos"].items():
        assert entry["dependency_refresh"] == "locked"
        assert policies[repo].dependency_refresh is rt.EnumDependencyRefresh.LOCKED
        assert policies[repo].first_party_lock_prefixes == ("omni", "onex")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dependency_refresh", None),
        ("dependency_refresh", "unknown"),
        ("dependency_refresh", ""),
        ("dependency_refresh", "none"),
        ("dependency_refresh_first_party_prefixes", None),
        ("dependency_refresh_first_party_prefixes", []),
        ("dependency_refresh_first_party_prefixes", [""]),
        ("dependency_refresh_first_party_prefixes", [" "]),
        ("dependency_refresh_first_party_prefixes", "omni"),
    ],
)
def test_release_train_dependency_refresh_invalid_policy(
    tmp_path: Path, field: str, value: Any
) -> None:
    raw = yaml.safe_load(_POLICY.read_text())
    entry = raw if field.endswith("prefixes") else raw["repos"]["omnibase_infra"]
    if value is None:
        del entry[field]
    else:
        entry[field] = value
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    match = "unknown value" if value == "unknown" else field
    with pytest.raises(rt.ReleaseTrainConfigError, match=match):
        rt.load_policy(path)


def test_release_train_dependency_refresh_none_with_note_loads(tmp_path: Path) -> None:
    raw = yaml.safe_load(_POLICY.read_text())
    raw["repos"]["omnibase_infra"].update(
        dependency_refresh="none", dependency_refresh_note="held for resolution repair"
    )
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    policy = rt.load_policy(path)["omnibase_infra"]
    assert policy.dependency_refresh is rt.EnumDependencyRefresh.NONE
    assert policy.dependency_refresh_note == "held for resolution repair"


def test_release_train_dependency_refresh_package_selection() -> None:
    assert rt.refreshable_lock_packages(_lock(), ["OMNI", "onex"]) == [
        "alpha-client",
        "zed-tools",
    ]


def test_release_train_dependency_refresh_version_moves() -> None:
    before = _lock()
    after = before.replace('version = "2.0"', 'version = "2.1"').replace(
        'name = "Zed.Tools"\nversion = "1.0"',
        'name = "zed-tools"\nversion = "1.1"',
    )
    after += '\n[[package]]\nname = "new-package"\nversion = "3.0"\n'
    assert rt.lock_version_moves(before, after) == [
        ("alpha-client", "2.0", "2.1"),
        ("zed-tools", "1.0", "1.1"),
    ]
    assert rt.lock_version_moves(after, before) == [
        ("alpha-client", "2.1", "2.0"),
        ("zed-tools", "1.1", "1.0"),
    ]


@pytest.mark.parametrize("changes", [True, False])
def test_release_train_dependency_refresh_runs_once_with_fenced_argv(
    clone: Path, changes: bool
) -> None:
    manifest = (clone / "pyproject.toml").read_bytes()
    calls = []

    def run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), cwd))
        if changes:
            lock = cwd / "uv.lock"
            lock.write_text(
                lock.read_text().replace('version = "2.0"', 'version = "2.1"')
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = rt.refresh_locked_dependencies(clone, _policy(), run)
    assert calls == [
        (
            [
                "uv",
                "lock",
                "--no-build",
                "--upgrade-package",
                "alpha-client",
                "--upgrade-package",
                "zed-tools",
            ],
            clone,
        )
    ]
    assert result.status is (
        rt.EnumDependencyRefreshStatus.REFRESHED
        if changes
        else rt.EnumDependencyRefreshStatus.UNCHANGED
    )
    assert result.moves == ((("alpha-client", "2.0", "2.1"),) if changes else ())
    assert result.to_dict() == {
        "status": "refreshed" if changes else "unchanged",
        "detail": result.detail,
        "moved_count": 1 if changes else 0,
        "moved": [{"name": "alpha-client", "from": "2.0", "to": "2.1"}]
        if changes
        else [],
    }
    assert (clone / "pyproject.toml").read_bytes() == manifest


def test_release_train_dependency_refresh_default_runner_strips_tokens(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "y")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(args), kwargs["env"]))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(rt.subprocess, "run", run)
    rt.refresh_locked_dependencies(clone, _policy())
    assert len(calls) == 1
    args, env = calls[0]
    assert "GH_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert args[:3] == ["uv", "lock", "--no-build"]


def test_release_train_dependency_refresh_non_public_host_restores_bytes(
    clone: Path,
) -> None:
    original = (clone / "uv.lock").read_bytes()
    manifest = (clone / "pyproject.toml").read_bytes()

    def run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        lock = cwd / "uv.lock"
        lock.write_text(
            lock.read_text().replace(
                'name = "Alpha__Client"\nversion = "2.0"\n'
                'source = { registry = "https://pypi.org/simple" }',
                'name = "Alpha__Client"\nversion = "2.1"\n'
                'source = { registry = "http://mirror.internal:3141/simple" }',
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(
        rt.ReleaseTrainConfigError, match=r"non-public host: mirror\.internal"
    ):
        rt.refresh_locked_dependencies(clone, _policy(), run)
    assert (clone / "uv.lock").read_bytes() == original
    assert (clone / "pyproject.toml").read_bytes() == manifest


def test_release_train_dependency_refresh_failure_restores_bytes(clone: Path) -> None:
    original = (clone / "uv.lock").read_bytes()
    manifest = (clone / "pyproject.toml").read_bytes()
    last_line = "resolution refused: " + "x" * 210

    def run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        (cwd / "uv.lock").write_text("partial failed resolution")
        return subprocess.CompletedProcess(
            argv, 1, "", f"earlier line\n{last_line}\n\n"
        )

    result = rt.refresh_locked_dependencies(clone, _policy(), run)
    assert result.status is rt.EnumDependencyRefreshStatus.SKIPPED
    assert result.detail == f"uv lock failed: {last_line[:200]}"
    assert result.moves == ()
    assert (clone / "uv.lock").read_bytes() == original
    assert (clone / "pyproject.toml").read_bytes() == manifest


@pytest.mark.parametrize(
    "package", ["OmniBase_Core", "ONEX.Tools", "omnibase-infra", "git-dep"]
)
def test_release_train_dependency_refresh_fence_restores_bytes(
    clone: Path, package: str
) -> None:
    original = (clone / "uv.lock").read_bytes()
    manifest = (clone / "pyproject.toml").read_bytes()

    def run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        lock = cwd / "uv.lock"
        lock.write_text(
            re.sub(
                rf'(name = "{re.escape(package)}"\nversion = ")[^"]+',
                r"\g<1>9.9.9",
                lock.read_text(),
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(rt.ReleaseTrainConfigError, match="escaped its fence"):
        rt.refresh_locked_dependencies(clone, _policy(), run)
    assert (clone / "uv.lock").read_bytes() == original
    assert (clone / "pyproject.toml").read_bytes() == manifest


def test_release_train_dependency_refresh_none_never_runs(clone: Path) -> None:
    original = (clone / "uv.lock").read_bytes()
    policy = replace(
        _policy(),
        dependency_refresh=rt.EnumDependencyRefresh.NONE,
        dependency_refresh_note="governed hold",
    )
    result = rt.refresh_locked_dependencies(clone, policy, _never_run)
    assert result.status is rt.EnumDependencyRefreshStatus.SKIPPED
    assert "governed hold" in result.detail
    assert (clone / "uv.lock").read_bytes() == original


def test_release_train_dependency_refresh_no_lock_never_runs(tmp_path: Path) -> None:
    result = rt.refresh_locked_dependencies(tmp_path, _policy(), _never_run)
    assert result.status is rt.EnumDependencyRefreshStatus.SKIPPED
    assert "no uv.lock" in result.detail


def test_release_train_dependency_refresh_no_targets_never_runs(clone: Path) -> None:
    lock = clone / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "omnibase-core"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n'
    )
    original = lock.read_bytes()
    result = rt.refresh_locked_dependencies(clone, _policy(), _never_run)
    assert result.status is rt.EnumDependencyRefreshStatus.UNCHANGED
    assert lock.read_bytes() == original


@pytest.mark.parametrize("status", list(rt.EnumDependencyRefreshStatus))
def test_release_train_dependency_refresh_cli_json(
    clone: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: Any,
) -> None:
    outcome = rt.ModelDependencyRefresh(status, "reported outcome", ())
    monkeypatch.setattr(
        rt, "refresh_locked_dependencies", lambda clone, policy: outcome
    )
    assert (
        rt.main(
            [
                "refresh-lock",
                "--policy",
                str(_POLICY),
                "--repo",
                "omnibase_infra",
                "--clone",
                str(clone),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == outcome.to_dict()


def test_release_train_dependency_refresh_cli_refusal(
    clone: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        rt.main(
            [
                "refresh-lock",
                "--policy",
                str(_POLICY),
                "--repo",
                "undeclared",
                "--clone",
                str(clone),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert not output.out
    assert "::error::" in output.err


def test_release_train_dependency_refresh_workflow_drops_the_token() -> None:
    lines = _WORKFLOW.read_text().splitlines()
    refresh_lines = [line for line in lines if 'release_train.py" refresh-lock' in line]
    assert len(refresh_lines) == 1
    assert "refresh-lock \\" in refresh_lines[0]
    assert "env -u GH_TOKEN -u GITHUB_TOKEN python3" in refresh_lines[0]


def test_release_train_dependency_refresh_workflow_shape() -> None:
    text = _WORKFLOW.read_text()
    assert text.count("refresh-lock") == 1
    workflow = yaml.safe_load(text)
    assert "refresh-lock" not in yaml.safe_dump(workflow["jobs"]["decide"])
    steps = workflow["jobs"]["cut"]["steps"]
    index = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Open the release pull request"
    )
    setup = next(
        step
        for step in steps[:index]
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    )
    assert (
        setup["uses"] == "astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78"
    )
    run = steps[index]["run"]
    refresh_index = run.index("refresh-lock")
    assert (
        run.index("bump-version")
        < refresh_index
        < run.index("git config user.name")
        < run.index("git add")
    )
    case = run[run.index('case "${action}"') : run.index("esac")]
    assert "noop)" in case and "refuse)" in case
    assert "exit 0" in case and "exit 1" in case
    assert run.index("esac") < refresh_index
    assert re.search(
        r'if \[ "\$\{dep_refresh_status\}" = "refreshed" \]; then\s+staged\+=\(uv.lock\)\s+fi',
        run,
    )
    assert run.count("staged+=(uv.lock)") == 1
    assert '--title "chore: release ${TARGET_REPO} ${VERSION} (${TICKET})"' in run
    assert 'git commit -m "chore: release ${TARGET_REPO} ${VERSION} (${TICKET})"' in run
    assert (
        "- Dependency refresh: ${dep_refresh_status} (${dep_refresh_moved} third-party packages moved; ${dep_refresh_detail})"
        in run
    )
    assert "ran once for this run" in run
    assert "same lab evidence as the release" in run


def test_release_train_dependency_refresh_lab_evidence_coverage() -> None:
    policies = rt.load_policy(_POLICY)
    covered = []
    for repo, policy in policies.items():
        assert "uv.lock" not in policy.counted_paths
        if policy.lab_evidence is rt.EnumLabEvidence.NONE:
            assert policy.lab_evidence_note.strip(), repo
        elif policy.dependency_refresh is rt.EnumDependencyRefresh.LOCKED:
            # An empty canonical result proves the trigger's own lane-state
            # supplement covers the lock independently of an external path list.
            runtime_paths = classifier.classify_runtime_paths(
                ["uv.lock"], classifier=lambda paths: [], source_repo=repo
            )
            assert runtime_paths == ["uv.lock"], repo
            assert classifier.is_runtime_affecting(runtime_paths, labels=[]), repo
            covered.append(repo)
    assert covered
    assert rt.is_release_bookkeeping_subject(
        "chore: release omnibase_infra 0.38.80 (OMN-18595)"
    )
