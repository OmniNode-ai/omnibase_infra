# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Group planning and squash-tree proof against local git repos (OMN-18893)."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GROUP_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_group.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


group = _load("prepr_pool_group", GROUP_PY)


def _candidate(
    number: int,
    *files: str,
    labels: tuple[str, ...] = (),
    repo: str = "omnibase_infra",
) -> Any:
    return group.Candidate(number, f"{number:040x}", tuple(files), labels, repo)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("docker/migrations/001_example.sql", "migration"),
        ("config/migration_classes.yaml", "migration"),
        ("src/omnibase_infra/nodes/example/contract.yaml", "contract or topic schema"),
        ("src/omnibase_infra/topics/example.py", "contract or topic schema"),
    ],
)
def test_exclusion_for_migrations_contracts_and_topics(path: str, reason: str) -> None:
    why = group.exclusion(_candidate(1, path))
    assert why is not None
    assert reason in why and path in why


@pytest.mark.unit
def test_exclusion_for_hold_label() -> None:
    why = group.exclusion(_candidate(1, "src/example.py", labels=("hold:auto-merge",)))
    assert why is not None and "hold:auto-merge" in why


@pytest.mark.unit
def test_clean_candidate_is_not_excluded() -> None:
    assert group.exclusion(_candidate(1, "src/example.py")) is None


@pytest.mark.unit
def test_empty_files_are_excluded_with_a_reason() -> None:
    assert group.exclusion(_candidate(1)) == "no changed files read"


@pytest.mark.unit
def test_three_disjoint_candidates_keep_priority_order() -> None:
    candidates = [_candidate(n, f"src/file_{n}.py") for n in (3, 1, 2)]
    plan = group.plan_groups(candidates)
    assert plan.groups == [candidates]
    assert plan.solo == []


@pytest.mark.unit
def test_shared_lockfile_candidates_never_share_a_group() -> None:
    candidates = [
        _candidate(1, "uv.lock", "src/one.py"),
        _candidate(2, "uv.lock", "src/two.py"),
        _candidate(3, "src/three.py"),
        _candidate(4, "src/four.py"),
    ]
    plan = group.plan_groups(candidates, max_size=2)
    assert plan.groups == [
        [candidates[0], candidates[2]],
        [candidates[1], candidates[3]],
    ]
    assert plan.solo == []


@pytest.mark.unit
def test_excluded_partner_leaves_clean_candidate_solo() -> None:
    clean = _candidate(1, "src/one.py")
    excluded = _candidate(2, "docker/migrations/002_example.sql")
    plan = group.plan_groups([clean, excluded])
    assert plan.groups == []
    reasons = {candidate.number: why for candidate, why in plan.solo}
    assert set(reasons) == {1, 2}
    assert "no disjoint partner" in reasons[1]
    assert "migration" in reasons[2]


@pytest.mark.unit
def test_max_size_caps_groups_and_leaves_last_member_solo() -> None:
    candidates = [_candidate(n, f"src/file_{n}.py") for n in range(1, 6)]
    plan = group.plan_groups(candidates, max_size=2)
    assert plan.groups == [candidates[:2], candidates[2:4]]
    assert len(plan.solo) == 1
    assert plan.solo[0][0] == candidates[4]
    assert "no disjoint partner" in plan.solo[0][1]


@pytest.mark.unit
@pytest.mark.parametrize("max_size", [-1, 0, 1])
def test_max_size_below_two_is_rejected(max_size: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        group.plan_groups([], max_size=max_size)


@pytest.mark.unit
@pytest.mark.parametrize(("size", "mid"), [(4, 2), (5, 3)])
def test_halves_keep_order(size: int, mid: int) -> None:
    members = [_candidate(n, f"src/file_{n}.py") for n in range(size, 0, -1)]
    assert group.halves(members) == (members[:mid], members[mid:])


@pytest.mark.unit
def test_one_member_cannot_be_bisected() -> None:
    with pytest.raises(ValueError, match="single PR"):
        group.halves([_candidate(1, "src/one.py")])


def _configure(git: Any) -> None:
    git("config", "user.name", "Group Test")
    git("config", "user.email", "group-test@example.invalid")
    git("config", "commit.gpgsign", "false")


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str, list[Any], Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = group.make_git(repo)
    git("init", "-q", "--initial-branch=dev")
    _configure(git)
    for name in ("one.txt", "two.txt", "three.txt"):
        (repo / name).write_text("base\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    members = []
    for number, name in enumerate(("one.txt", "two.txt", "three.txt", "one.txt"), 1):
        git("switch", "-qc", f"member-{number}", base)
        (repo / name).write_text(f"member {number}\n", encoding="utf-8")
        git("add", name)
        git("commit", "-qm", f"member {number}")
        members.append(group.Candidate(number, git("rev-parse", "HEAD"), (name,)))
    git("switch", "-q", "dev")
    return repo, base, members[:3], members[3]


def _squash_members(git: Any, members: list[Any]) -> list[tuple[int, str]]:
    landed = []
    for member in members:
        git("merge", "--squash", member.head)
        git("commit", "-qm", f"land PR {member.number}")
        landed.append((member.number, git("rev-parse", "HEAD")))
    return landed


@pytest.mark.unit
def test_group_tree_matches_sequential_squashes_in_a_second_clone(
    git_repo: tuple[Path, str, list[Any], Any], tmp_path: Path
) -> None:
    repo, base, members, _conflict = git_repo
    git = group.make_git(repo)
    built = group.build_group(git, base, members)
    clone = tmp_path / "sequential"
    git("clone", "-q", str(repo), str(clone))
    clone_git = group.make_git(clone)
    _configure(clone_git)
    _squash_members(clone_git, members)
    assert built.tree == clone_git("rev-parse", "HEAD^{tree}")
    assert git("rev-parse", f"{built.commit}^{{tree}}") == built.tree
    assert built.base == base
    assert [(pr, head) for pr, head, _, _ in built.steps] == [
        (c.number, c.head) for c in members
    ]


@pytest.mark.unit
def test_group_commits_are_deterministic(
    git_repo: tuple[Path, str, list[Any], Any],
) -> None:
    repo, base, members, _conflict = git_repo
    git = group.make_git(repo)
    first = group.build_group(git, base, members)
    git("config", "user.name", "Another Fixture User")
    git("config", "user.email", "another@example.invalid")
    second = group.build_group(git, base, members)
    assert first.commit == second.commit
    assert first.steps == second.steps


@pytest.mark.unit
def test_conflicting_member_error_names_its_pr(
    git_repo: tuple[Path, str, list[Any], Any],
) -> None:
    repo, base, members, conflict = git_repo
    with pytest.raises(RuntimeError, match=f"{conflict.key} does not squash cleanly"):
        group.build_group(group.make_git(repo), base, [members[0], conflict])


@pytest.mark.unit
def test_params_include_ordered_members_and_only_test_and_source_python_paths() -> None:
    members = [
        _candidate(
            3,
            "tests/unit/test_three.py",
            "tests/unit/helpers.py",
            "tests/unit/test_three.json",
            "src/pkg/three.py",
            "src/pkg/config.yaml",
            "scripts/test_outside.py",
        ),
        _candidate(1, "tests/integration/test_one.py", "src/pkg/one.py"),
    ]
    build = group.GroupBuild(
        base="a" * 40,
        steps=[
            (c.number, c.head, f"{i:040x}", f"{i + 10:040x}")
            for i, c in enumerate(members, 1)
        ],
    )
    params = dict(
        item.split("=", 1) for item in shlex.split(group.params_text(build, members))
    )
    assert params["INFRA_GROUP"] == f"3:{members[0].head} 1:{members[1].head}"
    assert params["INFRA_GROUP_BASE"] == build.base
    assert params["INFRA_GROUP_TREE"] == build.tree
    assert params["TESTS"].split() == [
        "omnibase_infra:tests/integration/test_one.py",
        "omnibase_infra:tests/unit/test_three.py",
    ]
    assert params["ID_FILES"].split() == [
        "omnibase_infra:pkg/one.py",
        "omnibase_infra:pkg/three.py",
    ]


@pytest.mark.unit
def test_omnimarket_candidate_key_and_params_use_market_prefix() -> None:
    members = [
        _candidate(
            3,
            "tests/unit/test_three.py",
            "src/omnimarket/three.py",
            repo="omnimarket",
        ),
        _candidate(
            1,
            "tests/integration/test_one.py",
            "src/omnimarket/one.py",
            repo="omnimarket",
        ),
    ]
    build = group.GroupBuild(
        base="a" * 40,
        steps=[
            (c.number, c.head, f"{i:040x}", f"{i + 10:040x}")
            for i, c in enumerate(members, 1)
        ],
    )
    params = dict(
        item.split("=", 1) for item in shlex.split(group.params_text(build, members))
    )
    assert members[0].key == "omnimarket#3"
    assert params["MARKET_GROUP"] == f"3:{members[0].head} 1:{members[1].head}"
    assert params["MARKET_GROUP_BASE"] == build.base
    assert params["MARKET_GROUP_TREE"] == build.tree
    assert "INFRA_GROUP" not in params
    assert params["TESTS"].split() == [
        "omnimarket:tests/integration/test_one.py",
        "omnimarket:tests/unit/test_three.py",
    ]
    assert params["ID_FILES"].split() == [
        "omnimarket:omnimarket/one.py",
        "omnimarket:omnimarket/three.py",
    ]


@pytest.mark.unit
def test_params_reject_members_from_different_repositories() -> None:
    members = [
        _candidate(1, "src/one.py"),
        _candidate(2, "src/two.py", repo="omnimarket"),
    ]
    build = group.GroupBuild(
        base="a" * 40,
        steps=[
            (c.number, c.head, f"{i:040x}", f"{i + 10:040x}")
            for i, c in enumerate(members, 1)
        ],
    )
    with pytest.raises(ValueError, match="mix repositories"):
        group.params_text(build, members)


def _candidates_json(tmp_path: Path, members: list[Any]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [{"pr": c.number, "head": c.head, "files": list(c.files)} for c in members]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_verify_landed_exact_consecutive_squashes(
    git_repo: tuple[Path, str, list[Any], Any],
) -> None:
    repo, base, members, _conflict = git_repo
    git = group.make_git(repo)
    built = group.build_group(git, base, members)
    landed = _squash_members(git, members)
    verdict = group.verify_landed(
        git, built.tree, landed, {c.number: c.files for c in members}
    )
    assert verdict.exact is True
    assert verdict.consecutive is True
    assert verdict.tip == landed[-1][1]
    assert verdict.tip_tree == built.tree
    assert verdict.drift == []
    assert len(verdict.lines) == 3
    assert all(line.endswith(" exact") for line in verdict.lines)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed_path", "label", "exit_code"),
    [
        ("unrelated.txt", "base-drift", 0),
        ("one.txt", "MISMATCH", 1),
    ],
)
def test_verify_landed_drift_and_member_mismatch_with_cli(
    git_repo: tuple[Path, str, list[Any], Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    changed_path: str,
    label: str,
    exit_code: int,
) -> None:
    repo, base, members, _conflict = git_repo
    git = group.make_git(repo)
    built = group.build_group(git, base, members)
    if label == "base-drift":
        (repo / changed_path).write_text("unrelated base advance\n", encoding="utf-8")
        git("add", changed_path)
        git("commit", "-qm", "unrelated base advance")
    landed = _squash_members(git, members)
    if label == "MISMATCH":
        (repo / changed_path).write_text("different landed content\n", encoding="utf-8")
        git("add", changed_path)
        git("commit", "--amend", "-qm", "last member with unexpected content")
        landed[-1] = (members[-1].number, git("rev-parse", "HEAD"))
    verdict = group.verify_landed(
        git, built.tree, landed, {c.number: c.files for c in members}
    )
    assert verdict.exact is False
    assert verdict.consecutive is True
    assert verdict.drift == ([changed_path] if label == "base-drift" else [])
    assert all(line.endswith(f" {label}") for line in verdict.lines[:3])
    assert changed_path in verdict.lines[-1]
    args = [
        "verify-landed",
        "--candidates",
        str(_candidates_json(tmp_path, members)),
        "--canonical",
        str(repo),
        "--remote",
        str(repo),
        "--scratch",
        str(tmp_path / "scratch.git"),
        "--proved-tree",
        built.tree,
    ]
    for pr, sha in landed:
        args.extend(["--landed", f"{pr}={sha}"])
    assert group.main(args) == exit_code
    captured = capsys.readouterr()
    assert label in captured.out and changed_path in captured.out
    assert captured.err == ""


@pytest.mark.unit
def test_main_plan_prints_groups_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    members = [_candidate(n, f"src/file_{n}.py") for n in (3, 1, 2)]
    assert (
        group.main(["plan", "--candidates", str(_candidates_json(tmp_path, members))])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "groups": [[{"pr": c.number, "head": c.head} for c in members]],
        "solo": [],
    }
    assert captured.err == ""


@pytest.mark.unit
def test_main_build_and_plan_support_omnimarket(
    git_repo: tuple[Path, str, list[Any], Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _base, infra_members, _conflict = git_repo
    git = group.make_git(repo)
    members = [
        group.Candidate(c.number, c.head, c.files, repo="omnimarket")
        for c in infra_members[:2]
    ]
    for member in members:
        git("update-ref", f"refs/pull/{member.number}/head", member.head)
    candidates = _candidates_json(tmp_path, members)
    params = tmp_path / "market.env"
    assert (
        group.main(
            [
                "build",
                "--repo",
                "omnimarket",
                "--candidates",
                str(candidates),
                "--canonical",
                str(repo),
                "--remote",
                str(repo),
                "--scratch",
                str(tmp_path / "market-scratch.git"),
                "--params-out",
                str(params),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "group-step omnimarket#1" in captured.out
    assert "MARKET_GROUP=" in params.read_text(encoding="utf-8")
    assert (
        group.main(["plan", "--repo", "omnimarket", "--candidates", str(candidates)])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["groups"]


@pytest.mark.unit
def test_verify_landed_names_omnimarket(
    git_repo: tuple[Path, str, list[Any], Any],
) -> None:
    repo, base, infra_members, _conflict = git_repo
    members = [
        group.Candidate(c.number, c.head, c.files, repo="omnimarket")
        for c in infra_members
    ]
    git = group.make_git(repo)
    built = group.build_group(git, base, members)
    landed = _squash_members(git, members)
    verdict = group.verify_landed(
        git,
        built.tree,
        landed,
        {c.number: c.files for c in members},
        repo="omnimarket",
    )
    assert verdict.lines[0].startswith("omnimarket#1 landed")


@pytest.mark.unit
def test_main_halves_rejects_single_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _candidates_json(tmp_path, [_candidate(1, "src/one.py")])
    assert group.main(["halves", "--candidates", str(path)]) == 5
    captured = capsys.readouterr()
    assert "single PR" in captured.err
    assert captured.out == ""
