# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-trigger behaviour of the unattended worktree prune (OMN-18832).

The classifier is NOT under test here and is not reimplemented: this runner
drives `omniclaude/scripts/worktree_auto_prune.py` as a subprocess. What IS
under test is the four things that script does not do — the removal-debris
path OMN-18826 found, the revocable standing consent, the ledger row, and the
handoff of refusals to a local model.

`git`, the classifier and the delegate wrapper are shimmed onto PATH rather
than monkeypatched, so each crosses the same subprocess boundary a real run
crosses, and a shim that FAILS is distinguishable from one that returns
nothing.

`test_control_*` are the positive controls: without them a runner that refused
every single thing would satisfy every refusal assertion in this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "scripts" / "worktree_prune_timer.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from worktree_prune_timer import (
    DELEGATE_PROMPT_CEILING,
    ModelTimerConfig,
    RefusedError,
    batch_refusals,
    classify_removal_debris,
    load_timer_config,
    removal_budget_seconds,
)

CONSENT_ROW = (
    "2026-09-19T12:55:31Z | OPERATOR-CONSENT | lane=fixture | "
    '"the clones should not have any stale branch wraps at all" | '
    "APPROVED SCOPE: removal of worktrees whose pull request is merged | "
    "OUT OF SCOPE: dirty trees, unpushed commits | "
    "This row is the durable authorization evidence"
)


# ---------------------------------------------------------------------------
# The removal-debris predicate (OMN-18826)
# ---------------------------------------------------------------------------
#
# A `git worktree remove` without --force is NOT atomic on git 2.50.1. One
# unwritable subdirectory makes it exit 255, delete the admin directory, and
# leave survivors behind a stale `.git` file. The SECOND removal over that
# half-deleted tree then refuses with "contains modified or untracked files",
# which reads as a peer lane's work when it is in fact debris from the first
# attempt.
#
# The signature that tells the two apart is the porcelain: debris is unstaged
# DELETIONS and nothing else. An untracked file is content that exists nowhere
# else; a modified file is an edit; a STAGED deletion is somebody having run
# `git rm` deliberately. Each of those is real work and none of them is debris.


@pytest.mark.parametrize(
    ("porcelain", "eligible", "why"),
    [
        (" D a.py\n D b/c.py\n", True, "unstaged deletions only — the debris shape"),
        (" D a.py\n", True, "one deletion is still deletions-only"),
        (" D a.py\n?? scratch.txt\n", False, "an untracked file is unique content"),
        (" D a.py\n M b.py\n", False, "a modified file is an edit, not debris"),
        (" D a.py\nD  b.py\n", False, "a STAGED deletion is a deliberate git rm"),
        (" D a.py\nA  new.py\n", False, "a staged add is work"),
        ("?? scratch.txt\n", False, "no deletions at all"),
        ("", False, "a clean tree is not debris and needs no force"),
        ("   \n\n", False, "whitespace is not a deletion"),
        (
            " D a.py\n!! ignored\n",
            False,
            "an unparsed status code is never assumed benign",
        ),
    ],
)
def test_the_debris_predicate_admits_only_the_deletions_only_shape(
    porcelain: str, eligible: bool, why: str
) -> None:
    verdict = classify_removal_debris(porcelain, pr_merged=True)
    assert verdict.eligible is eligible, why
    assert verdict.reason, "every verdict names its clause"


def test_debris_is_never_forced_on_an_unmerged_branch() -> None:
    """The porcelain says the tree is debris. The pull request is what says
    the CONTENT survives the removal, and without it a forced removal destroys
    whatever those deleted paths were."""
    verdict = classify_removal_debris(" D a.py\n D b.py\n", pr_merged=False)

    assert verdict.eligible is False
    assert "merged" in verdict.reason


def test_control_the_debris_predicate_can_say_yes() -> None:
    """Without this, a predicate hardcoded to False passes every case above."""
    assert classify_removal_debris(" D a.py\n", pr_merged=True).eligible is True


# ---------------------------------------------------------------------------
# The removal budget
# ---------------------------------------------------------------------------


def test_the_budget_scales_with_file_count_rather_than_a_fixed_constant() -> None:
    """A 17,600-file tree under load 40 needed 280s where the fixed 90s budget
    this replaces would have called it a timeout (OMN-18826)."""
    small = removal_budget_seconds(10)
    medium = removal_budget_seconds(5_000)
    large = removal_budget_seconds(17_600)

    assert small < medium < large
    assert large >= 280, "the measured 17,600-file case must fit inside the budget"


def test_the_budget_is_monotonic_and_bounded() -> None:
    counts = [0, 1, 100, 1_000, 10_000, 100_000, 10_000_000]
    budgets = [removal_budget_seconds(n) for n in counts]

    assert budgets == sorted(budgets), "monotonic in file count"
    assert budgets[0] > 0, "even an empty tree gets a real budget"
    assert budgets[-1] <= 3_600, "bounded, so a pathological count cannot hang a timer"


# ---------------------------------------------------------------------------
# Batching the refusal list for a local model
# ---------------------------------------------------------------------------
#
# Operator ruling 2026-09-19 ~13:36Z, verbatim: "if we need an LLM in the loop
# for triage it should go to local models". Measured on this backend the same
# day: a 5,497-char prompt hit the 240s handler budget and was cancelled; a
# 2,509-char prompt carrying the same facts completed in 78s.


def test_every_batch_fits_under_the_prompt_ceiling() -> None:
    items = [
        f"/path/to/worktree-{i} | dirty tree | 3 modified files" for i in range(400)
    ]

    batches = batch_refusals(items, DELEGATE_PROMPT_CEILING)

    assert batches, "400 items produce at least one batch"
    for batch in batches:
        assert len("\n".join(batch)) <= DELEGATE_PROMPT_CEILING
    assert [i for b in batches for i in b] == items, "no item is dropped or reordered"


def test_an_item_longer_than_the_ceiling_is_carried_alone_not_dropped() -> None:
    """Dropping it would silently shrink the refusal list, which is the one
    thing a triage handoff must never do."""
    giant = "x" * (DELEGATE_PROMPT_CEILING * 2)
    batches = batch_refusals(["short", giant, "also short"], DELEGATE_PROMPT_CEILING)

    flat = [i for b in batches for i in b]
    assert flat == ["short", giant, "also short"]
    assert [giant] in [list(b) for b in batches], "the oversize item rides alone"


def test_an_empty_refusal_list_produces_no_batches() -> None:
    assert batch_refusals([], DELEGATE_PROMPT_CEILING) == ()


# ---------------------------------------------------------------------------
# The revocable standing consent
# ---------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path: Path) -> dict[str, object]:
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(
        "2026-09-19T12:00:00Z | NOTE | lane=fixture | not a consent row\n"
        + CONSENT_ROW
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "worktree_prune_timer.yaml"
    config.write_text(
        "enabled: true\n"
        f'consent_citation: "{ledger}:2"\n'
        "delegate_refusal_review: true\n",
        encoding="utf-8",
    )
    return {"tmp": tmp_path, "ledger": ledger, "config": config}


def test_control_a_well_formed_config_loads(world: dict[str, object]) -> None:
    """The control for every refusal below."""
    config = load_timer_config(Path(str(world["config"])))

    assert isinstance(config, ModelTimerConfig)
    assert config.enabled is True
    assert config.consent_citation.endswith(":2")


def test_an_absent_config_refuses_rather_than_defaulting_to_enabled(
    tmp_path: Path,
) -> None:
    """A timer that runs when its authorisation file is missing is a timer
    whose authorisation cannot be revoked by deleting the file."""
    with pytest.raises(RefusedError, match="config"):
        load_timer_config(tmp_path / "nope.yaml")


def test_the_operator_can_revoke_by_flipping_one_flag(world: dict[str, object]) -> None:
    path = Path(str(world["config"]))
    path.write_text(
        path.read_text(encoding="utf-8").replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )

    config = load_timer_config(path)

    assert config.enabled is False


def test_a_config_with_no_consent_citation_refuses(world: dict[str, object]) -> None:
    path = Path(str(world["config"]))
    path.write_text("enabled: true\n", encoding="utf-8")

    with pytest.raises(RefusedError, match="consent_citation"):
        load_timer_config(path)


def test_the_citation_is_never_hardcoded_in_the_source() -> None:
    """The ledger line number the deletions are authorised by must come from
    the config. A source-literal citation cannot be revoked."""
    source = TOOL.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "ROLLING_WORK_LEDGER.md:2717" not in code
    assert ":2717" not in code


# ---------------------------------------------------------------------------
# End-to-end refusals: nothing is removed when the run cannot be authorised
# ---------------------------------------------------------------------------


def _shims(bin_dir: Path, *, delegate_exit: int = 0, delegate_out: str = "") -> Path:
    """`git`, the classifier and the delegate wrapper, all recording calls."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    calls = bin_dir / "calls.log"

    git = bin_dir / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "git $*" >> "{calls}"\n'
        'if [[ "$*" == *"worktree list"* ]]; then exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    git.chmod(0o755)

    delegate = bin_dir / "onex"
    delegate.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "onex $*" >> "{calls}"\n'
        f"if [[ {delegate_exit} -ne 0 ]]; then\n"
        '  echo "delegate: simulated refusal" >&2\n'
        f"  exit {delegate_exit}\n"
        "fi\n"
        f"printf '%s' '{delegate_out}'\n",
        encoding="utf-8",
    )
    delegate.chmod(0o755)
    return calls


def _run_timer(
    world: dict[str, object], *args: str
) -> subprocess.CompletedProcess[str]:
    tmp = Path(str(world["tmp"]))
    env = dict(os.environ)
    env["OMNI_HOME"] = str(tmp)
    env["PATH"] = f"{tmp / 'bin'}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--config",
            str(world["config"]),
            "--ledger",
            str(world["ledger"]),
            "--state-dir",
            str(tmp / "state"),
            "--json",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_an_unreadable_ledger_refuses_before_anything_is_removed(
    world: dict[str, object],
) -> None:
    """Rule: the run refuses if the ledger is unreadable. The ledger is both
    the authorisation and the only durable record of what was destroyed, so a
    run that cannot write one must not destroy anything."""
    calls = _shims(Path(str(world["tmp"])) / "bin")
    missing = Path(str(world["tmp"])) / "gone" / "LEDGER.md"

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--config",
            str(world["config"]),
            "--ledger",
            str(missing),
            "--state-dir",
            str(Path(str(world["tmp"])) / "state"),
            "--execute",
        ],
        env={
            **os.environ,
            "OMNI_HOME": str(world["tmp"]),
            "PATH": f"{Path(str(world['tmp'])) / 'bin'}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3, result.stdout
    assert "REFUSED" in result.stderr
    assert "worktree remove" not in (
        calls.read_text(encoding="utf-8") if calls.exists() else ""
    )


def test_a_disabled_config_refuses_and_removes_nothing(
    world: dict[str, object],
) -> None:
    calls = _shims(Path(str(world["tmp"])) / "bin")
    path = Path(str(world["config"]))
    path.write_text(
        path.read_text(encoding="utf-8").replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )

    result = _run_timer(world, "--execute")

    assert result.returncode == 3
    assert "disabled" in result.stderr
    assert "worktree remove" not in (
        calls.read_text(encoding="utf-8") if calls.exists() else ""
    )


def test_a_citation_to_a_non_consent_row_refuses(world: dict[str, object]) -> None:
    """Line 1 of the fixture ledger is deliberately a NOTE row."""
    _shims(Path(str(world["tmp"])) / "bin")
    path = Path(str(world["config"]))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"{world["ledger"]}:2"', f'"{world["ledger"]}:1"'
        ),
        encoding="utf-8",
    )

    result = _run_timer(world, "--execute")

    assert result.returncode == 3
    assert "OPERATOR-CONSENT" in result.stderr


def test_control_an_authorised_dry_run_reaches_the_report(
    world: dict[str, object],
) -> None:
    """The positive control for the three refusals above: with the same config
    resolving, the run proceeds and emits its report."""
    _shims(Path(str(world["tmp"])) / "bin")

    result = _run_timer(world)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["executed"] is False
    assert report["consent_citation"].endswith(":2")


def test_a_dry_run_writes_no_ledger_row(world: dict[str, object]) -> None:
    _shims(Path(str(world["tmp"])) / "bin")
    before = Path(str(world["ledger"])).read_text(encoding="utf-8")

    _run_timer(world)

    assert Path(str(world["ledger"])).read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# The forced removal itself: retry count, and stderr that is never discarded
# ---------------------------------------------------------------------------


def _fake_worktree(tmp_path: Path, porcelain: str) -> tuple[Path, Path]:
    """A directory plus a `git` shim answering status with `porcelain`."""
    worktree = tmp_path / "wt"
    worktree.mkdir(parents=True, exist_ok=True)
    canonical = tmp_path / "clone"
    canonical.mkdir(parents=True, exist_ok=True)
    return worktree, canonical


def test_a_forced_removal_records_the_exit_code_and_the_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guessed cause over a discarded stderr is what made OMN-18826 debris
    read as a peer lane's work for an hour."""
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(args))
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n D b.py\n", "")
        return subprocess.CompletedProcess(args, 128, "", "fatal: cannot unlink")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", lambda wt: "/snapshots/wt")
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    assert timer.force_remove_debris(worktree, canonical, "br", report) is False
    failure = report.removal_failures[0]
    assert failure["exit_code"] == 128
    assert "cannot unlink" in str(failure["stderr"])


def test_exactly_one_retry_is_taken_never_a_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")
    removes: list[list[str]] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n", "")
        removes.append(list(args))
        return subprocess.CompletedProcess(args, 1, "", "refused")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", lambda wt: "/snapshots/wt")
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    timer.force_remove_debris(worktree, canonical, "br", report)

    assert len(removes) == 2, "one attempt plus exactly one retry"
    assert all("--force" in r for r in removes)


def test_an_unmerged_branch_is_never_forced_even_when_the_shape_is_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")
    removes: list[list[str]] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n D b.py\n", "")
        removes.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: False)
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    assert timer.force_remove_debris(worktree, canonical, "br", report) is False
    assert removes == [], "no removal command was issued at all"
    assert report.debris_forced == 0


def test_control_a_proven_debris_tree_is_actually_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, a function hardcoded to False passes every case above."""
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n D b.py\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", lambda wt: "/snapshots/wt")
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    assert timer.force_remove_debris(worktree, canonical, "br", report) is True
    assert report.debris_forced == 1


def test_the_budget_passed_to_git_scales_with_the_file_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, "")
    porcelain = "".join(f" D f{i}.py\n" for i in range(5_000))
    seen: list[int | None] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, porcelain, "")
        seen.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", lambda wt: "/snapshots/wt")
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    timer.force_remove_debris(worktree, canonical, "br", report)

    assert seen == [timer.removal_budget_seconds(5_000)]
    assert seen[0] is not None and seen[0] > timer.BUDGET_FLOOR_SECONDS


# ---------------------------------------------------------------------------
# OMN-19539: nothing is force-removed unsaved
# ---------------------------------------------------------------------------


def test_a_failed_save_issues_no_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")
    removes: list[list[str]] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n D b.py\n", "")
        removes.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    def refuse(wt: Path) -> str:
        raise timer.RefusedError("pre-removal snapshot helper missing")

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", refuse)
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    assert timer.force_remove_debris(worktree, canonical, "br", report) is False
    assert removes == [], "no removal command was issued at all"
    assert "snapshot" in str(report.removal_failures[0]["detail"])


def test_control_the_save_comes_before_the_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worktree_prune_timer as timer

    worktree, canonical = _fake_worktree(tmp_path, " D a.py\n")
    order: list[str] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if "status" in args:
            return subprocess.CompletedProcess(args, 0, " D a.py\n D b.py\n", "")
        order.append("remove")
        return subprocess.CompletedProcess(args, 0, "", "")

    def save(wt: Path) -> str:
        order.append("save")
        return "/snapshots/wt"

    monkeypatch.setattr(timer, "_run", fake_run)
    monkeypatch.setattr(timer, "_branch_pr_is_merged", lambda *a, **k: True)
    monkeypatch.setattr(timer, "save_before_removal", save)
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    assert timer.force_remove_debris(worktree, canonical, "br", report) is True
    assert order == ["save", "remove"]
    assert "saved first to /snapshots/wt" in report.notes[-1]


_FAKE_HELPER = """import json, os, sys
if os.environ.get("FAKE_SNAPSHOT_FAIL"):
    print(json.dumps({"ok": False})); sys.exit(3)
d = os.path.join(os.environ["OMNI_HOME"], ".onex_state", "worktree-removal-snapshots", "x")
os.makedirs(d)
open(os.path.join(d, "argv.json"), "w").write(json.dumps(sys.argv[1:]))
print(json.dumps({"ok": True, "directory": d}))
"""


def test_save_before_removal_runs_the_shared_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper's own behaviour is tested in omniclaude; this pins the call."""
    import worktree_prune_timer as timer

    registry = tmp_path / "registry"
    helper = registry / timer.SNAPSHOT_HELPER_REL
    helper.parent.mkdir(parents=True)
    helper.write_text(_FAKE_HELPER, encoding="utf-8")
    monkeypatch.setenv("OMNI_HOME", str(registry))
    monkeypatch.delenv("FAKE_SNAPSHOT_FAIL", raising=False)

    directory = Path(timer.save_before_removal(tmp_path / "wt"))

    argv = json.loads((directory / "argv.json").read_text(encoding="utf-8"))
    assert argv[0] == str(tmp_path / "wt")
    assert "--allow-non-git" in argv

    monkeypatch.setenv("FAKE_SNAPSHOT_FAIL", "1")
    with pytest.raises(timer.RefusedError, match="exit 3"):
        timer.save_before_removal(tmp_path / "wt")

    helper.unlink()
    with pytest.raises(timer.RefusedError, match="helper missing"):
        timer.save_before_removal(tmp_path / "wt")

    monkeypatch.delenv("OMNI_HOME")
    with pytest.raises(timer.RefusedError, match="OMNI_HOME"):
        timer.save_before_removal(tmp_path / "wt")


# ---------------------------------------------------------------------------
# The local-model handoff never acts, and records a refusal verbatim
# ---------------------------------------------------------------------------


def test_a_refusing_delegate_leaves_the_items_unclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator ruling 2026-09-19: triage LLM work goes to local models. An
    unanswered triage question is not a triaged item."""
    import worktree_prune_timer as timer

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args, 7, "", "delegate: backend down")

    monkeypatch.setattr(timer, "_run", fake_run)
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    timer.delegate_refusal_review(["/a | refused | dirty"], tmp_path, report)

    assert report.delegate_proposals == [], (
        "nothing was proposed, so nothing is acted on"
    )
    receipts = report.delegate_receipt["receipts"]  # type: ignore[index]
    assert receipts[0]["ok"] is False
    assert "backend down" in str(receipts[0]["stderr"])
    assert any("unclassified" in n for n in report.notes)


def test_the_delegate_is_the_wrapper_on_the_deployed_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a Claude workflow, and not the plugin CLI: the sanctioned wrapper
    at omnibase_infra/scripts/onex, on the deployed lane over the bus."""
    import worktree_prune_timer as timer

    seen: list[list[str]] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(list(args))
        return subprocess.CompletedProcess(args, 0, "/a | remove | merged\n", "")

    monkeypatch.setattr(timer, "_run", fake_run)
    report = timer.ModelRunReport(started_at="t", executed=True, consent_citation="c")

    timer.delegate_refusal_review(["/a | refused | dirty"], tmp_path, report)

    argv = seen[0]
    assert argv[:3] == [
        "bash",
        str(tmp_path / "omnibase_infra" / "scripts" / "onex"),
        "delegate",
    ]
    assert "--locus" in argv and argv[argv.index("--locus") + 1] == "deployed-lane"
    assert "--bus" in argv and argv[argv.index("--bus") + 1] == "kafka"
    assert report.delegate_proposals == ["/a | remove | merged"]


# ---------------------------------------------------------------------------
# The launchd agent — the half that was missing entirely
# ---------------------------------------------------------------------------

PLIST = REPO_ROOT / "scripts" / "launchd" / "com.omninode.worktree-prune.plist.template"
INSTALLER = REPO_ROOT / "scripts" / "launchd" / "install-worktree-prune.sh"
TIMER_CONFIG = REPO_ROOT / "config" / "worktree_prune_timer.yaml"


def test_the_agent_names_the_literal_brew_interpreter_not_a_path_lookup() -> None:
    """launchd runs with a restricted PATH and no login shell, so `python3`
    and a brew-prefix expansion both fail inside an agent (Operating Rule 11).
    The installer resolves the literal path; the template must not smuggle in
    a lookup."""
    installer = INSTALLER.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in installer.splitlines() if not line.lstrip().startswith("#")
    )

    assert "/opt/homebrew/bin/python3.13" in code
    assert "/usr/local/bin/python3.13" in code, "the Intel path too"
    # The comments explain why a brew-prefix expansion cannot be used here, so
    # the prohibition is asserted against the CODE, not against the prose.
    assert "brew --prefix" not in code, "launchd cannot expand it"

    template = PLIST.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in template.splitlines() if not line.strip().startswith("<!--")
    )
    assert "@BREW_PYTHON@" in body, "the interpreter is substituted, not guessed"
    assert "<string>python3</string>" not in body


def test_the_template_carries_no_machine_absolute_path() -> None:
    """Operating Rule 6: a committed `/Users/` or `/Volumes/` literal is a bug.
    The plist is a template precisely because launchd needs a literal that the
    repository may not carry."""
    lines = [
        line
        for line in PLIST.read_text(encoding="utf-8").splitlines()
        if "<string>" in line
    ]
    offending = [line for line in lines if "/Users/" in line or "/Volumes/" in line]

    assert offending == [], f"machine paths in the template: {offending}"


def test_the_agent_is_daily_and_neither_runs_at_load_nor_keeps_alive() -> None:
    """Installing the agent must not start a removal pass as a side effect,
    and a batch job that is restarted on exit loops its removals."""
    template = PLIST.read_text(encoding="utf-8")

    assert "StartCalendarInterval" in template
    assert "<key>RunAtLoad</key>\n    <false/>" in template
    assert "<key>KeepAlive</key>\n    <false/>" in template


def test_the_agent_passes_execute_and_a_state_dir() -> None:
    """A timer that only ever dry-runs is the same non-automation this closes."""
    template = PLIST.read_text(encoding="utf-8")

    assert "<string>--execute</string>" in template
    assert ".onex_state/worktree-prune" in template


def test_the_shipped_config_resolves_and_is_revocable() -> None:
    """The config committed to the repo must itself parse: an unreadable one
    refuses every run, silently, from a timer nobody is watching."""
    config = load_timer_config(TIMER_CONFIG)

    assert config.enabled is True
    assert config.consent_citation.startswith("docs/tracking/ROLLING_WORK_LEDGER.md:")
    assert "enabled" in TIMER_CONFIG.read_text(encoding="utf-8")
