# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One runtime-affecting predicate for the rebuild trigger and the proof-subject walk.

RED STATE THIS REPRODUCES (OMN-19318, plan row D15)
---------------------------------------------------
``should_trigger`` in ``scripts/trigger_rebuild_on_merge.py`` fires on the
``runtime_change`` label OR a runtime path. ``load_runtime_affecting`` in
``scripts/ci/release_train.py`` -- the predicate ``resolve_lab_candidate`` walks
with -- reads changed PATHS only. So a merge that is runtime-affecting only by
its label is rebuilt and verified by the trigger, while the walk steps past it
to an older subject and accepts that older subject's receipt as proof for code
the receipt never ran.

THE FIX
-------
One function, ``is_runtime_affecting`` in ``scripts/runtime_change_classifier.py``:
the path rule unioned with the merged pull request's ``runtime_change`` label.
The trigger's ``should_trigger`` IS that function, and the train's per-commit
predicate calls it. A label read that fails raises ``LabelReadError``; the walk
then resolves nothing and the caller falls back to the head (fail closed).

Every green assertion here has a flipped sibling: the label-only merge stops the
walk, and the same merge without the label does not.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    artifact_name,
    build_receipt,
    evaluate_gate,
    resolve_required_subject,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER = _REPO_ROOT / "scripts" / "runtime_change_classifier.py"
_TRIGGER = _REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
_TRAIN = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_GATE = _REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py"

#: The ONE name the classifier module is registered under, whoever loads it
#: first. Two names would be two module objects, and so two predicates.
_CANONICAL_CLASSIFIER_MODULE = "_omnibase_infra_runtime_change_classifier"

REPO = "OmniNode-ai/omnibase_infra"


def _load(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load(_TRAIN, "release_train_under_test_omn19318")
trigger = _load(_TRIGGER, "trigger_under_test_omn19318")


def _classifier() -> Any:
    return sys.modules[_CANONICAL_CLASSIFIER_MODULE]


# --------------------------------------------------------------------------- #
# A real first-parent history: R (runtime path), then L (docs only), then N.    #
# --------------------------------------------------------------------------- #
class _History:
    def __init__(self, root: Path, validator: Path, shas: dict[str, str]) -> None:
        self.root = root
        self.validator = validator
        self.shas = shas

    def commits(self, head: str) -> list[str]:
        return rt.default_branch_commits(self.root, self.shas[head])


def _history(tmp_path: Path) -> _History:
    from omnibase_core.validators.no_unguarded_git_subprocess import (
        scrub_git_location_env,
    )

    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()

    git("init", "--initial-branch", "dev")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    shas: dict[str, str] = {}
    for key, path in (
        ("base", "README.md"),
        ("R", "src/omnibase_infra/runtime_thing.py"),
        ("L", "docs/label_only.md"),
        ("N", "docs/neither.md"),
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{key}\n", encoding="utf-8")
        git("add", path)
        git("commit", "-m", key)
        shas[key] = git("rev-parse", "HEAD")

    validator = tmp_path / "validate_pr_deploy_required.py"
    validator.write_text(
        "def find_runtime_paths(changed_files, *a, **k):\n"
        "    return [p for p in changed_files if p.startswith('src/')]\n",
        encoding="utf-8",
    )
    return _History(root, validator, shas)


def _labels(mapping: dict[str, Sequence[str]]) -> Any:
    def read(sha: str) -> Sequence[str]:
        return mapping.get(sha, [])

    return read


def _predicate(history: _History, labels_for: Any) -> Any:
    return rt.load_runtime_affecting(
        history.root, history.validator, labels_for=labels_for
    )


# --------------------------------------------------------------------------- #
# AC1 -- a label-only merge is its own proof subject.                           #
# --------------------------------------------------------------------------- #
class TestLabelOnlyMergeIsItsOwnSubject:
    def test_label_only_resolver_returns_the_label_only_merge(
        self, tmp_path: Path
    ) -> None:
        """The falsifier: R holds the PASS, L above it carries only the label."""
        h = _history(tmp_path)
        L, R = h.shas["L"], h.shas["R"]
        candidate = rt.resolve_lab_candidate(
            L,
            branch_commits=lambda: h.commits("L"),
            runtime_affecting=_predicate(h, _labels({L: ["runtime_change"]})),
        )
        assert candidate.sha == L, (
            f"the walk stepped past the label-only merge {L[:12]} to {candidate.sha[:12]}"
        )
        assert candidate.sha != R
        assert candidate.skipped == ()

    def test_label_only_older_receipt_is_refused_naming_the_label_only_merge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R's compose-dev PASS is not proof for L; the refusal names L."""
        h = _history(tmp_path)
        L, R = h.shas["L"], h.shas["R"]
        subject, note = resolve_required_subject(
            L,
            branch_commits=lambda: h.commits("L"),
            runtime_affecting=_predicate(h, _labels({L: ["runtime_change"]})),
        )
        assert subject == L
        assert note == f"{L} itself (runtime-affecting)"
        code, output = _gate_on(monkeypatch, head=L, subject=subject, pass_for=R)
        assert code == 1
        assert f"REQUIRED lane compose-dev does not pass for sha {L}" in output

    def test_label_only_positive_control_without_the_label_inherits_r(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flipped sibling: the same history with no label inherits R, and
        R's PASS passes. That is what makes the refusal above the label's doing."""
        h = _history(tmp_path)
        L, R = h.shas["L"], h.shas["R"]
        subject, _ = resolve_required_subject(
            L,
            branch_commits=lambda: h.commits("L"),
            runtime_affecting=_predicate(h, _labels({})),
        )
        assert subject == R
        code, _ = _gate_on(monkeypatch, head=L, subject=subject, pass_for=R)
        assert code == 0


# --------------------------------------------------------------------------- #
# AC2 -- neither signal still resolves to R.                                    #
# --------------------------------------------------------------------------- #
class TestNeitherSignalInheritsR:
    def test_neither_signal_resolves_to_r(self, tmp_path: Path) -> None:
        h = _history(tmp_path)
        N, L, R = h.shas["N"], h.shas["L"], h.shas["R"]
        candidate = rt.resolve_lab_candidate(
            N,
            branch_commits=lambda: h.commits("N"),
            runtime_affecting=_predicate(h, _labels({N: ["bug", "docs"]})),
        )
        assert candidate.sha == R
        assert candidate.skipped == (N, L)

    def test_neither_signal_but_a_labelled_merge_between_stops_there(
        self, tmp_path: Path
    ) -> None:
        h = _history(tmp_path)
        N, L = h.shas["N"], h.shas["L"]
        candidate = rt.resolve_lab_candidate(
            N,
            branch_commits=lambda: h.commits("N"),
            runtime_affecting=_predicate(h, _labels({L: ["runtime_change"]})),
        )
        assert candidate.sha == L
        assert candidate.skipped == (N,)


# --------------------------------------------------------------------------- #
# AC3 -- a label read that fails resolves nothing (fail closed).                #
# --------------------------------------------------------------------------- #
class TestLabelReadFailureFailsClosed:
    def test_label_read_error_the_shared_function_raises_an_explicit_token(
        self,
    ) -> None:
        def broken() -> Sequence[str]:
            raise RuntimeError("gh api exited 1: HTTP 502")

        with pytest.raises(_classifier().LabelReadError, match="HTTP 502"):
            _classifier().is_runtime_affecting([], broken)

    def test_label_read_error_non_list_is_refused_not_read_as_no_label(
        self,
    ) -> None:
        with pytest.raises(_classifier().LabelReadError):
            _classifier().is_runtime_affecting([], lambda: "runtime_change")

    def test_label_read_error_walk_resolves_nothing(self, tmp_path: Path) -> None:
        h = _history(tmp_path)
        N = h.shas["N"]

        def broken(sha: str) -> Sequence[str]:
            raise RuntimeError("gh api exited 1: HTTP 502")

        candidate = rt.resolve_lab_candidate(
            N,
            branch_commits=lambda: h.commits("N"),
            runtime_affecting=_predicate(h, broken),
        )
        assert candidate.sha == ""
        assert "label" in candidate.unresolved_reason

    def test_label_read_error_caller_falls_back_to_the_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's fallback path, exercised: the head is asked for its own
        receipt, and R's PASS does not stand in for it."""
        h = _history(tmp_path)
        N, R = h.shas["N"], h.shas["R"]

        def broken(sha: str) -> Sequence[str]:
            raise RuntimeError("gh api exited 1: HTTP 502")

        subject, note = resolve_required_subject(
            N,
            branch_commits=lambda: h.commits("N"),
            runtime_affecting=_predicate(h, broken),
        )
        assert subject == N
        assert "label" in note
        code, output = _gate_on(monkeypatch, head=N, subject=subject, pass_for=R)
        assert code == 1
        assert f"does not pass for sha {N}" in output

    def test_label_read_error_is_not_consulted_when_a_path_already_decides(
        self,
    ) -> None:
        """A runtime path stops the walk on its own, so an unreadable label on
        that commit changes nothing (the walk already stops there)."""

        def broken() -> Sequence[str]:
            raise RuntimeError("never read")

        assert _classifier().is_runtime_affecting(["src/x.py"], broken) is True


# --------------------------------------------------------------------------- #
# AC4 -- both call sites use the same function object.                          #
# --------------------------------------------------------------------------- #
class TestOnePredicateObject:
    def test_shared_predicate_should_trigger_is_the_classifier_function(
        self,
    ) -> None:
        assert trigger.should_trigger is _classifier().is_runtime_affecting

    def test_shared_predicate_the_train_calls_the_same_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spy on the ONE classifier module is what the train's predicate hits,
        so the train and the trigger cannot be reading two module copies."""
        h = _history(tmp_path)
        calls: list[tuple[list[str], Any]] = []
        real = _classifier().is_runtime_affecting

        def spy(runtime_paths: Sequence[str], labels: Any) -> bool:
            calls.append((list(runtime_paths), labels))
            return bool(real(runtime_paths, labels))

        monkeypatch.setattr(_classifier(), "is_runtime_affecting", spy)
        predicate = _predicate(h, _labels({h.shas["L"]: ["runtime_change"]}))
        assert predicate(h.shas["L"]) is True
        assert predicate(h.shas["N"]) is False
        assert len(calls) == 2

    @pytest.mark.parametrize("path", [_TRIGGER, _TRAIN, _GATE], ids=lambda p: p.name)
    def test_shared_predicate_no_call_site_defines_its_own(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.name not in {"should_trigger", "is_runtime_affecting"}, (
                    f"{path.name} defines {node.name}; the runtime-affecting "
                    "predicate has one definition, in runtime_change_classifier.py"
                )
            if isinstance(node, ast.Constant):
                assert node.value != "runtime_change", (
                    f"{path.name} spells the runtime_change label itself; it must "
                    "read RUNTIME_CHANGE_LABEL from the shared classifier"
                )

    def test_shared_predicate_classifier_registered_under_one_name(self) -> None:
        for path in (_TRIGGER, _TRAIN):
            assert _CANONICAL_CLASSIFIER_MODULE in path.read_text(encoding="utf-8"), (
                f"{path.name} must load the classifier under "
                f"{_CANONICAL_CLASSIFIER_MODULE}, or it holds a second module "
                "object and so a second predicate"
            )


# --------------------------------------------------------------------------- #
# The default label reader: the merged pull request's labels, read with gh.     #
# --------------------------------------------------------------------------- #
class TestDefaultMergedPrLabels:
    def test_merged_pr_labels_reads_the_merged_pull_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            seen.append(cmd)
            body = (
                '[{"number": 7, "merged_at": "2026-09-23T00:00:00Z", '
                '"labels": [{"name": "runtime_change"}, {"name": "bug"}]},'
                ' {"number": 8, "merged_at": null, '
                '"labels": [{"name": "stray"}]}]'
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=body, stderr="")

        monkeypatch.setattr(rt.subprocess, "run", fake_run)
        labels = rt.default_merged_pr_labels(REPO, "a" * 40)
        assert labels == ["runtime_change", "bug"]
        assert f"repos/{REPO}/commits/{'a' * 40}/pulls" in seen[0]

    def test_merged_pr_labels_a_non_list_response_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"message": "Not Found"}', stderr=""
            )

        monkeypatch.setattr(rt.subprocess, "run", fake_run)
        with pytest.raises(ValueError, match="not a list"):
            rt.default_merged_pr_labels(REPO, "a" * 40)


def _gate_on(
    monkeypatch: pytest.MonkeyPatch, *, head: str, subject: str, pass_for: str
) -> tuple[int, str]:
    """Run the delivery gate for ``head`` with compose-dev required of ``subject``,
    over a surface holding exactly one compose-dev PASS, for ``pass_for``."""
    from datetime import UTC, datetime

    from scripts.ci.lab_pass_receipt import ModelLabPassCheck

    receipt = build_receipt(
        sha=pass_for,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 9, 23, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 23, 12, 20, tzinfo=UTC),
        checks=[ModelLabPassCheck(name="ready_main", ok=True, evidence="200")],
        agent_command_id=None,
    )
    bodies = {artifact_name(receipt.lane, receipt.sha): receipt.to_json()}

    import json
    import zipfile

    def surface(path: str) -> bytes:
        if "/actions/artifacts?name=" in path:
            name = path.split("name=")[1].split("&")[0]
            if name not in bodies:
                return json.dumps({"artifacts": []}).encode()
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "id": 1,
                            "name": name,
                            "expired": False,
                            "created_at": "2026-09-23T12:20:00Z",
                        }
                    ]
                }
            ).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("receipt.json", next(iter(bodies.values())))
        return buf.getvalue()

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    out = io.StringIO()
    code = evaluate_gate(
        REPO,
        head,
        [],
        out,
        required=[EnumLabLane.COMPOSE_DEV],
        required_sha=subject,
    )
    return code, out.getvalue()
