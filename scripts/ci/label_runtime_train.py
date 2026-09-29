#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Label an open pull request ``runtime-train`` or ``no-runtime-train`` (OMN-19568, lab-proof plan T4).

Ruling 2026-09-25T13:17:25Z lane=merge-drain-83, operator, verbatim: "Two is
yes". Half (b) of that ruling: a PR that does not touch the deployed runtime
never enters the rebuild train, and it is *classified and labelled* so the
train and the arm script never each read the answer their own way.

Before this module the arm script's own "runtime class" check
(``omniclaude-internal`` ``plugins/omni/skills/arm-green/scripts/arm_green_prs.py``)
answered the question by invoking the SAME classifier this module wraps, as a
subprocess, on every arming pass -- correct but re-computed, unlabelled, and
unreadable by anything else (the lab pool, a human on the PR page, a future
scheduler). This module runs the identical predicate once, pre-merge, and
writes its answer as one of two mutually exclusive GitHub labels the PR
itself carries, so every reader agrees by construction.

THE ONE PREDICATE
------------------
This module adds no path list of its own. It loads
``scripts/runtime_change_classifier.py`` -- the SAME module
``trigger_rebuild_on_merge.py`` (the post-merge rebuild trigger) and
``scripts/ci/release_train.py`` (the release train) load, under the one
``sys.modules`` name the three share (OMN-19318, OMN-18664) -- and calls its
``classify_runtime_paths`` and ``is_runtime_affecting`` exactly as they do.
A merge that the trigger would rebuild for is ``runtime-train`` here; a merge
it would not is ``no-runtime-train``. Disagreement between "will this PR's
merge enter the rebuild train" and "is this PR labelled runtime-train" is a
bug in this module, never a second predicate to reconcile against it.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not read GitHub, build a diff or call ``gh`` unless invoked with the
subcommand that does so (``apply``). ``decide_label`` and ``label_actions``
are pure functions over an already-resolved changed-file list and an
already-resolved label list, exactly like ``is_runtime_affecting`` itself, so
a caller (a workflow step, the lab pool, a test) can supply canned inputs and
does not need a live repository to get a verdict.

Usage
-----
    python3 scripts/ci/label_runtime_train.py decide \\
        --changed-files-file changed.txt \\
        --pr-labels bug,needs-review \\
        --runtime-path-validator /path/to/deploy_gate/validate_pr_deploy_required.py \\
        --source-repo omnimarket

    python3 scripts/ci/label_runtime_train.py apply \\
        --repo OmniNode-ai/omnimarket --pr 2953 \\
        --runtime-path-validator /path/to/deploy_gate/validate_pr_deploy_required.py \\
        --source-repo omnimarket

``apply`` shells out to ``gh pr view``/``gh pr edit``; ``decide`` never shells
out and is what the test suite exercises.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

#: The ONE sys.modules name the classifier is registered under, shared with
#: trigger_rebuild_on_merge.py and scripts/ci/release_train.py (OMN-19318). A
#: second name would load a second module object -- a second copy of the
#: runtime-affecting predicate the callers must never disagree about.
_CLASSIFIER_MODULE_NAME = "_omnibase_infra_runtime_change_classifier"
_CLASSIFIER_PATH = _REPO_ROOT / "scripts" / "runtime_change_classifier.py"

#: The two labels this module ever sets. Mutually exclusive by construction:
#: exactly one is desired for any changed-file list, and label_actions()
#: never returns both in to_add.
LABEL_RUNTIME_TRAIN = "runtime-train"
LABEL_NO_RUNTIME_TRAIN = "no-runtime-train"
MANAGED_LABELS = (LABEL_RUNTIME_TRAIN, LABEL_NO_RUNTIME_TRAIN)


def _load_runtime_change_classifier() -> Any:
    """The shared classifier module, under its one canonical name."""
    existing = sys.modules.get(_CLASSIFIER_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _CLASSIFIER_MODULE_NAME, _CLASSIFIER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load the runtime-change classifier at {_CLASSIFIER_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CLASSIFIER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_CLASSIFIER_MODULE_NAME, None)
        raise
    return module


def decide_label(
    changed_files: Sequence[str],
    pr_labels: Sequence[str],
    classifier: Any,
    *,
    source_repo: str | None = None,
) -> str:
    """``runtime-train`` or ``no-runtime-train`` for one PR's changed files.

    ``classifier`` is the canonical omniclaude deploy-gate ``find_runtime_paths``
    callable (or a hermetic test double of the same shape); it is never
    re-implemented here. ``pr_labels`` is read for the ``runtime_change``
    override the trigger itself honours (``RUNTIME_CHANGE_LABEL``) -- this
    module's own two managed labels are irrelevant to the decision and a
    caller may pass them in without affecting the result.
    """
    rcc = _load_runtime_change_classifier()
    runtime_paths = rcc.classify_runtime_paths(
        list(changed_files),
        classifier,
        source_repo=source_repo,
    )
    affecting = rcc.is_runtime_affecting(runtime_paths, list(pr_labels))
    return LABEL_RUNTIME_TRAIN if affecting else LABEL_NO_RUNTIME_TRAIN


def label_actions(
    desired: str, existing_labels: Sequence[str]
) -> tuple[list[str], list[str]]:
    """``(to_add, to_remove)`` that leave the PR carrying exactly ``desired``.

    Only this module's two managed labels are ever removed; an unrelated
    label on the PR is untouched. Idempotent: a PR that already carries
    ``desired`` and not the other gets an empty pair.
    """
    if desired not in MANAGED_LABELS:
        msg = f"not a managed label: {desired!r}"
        raise ValueError(msg)
    other = (
        LABEL_NO_RUNTIME_TRAIN
        if desired == LABEL_RUNTIME_TRAIN
        else LABEL_RUNTIME_TRAIN
    )
    existing = set(existing_labels)
    to_add = [] if desired in existing else [desired]
    to_remove = [other] if other in existing else []
    return to_add, to_remove


def _load_classifier_from_validator(validator_path: Path) -> Any:
    rcc = _load_runtime_change_classifier()
    return rcc.load_runtime_path_classifier(validator_path)


def _run_decide(args: argparse.Namespace) -> int:
    changed_files = [
        line.strip()
        for line in Path(args.changed_files_file).read_text().splitlines()
        if line.strip()
    ]
    pr_labels = [lb.strip() for lb in (args.pr_labels or "").split(",") if lb.strip()]
    classifier = _load_classifier_from_validator(Path(args.runtime_path_validator))
    desired = decide_label(
        changed_files, pr_labels, classifier, source_repo=args.source_repo
    )
    to_add, to_remove = label_actions(desired, pr_labels)
    print(
        json.dumps(
            {"label": desired, "add": to_add, "remove": to_remove},
            sort_keys=True,
        )
    )
    return 0


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _run_apply(args: argparse.Namespace) -> int:
    pr = _gh_json(
        [
            "pr",
            "view",
            str(args.pr),
            "--repo",
            args.repo,
            "--json",
            "labels,files",
        ]
    )
    changed_files = [f["path"] for f in pr.get("files", [])]
    pr_labels = [lb["name"] for lb in pr.get("labels", [])]
    classifier = _load_classifier_from_validator(Path(args.runtime_path_validator))
    desired = decide_label(
        changed_files, pr_labels, classifier, source_repo=args.source_repo
    )
    to_add, to_remove = label_actions(desired, pr_labels)
    if not to_add and not to_remove:
        print(f"{args.repo}#{args.pr}: already {desired}, no change")
        return 0
    edit_args = ["pr", "edit", str(args.pr), "--repo", args.repo]
    for label in to_add:
        edit_args += ["--add-label", label]
    for label in to_remove:
        edit_args += ["--remove-label", label]
    subprocess.run(["gh", *edit_args], check=True)
    print(f"{args.repo}#{args.pr}: {desired} (added={to_add} removed={to_remove})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    decide = sub.add_parser(
        "decide", help="Print the label decision for a changed-file list (no I/O)."
    )
    decide.add_argument("--changed-files-file", required=True)
    decide.add_argument("--pr-labels", default="")
    decide.add_argument("--runtime-path-validator", required=True)
    decide.add_argument("--source-repo", default=None)
    decide.set_defaults(func=_run_decide)

    apply_ = sub.add_parser(
        "apply", help="Read a live PR with gh and set its runtime-train label."
    )
    apply_.add_argument("--repo", required=True)
    apply_.add_argument("--pr", required=True, type=int)
    apply_.add_argument("--runtime-path-validator", required=True)
    apply_.add_argument("--source-repo", default=None)
    apply_.set_defaults(func=_run_apply)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
