# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16906 — workflow input defaults must match their declared type.

This is the PR-time half of the fix for a silent, day-long delivery outage.

``deliver-dev-candidate-to-staging.yml`` (OMN-15796) returned GitHub's
``startup_failure`` on every run from the day it landed. ``startup_failure``
means the run graph never compiled: **no job is created, so there is no job
log, no failed step, and no red check on anything a human looks at.** Every
omnibase_infra ``dev`` merge went out merged-but-undelivered while onex-dev
kept serving an older candidate — the exact class OMN-15796 was built to close.

Root cause, isolated by single-variable bisect on a scratch branch
(control run 33224162317 = ``startup_failure``; the same file with one
character class changed = run 33224392268, graph compiled):

``build-workspace-candidate-runtime.yml`` declared its ``no-cache``
``workflow_call`` input as ``type: boolean`` with ``default: "false"`` — a YAML
*string*. GitHub type-checks a callee's ``workflow_call`` input defaults while
compiling the **caller's** graph, so the callee's own dispatch runs stayed
green (33107333244) while every caller run died before starting.

Two properties of that failure make a static gate the right response:

* it is **invisible at runtime** — there is nothing to observe, so no amount of
  log-reading or check-watching finds it; and
* it is **latent under ``workflow_dispatch``** — GitHub tolerates the same bad
  default there, so it sits in the tree looking healthy until someone adds
  ``workflow_call`` and copies the block down. That is literally how OMN-15796
  introduced it: the bad ``workflow_dispatch`` default already existed, and the
  new ``workflow_call`` block inherited it.

So the ratchet covers both trigger blocks, repo-wide.

Static artifact assertions only — no network, no ``gh``.
"""

from __future__ import annotations

import subprocess  # fixed argv, no shell, repo-local script
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_workflow_input_default_types import (
    main,
    scan_document,
    scan_paths,
    workflow_files,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CHECKER = REPO_ROOT / "scripts" / "ci" / "check_workflow_input_default_types.py"

# The workflow whose caller-side compile the incident actually broke.
RUNTIME_BUILD = WORKFLOWS / "build-workspace-candidate-runtime.yml"


class TestRepoIsClean:
    """The ratchet itself: this is what goes red if the defect comes back."""

    def test_no_workflow_input_default_mismatches_repo_wide(self) -> None:
        violations = scan_paths(workflow_files(WORKFLOWS))
        assert not violations, "\n".join(v.render() for v in violations)

    def test_the_scan_actually_inspected_workflows(self) -> None:
        """A clean result is only evidence if something was scanned.

        Guards the failure mode where a glob change quietly makes the ratchet
        pass by looking at nothing — an empty result is not evidence of absence.
        """
        files = workflow_files(WORKFLOWS)
        assert len(files) > 20, f"only found {len(files)} workflow files"

    def test_checker_exits_zero_as_invoked_by_pre_commit(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--workflows-dir", str(WORKFLOWS)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestTheIncidentShapeSpecifically:
    """Pin the exact declaration that produced run 33169436998."""

    def test_runtime_build_no_cache_workflow_call_default_is_a_real_boolean(
        self,
    ) -> None:
        document = yaml.safe_load(RUNTIME_BUILD.read_text(encoding="utf-8"))
        triggers = document.get("on", document.get(True))
        for trigger in ("workflow_call", "workflow_dispatch"):
            default = triggers[trigger]["inputs"]["no-cache"]["default"]
            assert isinstance(default, bool), (
                f"on.{trigger}.inputs.no-cache default is {default!r} "
                f"({type(default).__name__}); a string default on a "
                "type: boolean input fails the CALLER's graph compile with "
                "startup_failure (OMN-16906)"
            )


class TestCheckerSemantics:
    """The checker must actually catch the shapes it claims to catch."""

    def test_string_default_on_boolean_input_is_a_violation(self) -> None:
        document = yaml.safe_load(
            """
on:
  workflow_call:
    inputs:
      no-cache:
        required: false
        default: "false"
        type: boolean
"""
        )
        violations = scan_document("fixture.yml", document)
        assert len(violations) == 1
        assert violations[0].input_name == "no-cache"
        assert violations[0].trigger == "workflow_call"
        assert "startup_failure" not in violations[0].detail  # detail names the fix
        assert "true/false" in violations[0].detail

    def test_the_same_defect_under_workflow_dispatch_is_also_a_violation(self) -> None:
        """Latent, not harmless — this is the state OMN-15796 copied from."""
        document = yaml.safe_load(
            """
on:
  workflow_dispatch:
    inputs:
      run_full_tests:
        default: "true"
        type: boolean
"""
        )
        assert len(scan_document("fixture.yml", document)) == 1

    def test_well_typed_inputs_are_accepted(self) -> None:
        document = yaml.safe_load(
            """
on:
  workflow_call:
    inputs:
      no-cache:
        default: false
        type: boolean
      sibling_ref:
        default: "dev"
        type: string
      attempts:
        default: 3
        type: number
  workflow_dispatch:
    inputs:
      lane:
        default: "dev"
        type: choice
        options: ["dev", "stability-test"]
"""
        )
        assert scan_document("fixture.yml", document) == []

    def test_boolean_default_on_a_number_input_is_a_violation(self) -> None:
        """bool subclasses int in Python; the checker must not let that pass."""
        document = yaml.safe_load(
            """
on:
  workflow_call:
    inputs:
      attempts:
        default: true
        type: number
"""
        )
        assert len(scan_document("fixture.yml", document)) == 1

    def test_choice_default_outside_options_is_a_violation(self) -> None:
        document = yaml.safe_load(
            """
on:
  workflow_dispatch:
    inputs:
      lane:
        default: "prod"
        type: choice
        options: ["dev", "stability-test"]
"""
        )
        violations = scan_document("fixture.yml", document)
        assert len(violations) == 1
        assert "options" in violations[0].detail

    def test_input_without_a_declared_type_is_not_this_ratchets_business(self) -> None:
        document = yaml.safe_load(
            """
on:
  workflow_dispatch:
    inputs:
      legacy:
        default: "false"
"""
        )
        assert scan_document("fixture.yml", document) == []

    def test_input_without_a_default_is_not_a_violation(self) -> None:
        document = yaml.safe_load(
            """
on:
  workflow_call:
    inputs:
      git-ref:
        required: true
        type: string
"""
        )
        assert scan_document("fixture.yml", document) == []

    def test_yaml_1_1_bare_on_key_is_resolved(self) -> None:
        """PyYAML resolves a bare ``on:`` to boolean True.

        A checker that only looked up the literal string key would scan zero
        inputs in every real workflow file and report clean forever.
        """
        document = {
            True: {
                "workflow_call": {"inputs": {"f": {"type": "boolean", "default": "no"}}}
            }
        }
        assert len(scan_document("fixture.yml", document)) == 1


class TestCheckerFailsClosed:
    def test_missing_workflows_dir_is_an_error_not_a_clean_pass(
        self, tmp_path: Path
    ) -> None:
        assert main(["--workflows-dir", str(tmp_path / "nope")]) == 1

    def test_violation_exits_nonzero(self, tmp_path: Path) -> None:
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        (workflows / "bad.yml").write_text(
            'on:\n  workflow_call:\n    inputs:\n      f:\n        type: boolean\n        default: "false"\n',
            encoding="utf-8",
        )
        assert main(["--workflows-dir", str(workflows)]) == 1
