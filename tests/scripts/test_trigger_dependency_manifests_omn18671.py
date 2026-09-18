# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dependency floor bump changes what the lane RUNS (OMN-18671).

MEASURED: ``omnimarket#2629`` (merge ``d4fb6ddd``, 2026-09-18T08:11:09Z) raised
the ``omnibase-infra`` floor 0.38.30 -> 0.38.31, which swaps the installed lab
binding table (Qwen3.6-35B -> Qwen3.8-27B). Rebuild-trigger run 35323097013
logged "No rebuild trigger: no runtime_change label or runtime path changes
detected" and skipped both verify and announce. The exact-name lab-pass receipt
query returns 0 artifacts for that merge sha, with the positive control
``5be72e12`` returning 4 -- so the zero is real, not a broken query.

Under rule 24(a) every runtime-affecting merge produces a lab pass. The runtime
image is BUILT from these manifests: a floor bump is the change, not a
description of one, so the merge reached the ``.201`` lane functionally inert.

WHY THE SUPPLEMENT AND NOT THE CANONICAL LIST
---------------------------------------------
The trigger consults omniclaude's deploy-gate classifier (fetched from that
repo's ``main`` at ``.github/actions/deploy-gate/validate_pr_deploy_required.py``)
UNIONED with ``LANE_STATE_PATH_PATTERNS`` here. There is one composed list, not
two competing ones. The canonical half answers "does this PR need deploy
EVIDENCE" and is a required gate on four repositories; widening it would demand
deploy evidence of every dependency bump in all four, and -- because the trigger
pins that repo's ``main``, which only advances on a release fast-forward -- the
widening would not reach this trigger until a release cut. The supplement is
this repository's own file and takes effect on merge. So the manifests go here,
which is where OMN-18072 and OMN-18572 already put lane state the canonical
classifier is right to miss.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
RUNTIME_PATH_VALIDATOR = REPO_ROOT / "tests" / "fixtures" / "runtime_path_classifier.py"


def _load_trigger_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_trigger_rebuild_omn18671", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn18671"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_module() -> Any:
    return _load_trigger_module()


#: The exact changed-file list of omnimarket#2629, the merge that produced no
#: rebuild, no verify, no announce and no receipt.
PR_2629_FILES = [
    ".github/workflows/runtime-rebuild-trigger.yml",
    "pyproject.toml",
    "tests/ci/test_lab_pass_settle_budget_pin_omn18638.py",
    "uv.lock",
]

#: Positive control for every zero below: a docs-only merge must still yield
#: nothing, so a green "no hits" result is evidence the matcher ran rather than
#: evidence it matched everything.
DOCS_ONLY_FILES = [
    "docs/runbooks/cold-lane-full-bringup.md",
    "README.md",
]


@pytest.mark.unit
class TestDependencyManifestsAreLaneState:
    def test_pr_2629_file_list_now_yields_a_trigger(self, trigger_module: Any) -> None:
        """RED before OMN-18671: this exact list was classified inert."""
        hits = trigger_module.find_lane_state_paths(PR_2629_FILES)
        assert "pyproject.toml" in hits
        assert "uv.lock" in hits
        assert trigger_module.should_trigger(hits, [])

    def test_docs_only_merge_still_yields_nothing(self, trigger_module: Any) -> None:
        """The positive control for the zero above."""
        assert trigger_module.find_lane_state_paths(DOCS_ONLY_FILES) == []
        assert not trigger_module.should_trigger([], [])

    def test_the_deploy_agent_subpackage_manifests_stay_excluded(
        self, trigger_module: Any
    ) -> None:
        """OMN-18200's exclusion is not collateral of this widening.

        That ticket admitted two named directories under the agent's subtree
        rather than opening it, and pinned the exclusion in its own test. The
        agent is a systemd unit on the lab host rather than a layer of the lane
        image, so its manifests are a different argument from the repository
        roots this ticket is about. Overturning it belongs to a ticket that
        makes that case.
        """
        assert (
            trigger_module.find_lane_state_paths(
                ["scripts/deploy-agent/pyproject.toml", "scripts/deploy-agent/uv.lock"]
            )
            == []
        )

    def test_a_nested_manifest_outside_the_build_is_not_lane_state(
        self, trigger_module: Any
    ) -> None:
        """Root-anchored, never ``**/pyproject.toml``.

        omnimarket carries ``experiments/adk_eval/track_a_adk/pyproject.toml``,
        which no runtime image installs from. Every match here costs a full
        dev-lane rebuild, so the match is anchored segment-wise from the repo
        root and a deeper manifest is deliberately missed.
        """
        assert (
            trigger_module.find_lane_state_paths(
                ["experiments/adk_eval/track_a_adk/pyproject.toml"]
            )
            == []
        )


@pytest.mark.unit
class TestMatchedPatternIsNamed:
    """AC3: the decision line names the PATTERN, not only the file."""

    def test_attribution_pairs_each_path_with_its_pattern(
        self, trigger_module: Any
    ) -> None:
        pairs = trigger_module.attribute_runtime_paths(["pyproject.toml", "uv.lock"])
        assert pairs == [("pyproject.toml", "pyproject.toml"), ("uv.lock", "uv.lock")]

    def test_a_canonical_hit_is_attributed_to_the_canonical_gate(
        self, trigger_module: Any
    ) -> None:
        """A path this module does not list came from the canonical classifier.

        Re-deriving WHICH canonical pattern matched would mean a second
        implementation of that repository's matcher here, which is exactly the
        divergence this module exists to prevent. Naming the source is honest;
        guessing the pattern would not be.
        """
        pairs = trigger_module.attribute_runtime_paths(["src/omnimarket/nodes/n/h.py"])
        assert pairs == [
            ("src/omnimarket/nodes/n/h.py", trigger_module.CANONICAL_CLASSIFIER_SOURCE)
        ]

    def test_directory_pattern_is_named_whole(self, trigger_module: Any) -> None:
        pairs = trigger_module.attribute_runtime_paths(["docker/migrations/0001.sql"])
        assert pairs == [("docker/migrations/0001.sql", "docker/migrations/**")]

    def test_cli_decision_line_names_the_matched_pattern(
        self, trigger_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED before OMN-18671: the line carried ``files_matched`` only."""

        def _explode(**_kwargs: object) -> None:
            raise AssertionError("dry-run must not publish")

        monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)

        result = CliRunner().invoke(
            trigger_module.main,
            [
                "--changed-files",
                ",".join(PR_2629_FILES),
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
                "--base-branch",
                "dev",
                "--source-repo",
                "omnimarket",
                # A sibling trigger supplies the omnibase_infra commit the lane
                # is rebuilt at; the sibling's own merge sha names no commit here.
                "--primary-ref",
                "18f9eaead1c0f1e6a1d4b0c9f2a3e5d7c8b9a0f1",
                "--source-sha",
                "d4fb6dddddf6dcac289e16df5076a8fbbe0f4fa8",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Runtime change detected" in result.output
        assert "matched_by=" in result.output
        assert "pyproject.toml <- pyproject.toml" in result.output
        assert "uv.lock <- uv.lock" in result.output
