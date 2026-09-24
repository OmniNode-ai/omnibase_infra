# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Packaged source the runtime image installs is lane state (OMN-19378).

MEASURED: omnimarket#2813 (merge ``c4b70696``, 2026-09-24T05:53:05Z) changed
``src/omnimarket/events/runtime_deployment.py``, which runtime-effects imports.
Rebuild-trigger run 35961924174 logged "No rebuild trigger: no runtime_change
label or runtime path changes detected", so the dev lane only received the
change through a later, unrelated deploy.

The canonical deploy-gate classifier matches a fixed set of subtrees
(``nodes/``, ``runtime/``, ``handlers/``, ``services/``, ...). Replayed over
omnimarket dev at 2026-09-24T10:30Z it misses 939 of 4393 files under
``src/omnimarket`` (every file under ``events/``, ``models/``, ``enums/``,
``adapters/``, ``delegation/``, ...) and 1162 of 3322 under
``src/omnibase_infra`` (``models/``, ``event_bus/``, ``errors/``, ...).
OMN-19383 added ``events/``; this covers the rest of both trees. The lane
installs both trees whole: omnimarket from the staged clone
(``omnimarket @ file:///workspace/sibling-repos/omnimarket``, read back from
the running ``omninode-runtime`` container's ``direct_url.json``) and
omnibase_infra as the image's own project.

The test double ``tests/fixtures/runtime_path_classifier.py`` treats every
``src/omnimarket/`` path as runtime, which is why no existing test saw this.
These tests therefore drive the supplement with a canonical classifier that
returns nothing, which is what the real one returned for #2813.
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


def _load_trigger_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_trigger_rebuild_omn19378", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn19378"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_module() -> Any:
    return _load_trigger_module()


def _canonical_misses_everything(_changed_files: list[str]) -> list[str]:
    """What the real canonical classifier returned for omnimarket#2813."""
    return []


#: The exact changed-file list of omnimarket#2813.
PR_2813_FILES = [
    "tests/test_deploy_rejection_lineage_tokens_omn19270.py",
    "src/omnimarket/events/runtime_deployment.py",
    "tests/test_deploy_publish_monitor_rebuild_rejected_dispatch_omn18816.py",
]

#: The exact changed-file list of omnimarket#2817.
PR_2817_FILES = [
    "scripts/ci/run_delegation_response_contract_conformance.py",
    "src/omnimarket/delegation/response_contract_conformance_runner.py",
    "tests/unit/delegation/test_response_contract_conformance_classes_omn18700.py",
    "tests/unit/delegation/test_response_contract_conformance_runner_omn18700.py",
]

#: Positive control for the zeros below: a merge that touches only tests and
#: docs still yields nothing, so a hit above is evidence the pattern matched
#: the source file and not evidence it matches everything.
TESTS_AND_DOCS_ONLY = [
    "tests/test_deploy_rejection_lineage_tokens_omn19270.py",
    "tests/unit/runtime/test_service_kernel.py",
    "docs/runbooks/cold-lane-full-bringup.md",
    "README.md",
]


@pytest.mark.unit
class TestPackagedSourceIsLaneState:
    def test_runtime_deployment_module_classifies_runtime(
        self, trigger_module: Any
    ) -> None:
        """Positive control named by the ticket (OMN-19383 made this one GREEN)."""
        runtime_paths = trigger_module.classify_runtime_paths(
            PR_2813_FILES, _canonical_misses_everything
        )
        assert runtime_paths == ["src/omnimarket/events/runtime_deployment.py"]
        assert trigger_module.should_trigger(runtime_paths, [])

    @pytest.mark.parametrize(
        "path",
        [
            "src/omnimarket/models/model_runtime_target.py",
            "src/omnimarket/enums/enum_deploy_outcome.py",
            "src/omnimarket/adapters/adapter_kafka.py",
            "src/omnimarket/configs/routing_tiers.yaml",
            "src/omnimarket/nodes/node_x_effect/metadata.yaml",
            "src/omnibase_infra/models/model_event_envelope.py",
            "src/omnibase_infra/event_bus/event_bus_kafka.py",
            "src/omnibase_infra/errors/error_infra.py",
        ],
    )
    def test_other_installed_subtrees_the_canonical_list_misses(
        self, trigger_module: Any, path: str
    ) -> None:
        assert trigger_module.find_lane_state_paths([path]) == [path]

    def test_tests_and_docs_still_yield_nothing(self, trigger_module: Any) -> None:
        assert trigger_module.find_lane_state_paths(TESTS_AND_DOCS_ONLY) == []
        assert (
            trigger_module.classify_runtime_paths(
                TESTS_AND_DOCS_ONLY, _canonical_misses_everything
            )
            == []
        )

    @pytest.mark.parametrize(
        "path",
        [
            # Not the package: a sibling tree with the package name as a prefix.
            "src/omnimarket_extras/x.py",
            # Not at the root: a vendored copy under another directory.
            "experiments/src/omnimarket/x.py",
            # A different repository's package, not installed by this lane.
            "src/omniintelligence/models/x.py",
        ],
    )
    def test_the_match_is_root_anchored_to_the_two_packages(
        self, trigger_module: Any, path: str
    ) -> None:
        assert trigger_module.find_lane_state_paths([path]) == []

    def test_attribution_names_the_package_pattern(self, trigger_module: Any) -> None:
        path = "src/omnimarket/models/model_runtime_target.py"
        assert trigger_module.attribute_runtime_paths([path]) == [
            (path, "src/omnimarket/**")
        ]


@pytest.mark.unit
def test_cli_dry_run_on_pr_2817_now_triggers(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The decision line for omnimarket#2817, merged as No rebuild (47871f79).

    Its one source file is under ``delegation/``, which neither the canonical
    list nor the OMN-19383 ``events/`` entry covers. RED before OMN-19378.
    """

    def _explode(**_kwargs: object) -> None:
        raise AssertionError("dry-run must not publish")

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)
    validator = tmp_path / "validate_pr_deploy_required.py"
    validator.write_text(
        "def find_runtime_paths(changed_files):\n    return []\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            ",".join(PR_2817_FILES),
            "--runtime-path-validator",
            str(validator),
            "--base-branch",
            "dev",
            "--source-repo",
            "omnimarket",
            "--primary-ref",
            "18f9eaead1c0f1e6a1d4b0c9f2a3e5d7c8b9a0f1",
            "--source-sha",
            "47871f791c984b1fc39ce4cbe5f7c88e9dadbe40",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Runtime change detected" in result.output
    assert (
        "src/omnimarket/delegation/response_contract_conformance_runner.py"
        " <- src/omnimarket/**" in result.output
    )
