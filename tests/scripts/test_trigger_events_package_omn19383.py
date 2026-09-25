# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnimarket's shared events package is lane state the classifier missed (OMN-19383).

MEASURED: ``omnimarket#2813``'s change to ``src/omnimarket/events/runtime_deployment.py``
read "No rebuild trigger" (run 35961924174) although that module is imported by all
ten canonical redeploy-node handlers -- orchestrator, deploy-publish-monitor effect,
FSM reducer, prod-promotion-gate compute, grant-resolver effect, health-fact-resolver
effect. The change reached the ``.201`` lane only incidentally, via a later merge's
unrelated rebuild.

WHY THE WHOLE DIRECTORY, NOT ONLY THAT ONE FILE
------------------------------------------------
``runtime_deployment.py`` is not exceptional inside ``src/omnimarket/events/`` --
the whole package is shared cross-node infrastructure with the same handler
fan-in. Measured import counts against ``src/omnimarket/nodes/*/handlers/*.py``
at omnimarket dev HEAD (2026-09-24): ``events/__init__.py`` 252 importing
modules, ``events/topics.py`` 160, ``events/delegation.py`` 80,
``events/verification.py`` 51, ``events/generation.py`` 52, ``events/github.py``
49, ``events/ledger.py`` 31, ``events/runtime_deployment.py`` 10 -- and every
other file in the directory has at least one handler importer. A change to any
one of them changes what a node handler does at runtime the same way a change
inside ``nodes/*/handlers/`` itself does; the classifier's path-based rule just
never looked at this directory.

WHY THE SUPPLEMENT AND NOT THE CANONICAL LIST
-----------------------------------------------
Same reasoning as OMN-18671 (see ``test_trigger_dependency_manifests_omn18671.py``):
the canonical deploy-gate classifier (omniclaude's
``.github/actions/deploy-gate/validate_pr_deploy_required.py``) answers "does
this PR need deploy EVIDENCE" and is a required gate on four repositories, so
widening it there is a different, heavier decision than this trigger's "does
this merge change what the lane RUNS". The supplement is this repository's own
file and takes effect on merge without waiting on an omniclaude release cut.

``omnibase_infra#4038`` (OMN-19318, merged 2026-09-24T05:50:10Z) unified the
trigger's and the release-train's predicate call sites and added
``runtime_change`` label support, but touched neither path-pattern list, so it
did not fix this gap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"


def _load_trigger_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_trigger_rebuild_omn19383", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn19383"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_module() -> Any:
    return _load_trigger_module()


#: omnimarket#2813's exact changed-file list -- the merge measured as a
#: silent-inert runtime change.
PR_2813_FILES = [
    "src/omnimarket/events/runtime_deployment.py",
    "tests/test_deploy_publish_monitor_rebuild_rejected_dispatch_omn18816.py",
    "tests/test_deploy_rejection_lineage_tokens_omn19270.py",
]

#: Positive control for every zero below: a docs-only merge must still yield
#: nothing, so a green "no hits" result is evidence the matcher ran rather than
#: evidence it matched everything.
DOCS_ONLY_FILES = [
    "docs/runbooks/cold-lane-full-bringup.md",
    "README.md",
]


@pytest.mark.unit
class TestOmnimarketEventsPackageIsLaneState:
    def test_pr_2813_file_list_now_yields_a_trigger(self, trigger_module: Any) -> None:
        """RED before OMN-19383: this exact list was classified inert."""
        hits = trigger_module.find_lane_state_paths(PR_2813_FILES)
        assert "src/omnimarket/events/runtime_deployment.py" in hits
        assert trigger_module.should_trigger(hits, [])

    def test_a_different_events_file_is_also_lane_state(
        self, trigger_module: Any
    ) -> None:
        """The gap is the directory, not one named file."""
        hits = trigger_module.find_lane_state_paths(["src/omnimarket/events/topics.py"])
        assert hits == ["src/omnimarket/events/topics.py"]

    def test_a_nested_events_file_is_lane_state_too(self, trigger_module: Any) -> None:
        assert trigger_module.find_lane_state_paths(
            ["src/omnimarket/events/sub/deeper.py"]
        ) == ["src/omnimarket/events/sub/deeper.py"]

    def test_docs_only_merge_still_yields_nothing(self, trigger_module: Any) -> None:
        """The positive control for the zeros above."""
        assert trigger_module.find_lane_state_paths(DOCS_ONLY_FILES) == []
        assert not trigger_module.should_trigger([], [])

    def test_a_sibling_omnimarket_directory_is_lane_state_too(
        self, trigger_module: Any
    ) -> None:
        """OMN-19378 widened this to the whole installed tree, deliberately.

        events/ is not special. The same handler fan-in holds for the other
        directories the canonical list misses: measured against
        src/omnimarket/nodes/*/handlers/ at omnimarket dev (2026-09-24),
        routing/ is imported by 6 handler files, models/ by 25, enums/ by 19,
        inference/ by 38. So the rest of src/omnimarket/ is lane state as well.
        """
        path = "src/omnimarket/routing/some_unrelated_module.py"
        assert trigger_module.find_lane_state_paths([path]) == [path]


@pytest.mark.unit
class TestMatchedPatternIsNamed:
    """AC3 of OMN-18671's own pattern (attribution), preserved for this addition."""

    def test_attribution_names_the_events_pattern(self, trigger_module: Any) -> None:
        pairs = trigger_module.attribute_runtime_paths(
            ["src/omnimarket/events/runtime_deployment.py"]
        )
        assert pairs == [
            (
                "src/omnimarket/events/runtime_deployment.py",
                "src/omnimarket/events/**",
            )
        ]
