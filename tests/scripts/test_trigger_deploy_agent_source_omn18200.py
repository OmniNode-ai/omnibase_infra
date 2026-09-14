# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's own source is lane STATE (OMN-18200).

MEASURED 2026-09-14. ``omnibase_infra#3520`` (squash ``ead1f59b``, merged
06:49:18Z) fixed the deploy agent's own ``lab_overlay.build_and_import`` -- the
defect that had failed the ``images_pinned`` check on every lab-overlay re-apply
since 02:56Z. It changed exactly two files, both under
``scripts/deploy-agent/``. Run ``34815067432`` of ``runtime-rebuild-trigger.yml``
reported "No rebuild trigger: no runtime_change label or runtime path changes
detected" and skipped ``verify-lane-converged``.

That decline was correct against the path list as it stood, and the consequence
is a loop the agent cannot break out of on its own:

* no rebuild command is published, so the agent takes no job;
* ``deploy_agent.agent`` reaches ``executor.self_update`` from exactly two
  boundaries, ``PRE_ACCEPT`` and ``POST_TERMINAL``, both of which require a job;
* so the agent keeps running the clone it already has. Read on the lab host at
  07:10Z, ``/data/omninode/omnibase_infra`` was still at
  ``8fd252172098e81f4a4f63690c7725105c0bfd82`` -- behind the fix to itself.

A restart does not help and is the wrong reflex: there is no startup
self-update boundary, so the unit re-execs the same stale clone.

The canonical deploy-gate classifier is right to miss these paths -- its
question is "does this PR need deploy EVIDENCE", and the deploy agent is not
shipped in the runtime image. The rebuild trigger asks the wider question
OMN-18072 already separated out: "does this merge change what the lane RUNS".
The process that builds, recreates and verifies the lane answers yes.

Scope is the package and its launcher, not the whole directory: the agent's own
tests change no lane behaviour, and a full dev-lane rebuild is the accepted cost
of every match here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "trigger_rebuild_on_merge.py"
)


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("_trigger_rebuild_omn18200", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn18200"] = module
    spec.loader.exec_module(module)
    return module


# The exact changed-file list of omnibase_infra#3520, the merge that produced no
# trigger while being the fix to the very process that would have applied it.
PR_3520_FILES = [
    "scripts/deploy-agent/deploy_agent/lab_overlay.py",
    "scripts/deploy-agent/tests/unit/test_lab_overlay_omn18200.py",
]


@pytest.mark.unit
class TestDeployAgentSourceIsLaneState:
    def test_pr_3520_file_list_now_yields_a_trigger(self) -> None:
        """RED before this change: this exact list produced no rebuild."""
        module = _load_module()
        hits = module.find_lane_state_paths(PR_3520_FILES)  # type: ignore[attr-defined]
        assert "scripts/deploy-agent/deploy_agent/lab_overlay.py" in hits
        assert module.should_trigger(hits, [])  # type: ignore[attr-defined]

    def test_the_agents_own_tests_are_not_lane_state(self) -> None:
        """Positive control for the narrowing: the package matches, tests do not.

        Paired with the assertion above so the empty half is earned rather than
        asserted -- the same file list contains one path of each kind.
        """
        module = _load_module()
        hits = module.find_lane_state_paths(PR_3520_FILES)  # type: ignore[attr-defined]
        assert (
            "scripts/deploy-agent/tests/unit/test_lab_overlay_omn18200.py" not in hits
        )
        assert (  # type: ignore[attr-defined]
            module.find_lane_state_paths(
                ["scripts/deploy-agent/tests/unit/test_lab_overlay_omn18200.py"]
            )
            == []
        )

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/deploy-agent/deploy_agent/executor.py",
            "scripts/deploy-agent/deploy_agent/consumer.py",
            "scripts/deploy-agent/deploy_agent/lab_overlay.py",
            "scripts/deploy-agent/deploy/deploy-agent-launch.sh",
            "scripts/deploy-agent/deploy/preflight_port_free.sh",
        ],
    )
    def test_each_declared_deploy_agent_path_matches(self, path: str) -> None:
        module = _load_module()
        assert module.find_lane_state_paths([path]) == [path]  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/deploy-agent/tests/unit/test_executor.py",
            "scripts/deploy-agent/pyproject.toml",
            "scripts/deploy-agent/uv.lock",
            "scripts/pull-all.sh",
            "scripts/generate_deep_dive.py",
        ],
    )
    def test_paths_outside_the_named_two_do_not_match(self, path: str) -> None:
        """The widening is two named directories, not a directory opening."""
        module = _load_module()
        assert module.find_lane_state_paths([path]) == []  # type: ignore[attr-defined]

    def test_the_supplement_is_still_unioned_with_the_canonical_classifier(
        self,
    ) -> None:
        """Adding a pattern must not narrow or replace the canonical result."""
        module = _load_module()

        def _canonical(files: list[str]) -> list[str]:
            return [f for f in files if f.startswith("src/omnibase_infra/runtime/")]

        files = [
            "src/omnibase_infra/runtime/kernel.py",
            "scripts/deploy-agent/deploy_agent/executor.py",
        ]
        hits = module.classify_runtime_paths(files, _canonical)  # type: ignore[attr-defined]
        assert "src/omnibase_infra/runtime/kernel.py" in hits
        assert "scripts/deploy-agent/deploy_agent/executor.py" in hits
