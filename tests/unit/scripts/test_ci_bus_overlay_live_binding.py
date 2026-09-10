# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-repo binding for the CI bus overlay contract [OMN-18060].

THE COUPLING THIS GATE MAKES VISIBLE. ``config/ci_bus_lanes.yaml`` is written
in omnimarket and validated in omnibase_infra by ``ModelCiBusOverlay``, which
is ``extra="forbid"``. The strictness is correct -- an unknown key in that file
is a typo that would otherwise route a publisher to a lane it never declared --
but it means the two halves of one contract live in two repositories, and
nothing in either repository compared them until the dev-lane rebuild trigger
tried to publish.

That is a producer-side change failing on a consumer-side merge, hours later,
in a workflow that had nothing to do with it. It has now happened twice:

  * OMN-18012 (2026-09-07) -- ``security_protocol`` / ``sasl_mechanism`` added
    to the ``dev`` lane; every runtime-touching PR red on
    ``Extra inputs are not permitted ... lanes.dev.sasl_mechanism``
    (run 34160709151).
  * OMN-18060 (2026-09-09) -- ``projection_readback`` added at 02:15Z by
    omnimarket#2420; ten consecutive rebuild-trigger runs red on
    ``lanes.dev.projection_readback Extra inputs are not permitted``.

Both were found by the failure, not by a check. This binding is the check: it
loads the LIVE omnimarket overlay at ``origin/dev`` and validates it against
this repository's model, so a third occurrence reds here -- on a surface whose
job is the contract -- instead of on whichever unrelated PR merges next.

WHY IT IS NOT A NETWORK CALL FROM A UNIT TEST. Unit tests in this repository
are hermetic and stay that way. The fetch is a workflow-level sparse checkout
(``.github/workflows/ci-bus-overlay-binding.yml``), and the test reads the
resulting path out of ``CI_BUS_OVERLAY_LIVE_PATH``. Locally the variable is
unset and the binding case skips; the wiring case below does not, so the
workflow that supplies the variable cannot be quietly removed or repointed
without a red test in this file.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
BINDING_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-bus-overlay-binding.yml"

#: Set by the binding workflow to the checked-out omnimarket overlay. Unset
#: everywhere else, which is what makes the local run hermetic.
LIVE_OVERLAY_ENV_VAR = "CI_BUS_OVERLAY_LIVE_PATH"

#: The producer-side path this gate is bound to, spelled once.
OVERLAY_REPO = "OmniNode-ai/omnimarket"
OVERLAY_REF = "dev"
OVERLAY_PATH_IN_REPO = "config/ci_bus_lanes.yaml"


def _import_trigger_module():
    """Import the publisher module by file path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_binding", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.unit
class TestLiveOverlayBinding:
    """The live producer-side file must satisfy the consumer-side model."""

    def test_live_omnimarket_overlay_validates(self) -> None:
        """omnimarket@dev's overlay loads under this repository's strict model."""
        declared = os.environ.get(LIVE_OVERLAY_ENV_VAR, "").strip()
        if not declared:
            pytest.skip(
                f"{LIVE_OVERLAY_ENV_VAR} is unset: this case runs in "
                "ci-bus-overlay-binding.yml, which checks out "
                f"{OVERLAY_REPO}@{OVERLAY_REF}:{OVERLAY_PATH_IN_REPO} and "
                "exports the path. Local runs are deliberately hermetic."
            )

        overlay = Path(declared)
        # A checkout that silently produced nothing must not read as a pass:
        # the empty result is the exact shape this whole gate exists to refuse.
        assert overlay.is_file(), (
            f"{LIVE_OVERLAY_ENV_VAR}={declared!r} does not name a file. The "
            "sparse checkout did not produce the overlay, so this run proves "
            "nothing about the contract and is a wiring failure, not a pass."
        )

        mod = _import_trigger_module()
        # Raises ValueError naming the offending key on any skew; the message
        # is what tells the next reader which key to teach the model.
        model = mod.load_ci_bus_overlay(overlay)

        assert model.lanes, (
            f"{overlay} declares no lanes. An overlay with an empty `lanes` "
            "map validates structurally while resolving no broker at all."
        )

    def test_live_overlay_declares_the_lane_the_publisher_uses(self) -> None:
        """The publisher hardcodes ``--bus-lane dev``; the overlay must declare it."""
        declared = os.environ.get(LIVE_OVERLAY_ENV_VAR, "").strip()
        if not declared:
            pytest.skip(f"{LIVE_OVERLAY_ENV_VAR} is unset (see above)")

        mod = _import_trigger_module()
        model = mod.load_ci_bus_overlay(Path(declared))

        assert "dev" in model.lanes, (
            "runtime-rebuild-trigger.yml publishes with --bus-lane dev. An "
            f"overlay without a `dev` lane fails that run: {sorted(model.lanes)}"
        )


@pytest.mark.unit
class TestBindingWorkflowIsWired:
    """The workflow that supplies the live overlay is itself pinned.

    Without these, the binding case above degrades to a permanent skip the
    moment the workflow is deleted, renamed, or repointed -- and a permanent
    skip is indistinguishable from a passing gate in a summary line.
    """

    def test_binding_workflow_exists(self) -> None:
        assert BINDING_WORKFLOW.is_file(), (
            f"{BINDING_WORKFLOW.name} is the only thing that sets "
            f"{LIVE_OVERLAY_ENV_VAR}; without it the cross-repo binding never runs."
        )

    def test_binding_workflow_checks_out_the_producer_side_file(self) -> None:
        workflow = yaml.safe_load(BINDING_WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["ci-bus-overlay-binding"]["steps"]

        checkouts = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout")
            and step.get("with", {}).get("repository") == OVERLAY_REPO
        ]
        assert checkouts, f"no checkout of {OVERLAY_REPO} in {BINDING_WORKFLOW.name}"

        with_block = checkouts[0]["with"]
        assert with_block["ref"] == OVERLAY_REF, (
            "the binding must read the branch the publisher reads at merge "
            f"time ({OVERLAY_REF}), not a pinned or stale ref"
        )
        assert OVERLAY_PATH_IN_REPO in str(with_block.get("sparse-checkout", ""))

    def test_binding_workflow_exports_the_env_var_to_pytest(self) -> None:
        text = BINDING_WORKFLOW.read_text(encoding="utf-8")

        assert LIVE_OVERLAY_ENV_VAR in text, (
            f"{BINDING_WORKFLOW.name} must export {LIVE_OVERLAY_ENV_VAR}; "
            "without it every case in this file skips and the job is green "
            "while checking nothing."
        )
        assert Path(__file__).name in text, (
            f"{BINDING_WORKFLOW.name} must run this test file by name"
        )

    def test_binding_workflow_runs_on_a_standing_schedule(self) -> None:
        """A producer-side key can land with no omnibase_infra PR at all.

        omnimarket#2420 merged at 02:15Z and the skew existed from that moment.
        A PR-only gate would have caught it on this repository's next PR, which
        is better than the trigger catching it but is still someone else's
        merge paying for it. The schedule is the surface that owns it.
        """
        workflow = yaml.safe_load(BINDING_WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML resolves a bare `on:` key to the boolean True.
        triggers = workflow.get("on", workflow.get(True))

        assert "schedule" in triggers, (
            f"{BINDING_WORKFLOW.name} needs a schedule: a producer-side change "
            "in omnimarket does not create a pull_request here."
        )
        assert "pull_request" in triggers
