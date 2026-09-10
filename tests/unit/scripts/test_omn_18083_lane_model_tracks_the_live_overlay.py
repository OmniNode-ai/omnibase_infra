# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18083 — the rebuild trigger's lane model must accept the LIVE overlay.

`scripts/trigger_rebuild_on_merge.py` parses omnimarket's
``config/ci_bus_lanes.yaml`` through ``ModelCiBusOverlay``, whose lane model
sets ``extra="forbid"``. When omnimarket declares a lane key that model does
not know, the overlay stops validating and **every merge that reaches the
validation stops rebuilding the runtime**.

That has now happened twice:

* ``security_protocol`` / ``sasl_mechanism`` (OMN-18012) — recorded in
  ``ModelCiBusLane``'s own docstring: *"a model that knew only ``broker``
  rejected the live overlay outright and took the dev-lane redeploy trigger red
  on every runtime PR"* (omnibase_infra run 34160709151).
* ``projection_readback`` (OMN-18060, omnimarket#2420) — five consecutive red
  runs, 2026-09-09 12:42:50Z through 15:50:51Z, including the merge of #3356.

**The strictness is correct and must stay.** An unknown key in this overlay is a
typo that would otherwise route a publisher to a default it never declared, so
widening to ``extra="ignore"`` would be a worse defect than the one this file
exists to catch. What was missing is anything that FAILS when the overlay grows
a key the model does not know — which is why the docstring's warning did not
prevent the second occurrence. Prose had already failed once.

WHY THIS TEST READS THE LIVE FILE RATHER THAN A FIXTURE
------------------------------------------------------
A fixture would pin the shape this repo *believes* omnimarket declares, and
would have passed happily through both outages. The failure mode is precisely
that the two repos disagree, so the only test that can catch it is one that
parses what omnibase_infra actually consumes at run time.

That makes this test dependent on a sibling checkout. Local runs and the
ordinary hermetic suite skip when the overlay is absent because a missing clone
is not evidence of drift. The dedicated ``ci-bus-overlay-binding`` workflow
checks out the live file, exports ``CI_BUS_OVERLAY_LIVE_PATH`` and fails closed
before pytest if that checkout is missing. When ``OMNI_HOME`` is set it is
authoritative, so a stale ancestor checkout cannot silently win precedence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.trigger_rebuild_on_merge import ModelCiBusOverlay

pytestmark = pytest.mark.unit

_LIVE_OVERLAY_ENV_VAR = "CI_BUS_OVERLAY_LIVE_PATH"
_OVERLAY_RELATIVE = Path("config/ci_bus_lanes.yaml")


def _live_overlay_path() -> Path | None:
    """Locate omnimarket's overlay beside this checkout, or via OMNI_HOME.

    ``CI_BUS_OVERLAY_LIVE_PATH`` is authoritative when supplied by the
    dedicated binding workflow. ``OMNI_HOME`` is authoritative otherwise.
    Returns None when the selected checkout is absent; the caller distinguishes
    a configured binding path from an ordinary hermetic-suite skip.
    """
    ci_overlay = os.environ.get("CI_BUS_OVERLAY_LIVE_PATH")
    if ci_overlay:
        candidate = Path(ci_overlay)
        if not candidate.is_file():
            pytest.fail(
                f"{_LIVE_OVERLAY_ENV_VAR}={ci_overlay!r} does not name a file. "
                "The configured live-overlay guard proves nothing without the "
                "producer-side config/ci_bus_lanes.yaml checkout."
            )
        return candidate

    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        candidate = Path(omni_home) / "omnimarket" / _OVERLAY_RELATIVE
        return candidate if candidate.is_file() else None

    candidates: list[Path] = []
    # The CI job checks the overlay out beside the workspace as .ci-bus-overlay.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".ci-bus-overlay" / _OVERLAY_RELATIVE)
        candidates.append(parent / "omnimarket" / _OVERLAY_RELATIVE)
    return next((c for c in candidates if c.is_file()), None)


def test_the_live_overlay_still_validates_against_this_repos_lane_model() -> None:
    """The guard. A key omnimarket adds that this model rejects fails HERE.

    Without this, the first thing that notices is the dev-lane rebuild trigger,
    and the symptom arrives as "no merge rebuilds the runtime" rather than as
    "the two repos disagree about a lane key".
    """
    overlay = _live_overlay_path()
    if overlay is None:
        if os.environ.get("CI_BUS_OVERLAY_LIVE_PATH"):
            pytest.fail(
                "the configured CI_BUS_OVERLAY_LIVE_PATH does not contain "
                "omnimarket's live "
                "config/ci_bus_lanes.yaml; the cross-repository drift guard "
                "is not wired to its required input"
            )
        pytest.skip(
            "omnimarket's config/ci_bus_lanes.yaml is not present beside this "
            "checkout; a missing sibling clone is not evidence of drift"
        )

    try:
        data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        pytest.fail(f"could not read or parse live overlay {overlay}: {exc}")
    if not isinstance(data, dict):
        pytest.fail(f"live overlay {overlay} must contain a YAML mapping")

    try:
        ModelCiBusOverlay.model_validate(data)
    except ValidationError as exc:
        pytest.fail(
            f"{overlay} no longer validates against ModelCiBusOverlay in this "
            f"repo. omnimarket has declared a lane key this model does not "
            f"know, so the dev-lane rebuild trigger will go red on every merge "
            f"that reaches the overlay parse. Teach the model the new field — "
            f"do NOT relax extra='forbid', which exists to catch typos that "
            f"would route a publisher to an undeclared default.\n\n{exc}"
        )


def test_extra_forbid_is_retained_so_a_typo_is_still_refused() -> None:
    """AC2. The tempting one-character fix is the one that must not happen.

    Widening to extra="ignore" would make the test above pass forever and
    silently accept a misspelled key — routing a publisher to a default it
    never declared, which is the hole the strictness defends. This pins the
    strictness itself, so a future fix cannot buy green by removing it.
    """
    lane = {"broker": "inmemory", "brokerr": "typo"}
    with pytest.raises(ValidationError, match="brokerr"):
        ModelCiBusOverlay.model_validate(
            {"default": "inmemory", "lanes": {"dev": lane}}
        )


def test_a_declared_projection_readback_is_accepted() -> None:
    """The specific shape OMN-18060 added, pinned independently of the live file.

    The live-overlay test above is the general guard; this one states the shape
    in the repo so the intent survives even if the sibling checkout is absent
    and that test skips.
    """
    lane = {
        "broker": "inmemory",
        "projection_readback": {"dsn_env": "CHAIN_CANARY_PROJECTION_DSN"},
    }
    parsed = ModelCiBusOverlay.model_validate(
        {"default": "inmemory", "lanes": {"dev": lane}}
    )
    readback = parsed.lanes["dev"].projection_readback
    assert readback is not None
    assert readback.dsn_env == "CHAIN_CANARY_PROJECTION_DSN"


def test_an_unknown_key_inside_projection_readback_is_still_refused() -> None:
    """The nested block is typed, not a free dict.

    A dict[str, str] would accept `dsn` or `dsn_value` — the latter being the
    shape that would carry a DSN by VALUE, which the overlay's own contract
    forbids ("what is declared is a NAME, never a value"). Typing the block
    keeps that refusal.
    """
    lane = {
        "broker": "inmemory",
        "projection_readback": {"dsn_env": "X", "dsn_value": "postgres://leak"},
    }
    with pytest.raises(ValidationError, match="dsn_value"):
        ModelCiBusOverlay.model_validate(
            {"default": "inmemory", "lanes": {"dev": lane}}
        )
