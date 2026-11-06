# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The restart-bound rule for the alarm lane's crash-looping projection writer.

OMN-22644. The savings writer crash-looped unobserved for nine days, so while a
census lane runs it as a service its restart bound is the tightest on the lane.
omnibase_infra#5518 folded the writer into the dev runtime, and a lane manifest
then declares it ``profile_gated`` (expected absent, no restart bound). The rule
accepts exactly those two states and refuses everything else, so dropping the
writer from the manifest, or declaring it a service without a bound, still fails.

The rule lives here, apart from the test that reads the lab's real manifest, so
that its refusals are shown by manifests built in the test and not only by the
lab's current one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def assert_writer_bound_rule(
    manifest: dict[str, Any], lane: str, writer: str, reference: str
) -> None:
    """Raise AssertionError unless ``writer`` is bounded tighter than ``reference``
    on ``lane``, or the manifest declares it ``profile_gated`` on that lane."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from lane_census_plan import restart_bounds_for_lane

    bounds = restart_bounds_for_lane(manifest, lane)
    if writer in bounds:
        assert reference in bounds, f"{reference} carries no restart bound on {lane}"
        assert bounds[writer] < bounds[reference], (
            f"{writer} bound {bounds[writer]} is not tighter than "
            f"{reference} bound {bounds[reference]} on {lane}"
        )
        return
    declared = {
        service["name"]: service
        for service in manifest["lanes"][lane].get("services") or []
    }
    assert writer in declared, f"{writer} is neither bounded nor declared on {lane}"
    assert declared[writer]["kind"] == "profile_gated", (
        f"{writer} on {lane} has no restart bound and is declared "
        f"{declared[writer]['kind']!r}, not 'profile_gated'"
    )
