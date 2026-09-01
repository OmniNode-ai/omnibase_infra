# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The governed dev-lane refresh must rebuild the dev-lane-only writers (OMN-17448).

The drift this closes
---------------------
``refresh_dev_lane.sh`` exports ``RUNTIME_BUILD_SERVICES_OVERRIDE``, and
``deploy-runtime.sh`` treats an explicit override as an instruction to touch
ONLY the named services — ``resolve_lane_runtime_services()`` deliberately does
not widen it, because a scoped build is an operator decision rather than a
default.

So a service declared only in ``docker-compose.dev-lane.yml`` and absent from
the refresh's build scope is never rebuilt. It does not disappear —
``restart: unless-stopped`` keeps it running — which is what makes the failure
quiet: a running projection writer on last-release code, indistinguishable from
a current one, writing rows nobody doubts. That is a slower version of the very
defect OMN-17448 exists to close, so it gets a gate rather than a comment.

Derivation, not a second hand-maintained list: the expected set is read from
``docker-compose.dev-lane.yml`` itself, so a writer added to the overlay later
extends this obligation automatically.

Related Tickets:
    - OMN-17448: this gate, and the two writers it covers
    - OMN-14873: ``RUNTIME_BUILD_SERVICES_OVERRIDE``, the scoping mechanism
    - OMN-15379: why the dev-lane overlay is a separate file at all
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"


def _dev_lane_only_services() -> set[str]:
    """Services the dev-lane overlay INTRODUCES, not ones it merely patches.

    An overlay entry that also exists in the base file is an override (an env
    block, a label); only a key absent from the base is a service that exists
    on this lane and no other, and therefore only those are in scope here.
    """
    overlay = yaml.safe_load(DEV_LANE_OVERLAY.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_COMPOSE.read_text(encoding="utf-8"))
    overlay_services = set(overlay.get("services", {}))
    base_services = set(base.get("services", {}))
    return overlay_services - base_services


def _refresh_build_scope() -> set[str]:
    """The service list ``refresh_dev_lane.sh`` hands to the build override."""
    script = REFRESH_SCRIPT.read_text(encoding="utf-8")

    core = re.search(r"readonly CORE_SERVICES=\((?P<items>[^)]*)\)", script)
    assert core is not None, (
        f"CORE_SERVICES moved or changed shape in {REFRESH_SCRIPT.name}; this "
        "drift check can no longer read it."
    )
    build = re.search(r"readonly REFRESH_BUILD_SERVICES=\((?P<items>[^)]*)\)", script)
    assert build is not None, (
        f"REFRESH_BUILD_SERVICES moved or changed shape in {REFRESH_SCRIPT.name}; "
        "this drift check can no longer read it."
    )

    scope = set(build.group("items").split())
    # The array expands CORE_SERVICES by reference; substitute its members.
    if '"${CORE_SERVICES[@]}"' in scope:
        scope.discard('"${CORE_SERVICES[@]}"')
        scope |= set(core.group("items").split())
    return scope


def test_every_dev_lane_only_service_is_in_the_refresh_build_scope() -> None:
    """A dev-lane-only service outside the build scope silently goes stale."""
    introduced = _dev_lane_only_services()
    assert introduced, (
        "the dev-lane overlay introduces no services of its own — the fixture, "
        "not the script, is what changed."
    )

    missing = sorted(introduced - _refresh_build_scope())
    assert not missing, (
        f"dev-lane-only service(s) {missing} are not in "
        "refresh_dev_lane.sh's REFRESH_BUILD_SERVICES, so a governed refresh "
        "never rebuilds them. They keep running (restart: unless-stopped) on a "
        "stale image, which reads as healthy."
    )


def test_the_build_override_is_wired_to_the_wider_array() -> None:
    """The export must read REFRESH_BUILD_SERVICES, not CORE_SERVICES.

    Declaring the wider array and then exporting the narrow one would pass the
    test above while changing nothing at all.
    """
    script = REFRESH_SCRIPT.read_text(encoding="utf-8")
    exports = re.findall(
        r'export RUNTIME_BUILD_SERVICES_OVERRIDE="\$\{(\w+)\[\*\]\}"', script
    )
    assert exports, (
        "no RUNTIME_BUILD_SERVICES_OVERRIDE export found in "
        f"{REFRESH_SCRIPT.name}; this drift check can no longer read it."
    )
    assert set(exports) == {"REFRESH_BUILD_SERVICES"}, (
        "RUNTIME_BUILD_SERVICES_OVERRIDE must be exported from "
        f"REFRESH_BUILD_SERVICES; found {sorted(set(exports))}"
    )


def test_core_services_stays_the_narrow_verification_surface() -> None:
    """CORE_SERVICES drives the health gate; widening it is a separate change.

    ``verify_dev_refresh.py`` carries its own ``CORE_SERVICE_NAMES`` that must
    stay matched to this array, and the pre-image-id capture and rollback
    anchors key off it too. Merging the writers into CORE_SERVICES would widen
    the verification surface without the live proof that change needs, so this
    asserts the split is intentional rather than an oversight.
    """
    assert not (_dev_lane_only_services() & _core_services()), (
        "a dev-lane-only writer was added to CORE_SERVICES. That widens the "
        "health gate, the pre-image capture and the rollback anchors, and it "
        "requires verify_dev_refresh.py's CORE_SERVICE_NAMES to move in the "
        "same change with live proof — see OMN-17448."
    )


def _core_services() -> set[str]:
    script = REFRESH_SCRIPT.read_text(encoding="utf-8")
    core = re.search(r"readonly CORE_SERVICES=\((?P<items>[^)]*)\)", script)
    assert core is not None
    return set(core.group("items").split())
