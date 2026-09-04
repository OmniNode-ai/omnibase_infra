# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A governed lane refresh must rebuild that lane's own writers (OMN-17448/OMN-17562).

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
each lane's overlay itself, so a writer added to an overlay later extends this
obligation automatically.

OMN-17562 widened this from the dev lane to every lane that introduces services
of its own. The stability-test lane is the one that made that necessary: it is
the surface the ``stability-proven`` premise of every live prod-promotion grant
is resolved from (OMN-15243), so a writer running last-release code THERE is
worse than the same drift on the throwaway dev lane, not better. Its refresh is
additionally scoped by construction — ``refresh_stability_lane.sh`` has always
exported an override, deliberately, to route around the open OMN-14262
BUILD_SOURCE selector mismatch on four release-only services — so a new service
is excluded from its build by default rather than by omission.

Related Tickets:
    - OMN-17448: this gate, and the first two writers it covered
    - OMN-17562: the full six-writer ADOPT set, on both mutable lanes
    - OMN-14873: ``RUNTIME_BUILD_SERVICES_OVERRIDE``, the scoping mechanism
    - OMN-15379: why the dev-lane overlay is a separate file at all
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
# Lane -> (its overlay, the governed refresh script that rebuilds it). Only
# lanes this repo actually deploys: prod and judge have no lane-only services
# and no refresh script here, and `lakshman` is its owner's to deploy.
GOVERNED_REFRESHES: dict[str, tuple[Path, Path]] = {
    "dev": (
        REPO_ROOT / "docker" / "docker-compose.dev-lane.yml",
        REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh",
    ),
    "stability-test": (
        REPO_ROOT / "docker" / "docker-compose.stability-test.yml",
        REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh",
    ),
}
# Top-level keys of the `services:` block. Scraped rather than parsed because
# the lane overlays carry compose's `!override` / `!reset` merge tags, which
# `yaml.safe_load` refuses — the same reason
# tests/unit/scripts/test_lane_census_manifest_parity.py scrapes.
_SERVICE_KEY = re.compile(r"^  ([a-z0-9][a-z0-9._-]*):\s*$", re.MULTILINE)


def _compose_services(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    start = text.index("\nservices:\n")
    body = text[start + len("\nservices:\n") :]
    end = re.search(r"^[a-z]", body, re.MULTILINE)
    if end is not None:
        body = body[: end.start()]
    return set(_SERVICE_KEY.findall(body))


def _lane_only_services(lane: str) -> set[str]:
    """Services this lane's overlay INTRODUCES, not ones it merely patches.

    An overlay entry that also exists in the base file is an override (an env
    block, a label); only a key absent from the base is a service that exists
    on this lane and no other, and therefore only those are in scope here.
    """
    overlay, _ = GOVERNED_REFRESHES[lane]
    return _compose_services(overlay) - _compose_services(BASE_COMPOSE)


def _refresh_build_scope(lane: str) -> set[str]:
    """The service list this lane's refresh hands to the build override."""
    _, refresh_script = GOVERNED_REFRESHES[lane]
    script = refresh_script.read_text(encoding="utf-8")

    core = re.search(r"readonly CORE_SERVICES=\((?P<items>[^)]*)\)", script)
    assert core is not None, (
        f"CORE_SERVICES moved or changed shape in {refresh_script.name}; this "
        "drift check can no longer read it."
    )
    build = re.search(r"readonly REFRESH_BUILD_SERVICES=\((?P<items>[^)]*)\)", script)
    assert build is not None, (
        f"REFRESH_BUILD_SERVICES moved or changed shape in {refresh_script.name}; "
        "this drift check can no longer read it."
    )

    scope = set(build.group("items").split())
    # The array expands CORE_SERVICES by reference; substitute its members.
    if '"${CORE_SERVICES[@]}"' in scope:
        scope.discard('"${CORE_SERVICES[@]}"')
        scope |= set(core.group("items").split())
    return scope


def _core_services(lane: str) -> set[str]:
    _, refresh_script = GOVERNED_REFRESHES[lane]
    script = refresh_script.read_text(encoding="utf-8")
    core = re.search(r"readonly CORE_SERVICES=\((?P<items>[^)]*)\)", script)
    assert core is not None
    return set(core.group("items").split())


@pytest.mark.parametrize("lane", sorted(GOVERNED_REFRESHES))
def test_every_lane_only_service_is_in_the_refresh_build_scope(lane: str) -> None:
    """A lane-only service outside the build scope silently goes stale."""
    introduced = _lane_only_services(lane)
    assert introduced, (
        f"the {lane} overlay introduces no services of its own — the fixture, "
        "not the script, is what changed."
    )

    _, refresh_script = GOVERNED_REFRESHES[lane]
    missing = sorted(introduced - _refresh_build_scope(lane))
    assert not missing, (
        f"{lane}-only service(s) {missing} are not in "
        f"{refresh_script.name}'s REFRESH_BUILD_SERVICES, so a governed refresh "
        "never rebuilds them. They keep running (restart: unless-stopped) on a "
        "stale image, which reads as healthy."
    )


@pytest.mark.parametrize("lane", sorted(GOVERNED_REFRESHES))
def test_the_build_override_is_wired_to_the_wider_array(lane: str) -> None:
    """The export must read REFRESH_BUILD_SERVICES, not CORE_SERVICES.

    Declaring the wider array and then exporting the narrow one would pass the
    test above while changing nothing at all. This is not hypothetical for the
    stability lane: it exported ``CORE_SERVICES`` directly until OMN-17562, so
    the wider array and the export had to move together or the writers would
    have been declared, censused, and never built.
    """
    _, refresh_script = GOVERNED_REFRESHES[lane]
    script = refresh_script.read_text(encoding="utf-8")
    exports = re.findall(
        r'export RUNTIME_BUILD_SERVICES_OVERRIDE="\$\{(\w+)\[\*\]\}"', script
    )
    assert exports, (
        "no RUNTIME_BUILD_SERVICES_OVERRIDE export found in "
        f"{refresh_script.name}; this drift check can no longer read it."
    )
    assert set(exports) == {"REFRESH_BUILD_SERVICES"}, (
        "RUNTIME_BUILD_SERVICES_OVERRIDE must be exported from "
        f"REFRESH_BUILD_SERVICES; found {sorted(set(exports))} in "
        f"{refresh_script.name}"
    )


@pytest.mark.parametrize("lane", sorted(GOVERNED_REFRESHES))
def test_core_services_stays_the_narrow_verification_surface(lane: str) -> None:
    """CORE_SERVICES drives the health gate; widening it is a separate change.

    Each refresh's verifier carries its own ``CORE_SERVICE_NAMES`` that must
    stay matched to this array, and the pre-image-id capture and rollback
    anchors key off it too. Merging the writers into CORE_SERVICES would widen
    the verification surface without the live proof that change needs, so this
    asserts the split is intentional rather than an oversight.
    """
    assert not (_lane_only_services(lane) & _core_services(lane)), (
        f"a {lane}-only writer was added to CORE_SERVICES. That widens the "
        "health gate, the pre-image capture and the rollback anchors, and it "
        "requires the lane verifier's CORE_SERVICE_NAMES to move in the "
        "same change with live proof — see OMN-17448."
    )
