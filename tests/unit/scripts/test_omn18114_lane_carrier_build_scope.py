# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's restart set and its refresh build scope must name the same services.

The drift this closes
---------------------
``deploy-runtime.sh`` keeps a per-lane array of services that lane restarts and
no other -- ``DEV_LANE_ONLY_RUNTIME_SERVICES`` and
``STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES``. ``resolve_lane_runtime_services``
appends that array to the restart set, but ONLY when no build override is in
play::

    # A scoped-build override is an explicit operator instruction to touch ONLY
    # the named services; silently appending to it would defeat the point.
    if [[ -n "${RUNTIME_BUILD_SERVICES_OVERRIDE:-}" ]]; then
        return 0
    fi

Both governed refresh scripts ALWAYS export that override. So for the two lanes
this repo refreshes, the lane-only array is unreachable and the refresh script's
own ``REFRESH_BUILD_SERVICES`` is the whole restart set. A service named in one
array and not the other is therefore never started by the governed path, no
matter how correctly it is declared.

Measured live on 2026-09-10. OMN-18114 added ``tenant-projection-writer`` to
both lane overlays, to the lane census manifest, and to both lane-only arrays
in ``deploy-runtime.sh`` -- and to neither refresh script. The stability-test
lane's own dry-run printed a ten-service build scope with the carrier absent,
while the lane census reported ``container_absent`` for it at severity
critical. The ticket was marked Done on the merge. Its AC4 asks that both lab
lanes RUN a carrier; neither did, and neither could.

Why the OMN-17448 gate did not catch it
---------------------------------------
``test_omn17448_dev_lane_writer_build_scope.py`` derives its expected set as
overlay-services MINUS base-services, on the reasoning that an overlay key also
present in the base is "an override, an env block, a label". That is true for
every service it was written against and false for this one:
``tenant-projection-writer`` IS declared in ``docker-compose.infra.yml``, gated
behind the inert ``profile-carrier-optin`` compose profile so that no lane
starts it by default. What makes it a real service on these two lanes is the
overlay's ``profiles: !override ["runtime", "full"]``. Subtracting the base
therefore subtracted precisely the service that needed covering.

So this file derives the obligation two independent ways and asserts both. The
first is the restart-set array. The second is compose itself: a service whose
LANE OVERLAY grants it ``runtime``/``full`` profile membership is a service that
lane starts, whether or not the base also declares it. Neither derivation is a
hand-maintained list, and the second holds even if the arrays in
``deploy-runtime.sh`` are wrong.

Related Tickets:
    - OMN-18114: the carrier, and the omission this gate closes
    - OMN-17448: the sibling gate whose base-subtraction blind spot let it through
    - OMN-17562: the six standalone writers, on both mutable lanes
    - OMN-14873: ``RUNTIME_BUILD_SERVICES_OVERRIDE``, the scoping mechanism
    - OMN-15243: why the stability-test lane in particular must not run degraded
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_RUNTIME = REPO_ROOT / "scripts" / "deploy-runtime.sh"
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"

# lane -> its overlay, its governed refresh script, and the deploy-runtime.sh
# array naming the services that lane restarts and no other. Only the two lanes
# this repo refreshes: prod and judge declare no lane-only services and have no
# refresh script here, and the collaborator lane is its owner's to deploy.
GOVERNED_LANES: dict[str, tuple[str, str, str]] = {
    "dev": (
        "docker-compose.dev-lane.yml",
        "refresh_dev_lane.sh",
        "DEV_LANE_ONLY_RUNTIME_SERVICES",
    ),
    "stability-test": (
        "docker-compose.stability-test.yml",
        "refresh_stability_lane.sh",
        "STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES",
    ),
}

# Top-level keys of the `services:` block. Scraped rather than parsed because
# the lane overlays carry compose's `!override` / `!reset` merge tags, which
# `yaml.safe_load` refuses -- the same reason the OMN-17448 gate and
# tests/unit/scripts/test_lane_census_manifest_parity.py scrape.
_SERVICE_KEY = re.compile(r"^  ([a-z0-9][a-z0-9._-]*):\s*$", re.MULTILINE)
_PROFILES_LINE = re.compile(r"^\s*profiles:.*$", re.MULTILINE)
# The compose profiles that mean "this lane's governed refresh starts it".
RUNTIME_PROFILE_NAMES = ("runtime", "full")


def _overlay_path(lane: str) -> Path:
    return REPO_ROOT / "docker" / GOVERNED_LANES[lane][0]


def _refresh_path(lane: str) -> Path:
    return REPO_ROOT / "scripts" / "runtime_build" / GOVERNED_LANES[lane][1]


def _service_blocks(path: Path) -> dict[str, str]:
    """Map each service name to its own slice of the `services:` block."""
    text = path.read_text(encoding="utf-8")
    start = text.index("\nservices:\n")
    body = text[start + len("\nservices:\n") :]
    end = re.search(r"^[a-z]", body, re.MULTILINE)
    if end is not None:
        body = body[: end.start()]
    keys = [(m.group(1), m.start()) for m in _SERVICE_KEY.finditer(body)]
    blocks: dict[str, str] = {}
    for index, (name, begins) in enumerate(keys):
        ends = keys[index + 1][1] if index + 1 < len(keys) else len(body)
        blocks[name] = body[begins:ends]
    return blocks


def _bash_array(script: str, name: str, source: Path) -> set[str]:
    """Read a `readonly NAME=( ... )` array's word members.

    Line-based and comment-stripping, deliberately, rather than the
    `readonly NAME=\\((?P<items>[^)]*)\\)` regex the OMN-17448 gate uses. That
    body is a negated character class excluding the closing round bracket, so a
    round bracket anywhere in the array's PROSE ends the match early and every
    entry after it silently stops being covered. It is not hypothetical: a
    single parenthesised aside once hid the last two entries of the dev refresh
    array, and the note asking future editors to avoid brackets is the scar.

    A reader that cannot be truncated needs no such note, so this one strips
    `#` comments and scans to the closing bracket instead of asking the prose
    to stay bracket-free.
    """
    opener = f"readonly {name}=("
    start = script.find(opener)
    assert start != -1, (
        f"{name} moved or changed shape in {source.name}; this drift check can "
        "no longer read it, so it would silently pass. Fix the reader in the "
        "same change that moves the array."
    )

    words: set[str] = set()
    remainder = script[start + len(opener) :]
    for raw_line in remainder.splitlines():
        line = raw_line.split("#", 1)[0]
        closed = ")" in line
        if closed:
            line = line[: line.index(")")]
        for word in line.split():
            # `"${CORE_SERVICES[@]}"` and friends are expansions, not members;
            # each caller substitutes the referenced array itself.
            if not word.startswith('"${'):
                words.add(word)
        if closed:
            return words
    raise AssertionError(
        f"{name} in {source.name} is never closed -- the reader ran off the "
        "end of the file, so its membership cannot be trusted."
    )


def _lane_only_restart_services(lane: str) -> set[str]:
    """The deploy-runtime.sh array of services only this lane restarts."""
    _, _, array_name = GOVERNED_LANES[lane]
    return _bash_array(
        DEPLOY_RUNTIME.read_text(encoding="utf-8"), array_name, DEPLOY_RUNTIME
    )


def _overlay_runtime_profile_services(lane: str) -> set[str]:
    """Services this lane's OVERLAY puts into the runtime/full compose profile.

    Derived from compose rather than from any array, and deliberately NOT
    subtracting the base: a service the base declares behind an inert profile
    is started by this lane and by no other precisely because the overlay
    rewrites its profile list. That subtraction is the OMN-17448 blind spot.
    """
    blocks = _service_blocks(_overlay_path(lane))
    return {
        name
        for name, body in blocks.items()
        if any(
            profile in line
            for line in _PROFILES_LINE.findall(body)
            for profile in RUNTIME_PROFILE_NAMES
        )
    }


def _refresh_build_scope(lane: str) -> set[str]:
    """The service list this lane's refresh hands to the build override."""
    refresh = _refresh_path(lane)
    script = refresh.read_text(encoding="utf-8")
    scope = _bash_array(script, "REFRESH_BUILD_SERVICES", refresh)
    # The array expands CORE_SERVICES by reference; substitute its members.
    scope |= _bash_array(script, "CORE_SERVICES", refresh)
    return scope


@pytest.mark.parametrize("lane", sorted(GOVERNED_LANES))
def test_every_lane_only_restart_service_is_in_the_refresh_build_scope(
    lane: str,
) -> None:
    """The two arrays must agree, or the governed refresh never starts it."""
    restart_set = _lane_only_restart_services(lane)
    assert restart_set, (
        f"the {lane} lane-only restart array is empty -- the fixture, not the "
        "script, is what changed."
    )

    missing = sorted(restart_set - _refresh_build_scope(lane))
    assert not missing, (
        f"{sorted(GOVERNED_LANES[lane][2:])} names {missing} for the {lane} "
        f"lane, but {GOVERNED_LANES[lane][1]}'s REFRESH_BUILD_SERVICES does "
        "not. The refresh always exports RUNTIME_BUILD_SERVICES_OVERRIDE, and "
        "resolve_lane_runtime_services returns early on an override without "
        "appending the lane-only array, so the governed path never starts "
        "them. Declared, censused, and never running -- OMN-18114."
    )


@pytest.mark.parametrize("lane", sorted(GOVERNED_LANES))
def test_every_overlay_runtime_profile_service_is_in_the_refresh_build_scope(
    lane: str,
) -> None:
    """Compose-derived, so it holds even if the deploy-runtime arrays are wrong.

    This is the derivation that catches a service the BASE declares behind an
    inert profile and the overlay promotes into `runtime`. Subtracting the base
    -- what the OMN-17448 gate does -- removes exactly that case.
    """
    promoted = _overlay_runtime_profile_services(lane)
    assert promoted, (
        f"the {lane} overlay puts no service into the "
        f"{RUNTIME_PROFILE_NAMES} profiles -- the fixture, not the script, is "
        "what changed."
    )

    missing = sorted(promoted - _refresh_build_scope(lane))
    assert not missing, (
        f"the {lane} overlay puts {missing} into the runtime compose profile, "
        f"so that lane starts them, but {GOVERNED_LANES[lane][1]}'s "
        "REFRESH_BUILD_SERVICES omits them and the refresh's build override "
        "means the restart set is exactly that array. Being declared in the "
        "BASE compose file is not a reason to leave a service out: the base "
        "may gate it behind an inert profile that this overlay overrides."
    )


@pytest.mark.parametrize("lane", sorted(GOVERNED_LANES))
def test_the_reader_survives_a_round_bracket_in_the_array_prose(lane: str) -> None:
    """Regression guard on this file's own reader, not on the scripts.

    The OMN-17448 gate's `[^)]*` body stops at the first round bracket in an
    array's comments, which silently shrinks its coverage to the entries before
    it. The dev refresh array carries a NOTE begging editors to keep brackets
    out of the prose for exactly that reason. This asserts the reader here does
    not need that promise: a bracketed aside injected into a copy of the real
    array must not hide the entry that follows it.
    """
    refresh = _refresh_path(lane)
    real = _bash_array(refresh.read_text(encoding="utf-8"), "CORE_SERVICES", refresh)
    assert real, f"CORE_SERVICES read empty for {lane}; the reader is broken."

    injected = (
        "readonly PROBE_SERVICES=(\n"
        "    first-service\n"
        "    # an aside with a round bracket (like this one) in the prose\n"
        "    service-after-the-bracket\n"
        ")\n"
    )
    probe = _bash_array(injected, "PROBE_SERVICES", refresh)
    assert probe == {"first-service", "service-after-the-bracket"}, (
        "the array reader lost an entry following a bracketed comment, which "
        "is the OMN-17448 truncation defect reproduced in this gate. A reader "
        f"that truncates reports a clean bill of health for {lane}."
    )


def test_the_two_derivations_do_not_silently_agree_by_being_empty() -> None:
    """Positive control: both derivations must actually resolve the carrier.

    An empty set satisfies a subset assertion trivially. This names the service
    OMN-18114 exists for and requires each derivation to see it on each lane, so
    a future refactor that quietly stops resolving anything fails here rather
    than reporting a clean bill of health.
    """
    carrier = "tenant-projection-writer"
    for lane in GOVERNED_LANES:
        assert carrier in _lane_only_restart_services(lane), (
            f"{carrier} is not in the {lane} lane-only restart array; either "
            "OMN-18114 was reverted or the array reader is broken."
        )
        assert carrier in _overlay_runtime_profile_services(lane), (
            f"the {lane} overlay no longer promotes {carrier} into the runtime "
            "compose profile; either OMN-18114 was reverted or the compose "
            "reader is broken."
        )
