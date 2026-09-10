# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane must run a process for every runtime profile a contract pins to [OMN-18114].

The defect this gate exists for, read live off the ``.201`` **dev** lane on
2026-09-10. ``omninode-runtime`` logs::

    Discovered contract: node_projection_delegation_inference_response (reducer)
    from omnimarket 0.4.47

and then nothing subscribes it. Its consumer group::

    local.omnimarket.node_projection_delegation_inference_response.consume.1.0.0
      .__i.runtime-main.__t.onex.evt.omnibase-infra.inference-response.v1

    STATE Empty   MEMBERS 0   CURRENT-OFFSET 644   LOG-END-OFFSET 840   LAG 196

The last commit on that group was 2026-09-04T09:46Z; the lag was 126 on 09-06 and
143 on 09-08, so it grows monotonically and no process is draining it.

Why, mechanically
-----------------
The contract declares ``runtime_profiles: [tenant-projection]``.
``filter_manifest_for_runtime_profile``
(``omnibase_infra/runtime/auto_wiring/profile_ownership.py``) admits a contract to
a profile only when that profile is named in the contract's own list, so ``main``
and ``effects`` both SKIP it. Run inside the live container::

    total_discovered 499
    main owned 301
    effects owned 177
    tenant-projection owned 8
    main skipped_target True   effects skipped_target True

**No service on any .201 compose lane binds** ``RUNTIME_PROFILE=tenant-projection``.
The only carrier that exists anywhere is an ``onex-dev`` k8s Deployment. So all
eight contracts are discovered, filtered off every process that runs, and consumed
by nothing — permanently, and with no error anywhere, because from each process's
own point of view the manifest simply does not contain them.

Why no existing gate saw it
---------------------------
``test_lane_projection_writer_coverage_omn17562`` measures the adjacent class:
contracts the kernel SUBSCRIBES and never dispatches. A contract removed from the
manifest by the profile filter is never subscribed at all, so it cannot appear in
that gate's ``WRITER_OWNED_PROJECTIONS`` registry — and it does not. OMN-17641,
which moved the eight onto the profile, scoped compose out in as many words ("no
compose-lane change, the .201 lab lanes are OMN-17562's"). OMN-17562 scoped
profile-filtered contracts out implicitly. The class fell between the two.

This gate is the missing half: *given that a contract names a profile as its sole
owner, does the lane run a process that carries that profile?*

What is enforced
----------------
1. **Registration** — every profile this gate requires a carrier for resolves
   through the real ``load_runtime_profile`` and is consumer-attached in core.
   Positive/negative controlled, so a container that admits anything cannot read
   as a real membership answer.
2. **Carriage on both mutable lab lanes** — ``dev`` and ``stability-test`` each
   START a service binding that profile literally. "Started" is the merged
   compose's own ``profiles:`` list, so a service that exists but is inert does
   not count as carriage.
3. **Shrink-only ratchet** — each lane's count of uncarried profiles is pinned
   and may fall, never rise. ``lakshman`` is measured and not deployed here.
4. **No unregistered profile on a governed lane** — a literal ``RUNTIME_PROFILE``
   in a lane's merged compose must resolve. This is OMN-17985's k8s finding
   applied at the compose layer, where it was never applied.
5. **Blast radius** — the carrier is declared in the BASE compose (that is where
   the runtime env anchors live, and re-declaring 340 lines of them in an overlay
   would be the drift this repo keeps paying for), so ``prod``, ``judge`` and
   ``lakshman`` all RENDER it. They must not START it. The lab overlays opt in by
   overriding ``profiles:``; every other lane leaves it inert.

Deliberately NOT enforced here: that a profile has a carrier on ``onex-dev``. That
is ``omninode_infra``'s manifest set (it already carries this one) and is governed
by OMN-17556, not by a file in this repo. The honest limit is the same one
OMN-17456 records: no single repo's CI can see contracts, k8s manifests and
compose at once.

Related tickets:
    - OMN-18114: this gate and the dev/stability carrier.
    - OMN-17641: moved the eight contracts onto ``tenant-projection``.
    - OMN-17556: ratified that move as a permanent ownership boundary.
    - OMN-17562: the adjacent writer-coverage gate this one sits beside.
    - OMN-17985: unregistered ``RUNTIME_PROFILE`` values, k8s side.
    - OMN-17298: what the inference-response projection cannot yet WRITE once it
      is carried. Carriage is this ticket; the row is that one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from omnibase_infra.runtime.runtime_profile import load_runtime_profile
from tests.ci.test_lane_projection_writer_coverage_omn17562 import (
    BASE_COMPOSE,
    DOCKER_DIR,
    GOVERNED_LANES,
    _environment,
    _services,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The compose service profile that keeps the carrier INERT. It is deliberately
# not "runtime" and not "full": every lane's bring-up names those two, so a
# service in either starts everywhere the base is merged — including prod and
# judge, which this change must not touch. A lab lane opts in by OVERRIDING the
# service's `profiles:` in its own overlay, which is a visible, per-lane decision
# rather than a default that leaks.
INERT_COMPOSE_PROFILE = "profile-carrier-optin"

# Compose profiles that a lane's runtime bring-up actually requests. Taken from
# the base file's own runtime services, which every lane starts with
# `--profile runtime`.
STARTED_COMPOSE_PROFILES: frozenset[str] = frozenset({"runtime", "full"})

# Lanes this repo both measures AND deploys. `lakshman` is in GOVERNED_LANES and
# is measured below, but its ratchet moves when its owner deploys, not here
# (OMN-17150).
MUTABLE_LAB_LANES: tuple[str, ...] = ("dev", "stability-test")


class CarrierProfile(NamedTuple):
    """A runtime profile that some contract names as its sole owner."""

    # The omnimarket contracts pinned to this profile. Derived, not chosen: see
    # `test_carrier_profile_contracts_match_the_omnimarket_manifest`, which
    # re-runs the real discovery + ownership filter and asserts this tuple.
    contracts: tuple[str, ...]
    reason: str


# Profiles that a contract pins itself to and that therefore REQUIRE a carrying
# process wherever the platform runs.
#
# Derived on 2026-09-10 by running the real predicate against the live dev-lane
# image (`discover_contracts()` -> `filter_manifest_for_runtime_profile(m, p)`)
# and keeping every profile with a non-empty owned set that is not carried by one
# of the shared kernels. `main`, `effects`, `workers` and `projection-api` are
# excluded because the base compose already starts a process for each on every
# lane; the seven `projection-writer-*` profiles are excluded because NO contract
# declares them — they are carried by `python -m <runner>` services and are
# OMN-17562's subject, not this gate's.
CARRIER_PROFILES: dict[str, CarrierProfile] = {
    "tenant-projection": CarrierProfile(
        contracts=(
            "canary_score_reducer",
            "node_hook_event_capture",
            "node_projection_cost_summary",
            "node_projection_delegation_inference_response",
            "node_projection_dep_health",
            "projection_context_roi",
            "projection_pattern_learning",
            "projection_routing_decision",
        ),
        reason=(
            "OMN-17641/OMN-17556: the eight TENANT-domain projections that resolve "
            "the `tenant_projection` topology binding. One process owns the "
            "binding, so one process owns the contracts."
        ),
    ),
}

# Carrier profiles with NO carrying service on the lane, pinned at the value
# measured on 2026-09-10. Shrink-only: a change that lands a carrier lowers the
# number here in the same edit; one that removes a carrier fails.
#
# `lakshman` stays at 1: this repo measures that lane so it cannot silently
# diverge, but it does not deploy it.
LANE_UNCARRIED_PROFILE_RATCHET: dict[str, int] = {
    "dev": 0,
    "stability-test": 0,
    "lakshman": 1,
}

_OMNIMARKET_ABSENT_REASON = (
    "omnimarket is not importable here. It is deliberately absent from this "
    "repo's canonical venv (the OMN-15620 purity gate rejects an undeclared "
    "`onex.nodes` provider), so the contract fidelity cross-check cannot run in "
    "that job. The manifest-only gates above are unaffected and still run."
)

_UNREGISTERED_CONTROL_PROFILE = "omn18114-not-a-real-profile"


def _merged_services(lane: str) -> dict[str, dict[str, Any]]:
    """Base + overlay, merged the way `docker compose -f a -f b` merges them.

    Only the keys this gate reads are merged, and each on its own compose rule:
    `environment` is a MAPPING, so the overlay's entries win key by key and the
    base's survive; `profiles` is a SEQUENCE that the overlays override wholesale
    with `!override`, so the overlay's list REPLACES the base's when present.
    Spelling both rules out beats a generic deep-merge that would quietly get one
    of them backwards.
    """
    base = _services(DOCKER_DIR / BASE_COMPOSE)
    overlay = _services(DOCKER_DIR / GOVERNED_LANES[lane].overlay)

    merged: dict[str, dict[str, Any]] = {}
    for name in set(base) | set(overlay):
        base_body = base.get(name, {})
        overlay_body = overlay.get(name, {})
        body: dict[str, Any] = {**base_body, **overlay_body}
        body["environment"] = {
            **_environment(base_body),
            **_environment(overlay_body),
        }
        if "profiles" not in overlay_body and "profiles" in base_body:
            body["profiles"] = base_body["profiles"]
        merged[name] = body
    return merged


def _compose_profiles(body: dict[str, Any]) -> frozenset[str]:
    raw = body.get("profiles")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(item) for item in raw)


def _is_started(body: dict[str, Any]) -> bool:
    """Whether a lane's runtime bring-up starts this service.

    A service with no `profiles:` key at all is started unconditionally by
    compose. A service that names profiles is started only when the bring-up
    requests one of them, and every lane requests exactly `runtime` (or `full`).
    """
    profiles = _compose_profiles(body)
    if not profiles:
        return True
    return bool(profiles & STARTED_COMPOSE_PROFILES)


def _literal_runtime_profile(body: dict[str, Any]) -> str | None:
    """A service's literal `RUNTIME_PROFILE`, or None if absent/interpolated.

    An interpolated value (`${...}`) is not a claim this gate can check
    statically, and treating one as a carrier would let a lane satisfy the
    invariant with a variable that resolves to something else at deploy time.
    """
    value = body.get("environment", {}).get("RUNTIME_PROFILE")
    if not isinstance(value, str) or "${" in value:
        return None
    stripped = value.strip()
    return stripped or None


def _carried_profiles_on_lane(lane: str) -> dict[str, str]:
    """Map ``runtime profile -> service name`` for the services a lane STARTS."""
    carried: dict[str, str] = {}
    for service, body in _merged_services(lane).items():
        if not _is_started(body):
            continue
        profile = _literal_runtime_profile(body)
        if profile is not None:
            carried.setdefault(profile, service)
    return carried


def _uncarried_profiles_on_lane(lane: str) -> list[str]:
    carried = set(_carried_profiles_on_lane(lane))
    return sorted(set(CARRIER_PROFILES) - carried)


def test_every_carrier_profile_is_registered_and_consumer_attached() -> None:
    """A profile this gate demands a carrier for must actually resolve.

    Otherwise the gate would happily require a container for a name the runtime
    refuses at boot — which is OMN-17985's defect wearing this gate's clothes.
    """
    for name in sorted(CARRIER_PROFILES):
        profile = load_runtime_profile(name)
        assert profile.name == name, (
            f"load_runtime_profile({name!r}) resolved to {profile.name!r}. A "
            "carrier profile must resolve to itself; a silent fallback means the "
            "deployed process would carry a different role than the contracts "
            "that name it."
        )

    # Negative control. Without it, three passing resolutions above prove nothing
    # about the registry — a container that admits anything would look identical.
    with pytest.raises(Exception):
        load_runtime_profile(_UNREGISTERED_CONTROL_PROFILE)


def test_carrier_profiles_are_carried_on_every_mutable_lab_lane() -> None:
    """dev and stability-test must each START a process for every carrier profile.

    This is the assertion the ticket exists for. Before the carrier service
    lands, `tenant-projection` is uncarried on both lanes and this fails naming
    them.
    """
    missing: list[str] = []
    for lane in MUTABLE_LAB_LANES:
        carried = _carried_profiles_on_lane(lane)
        for name, spec in sorted(CARRIER_PROFILES.items()):
            if name not in carried:
                missing.append(
                    f"{lane}: no started service binds RUNTIME_PROFILE={name} "
                    f"({spec.reason}). {len(spec.contracts)} contract(s) name it "
                    f"as their sole owner and are consumed by nothing on this "
                    f"lane: {', '.join(spec.contracts)}"
                )

    assert not missing, (
        "Contract-pinned runtime profile(s) have no carrying process on a lab "
        "lane. Every contract that names the profile is discovered, dropped by "
        "filter_manifest_for_runtime_profile from main and effects, and "
        "subscribed by nothing — silently, with no error on any process, because "
        "the manifest each process sees simply does not contain them.\n  "
        + "\n  ".join(missing)
        + "\nAdd a service binding that RUNTIME_PROFILE to the lane's compose "
        "(the base declares the carrier; the lane overlay opts it into the "
        "`runtime` compose profile), or move the contracts off the profile. Do "
        "NOT hand-add a subscription anywhere: carriage is a deployment fact."
    )


def test_lane_uncarried_profile_count_is_shrink_only() -> None:
    """Each lane's uncarried-profile count is pinned and may only fall."""
    for lane in sorted(GOVERNED_LANES):
        assert lane in LANE_UNCARRIED_PROFILE_RATCHET, (
            f"Lane {lane!r} is governed but has no LANE_UNCARRIED_PROFILE_RATCHET "
            "entry. Measure it and pin the measured value; do not guess."
        )
        uncarried = _uncarried_profiles_on_lane(lane)
        pinned = LANE_UNCARRIED_PROFILE_RATCHET[lane]
        assert len(uncarried) <= pinned, (
            f"Lane {lane!r} now has {len(uncarried)} uncarried carrier "
            f"profile(s) {uncarried}, above its pinned {pinned}. A carrier was "
            "removed, or a contract pinned a profile no lane runs. Restore the "
            "carrier rather than raising the pin."
        )
        assert len(uncarried) == pinned, (
            f"Lane {lane!r} has {len(uncarried)} uncarried carrier profile(s) "
            f"{uncarried} but the ratchet still pins {pinned}. A carrier landed "
            "and the pin was not lowered in the same change — lower it here so "
            "the gain cannot be silently given back."
        )


def test_no_governed_lane_starts_a_service_naming_an_unregistered_profile() -> None:
    """A literal RUNTIME_PROFILE on a governed lane must resolve.

    OMN-17985 closed exactly this on the k8s side, where seven deployed values
    were registered nowhere. Nothing applied it to compose, where the same
    unresolvable name empties the manifest and wires zero subscriptions while the
    process still passes readiness.
    """
    unregistered: list[str] = []
    for lane in sorted(GOVERNED_LANES):
        for service, body in sorted(_merged_services(lane).items()):
            if not _is_started(body):
                continue
            profile = _literal_runtime_profile(body)
            if profile is None:
                continue
            try:
                load_runtime_profile(profile)
            except Exception as exc:  # noqa: BLE001 - the raise type is core's
                unregistered.append(f"{lane}/{service}: {profile!r} ({exc})")

    assert not unregistered, (
        "Started service(s) bind a RUNTIME_PROFILE the runtime cannot resolve: "
        f"{unregistered}. An unresolvable role is not a milder role — the "
        "ownership filter matches no contract, the manifest empties, and the "
        "process boots green with zero subscriptions."
    )


def test_the_carrier_is_inert_on_every_lane_this_repo_does_not_deploy() -> None:
    """prod, judge and lakshman render the carrier and must never start it.

    The carrier lives in the base compose because that is where the runtime env
    anchors are, and every lane merges the base. Rendering it there is fine;
    starting it there is a deploy this ticket does not authorize and a second
    consumer group on lanes that never asked for one.
    """
    base = _services(DOCKER_DIR / BASE_COMPOSE)
    carriers = {
        name: body
        for name, body in base.items()
        if _literal_runtime_profile({"environment": _environment(body)})
        in CARRIER_PROFILES
    }
    assert carriers, (
        "No service in the base compose binds a carrier RUNTIME_PROFILE. The "
        "carrier must be declared in the base so it inherits the runtime env "
        "anchors; re-declaring those in an overlay is the drift this repo keeps "
        "paying for."
    )

    for name, body in sorted(carriers.items()):
        profiles = _compose_profiles(body)
        assert profiles == frozenset({INERT_COMPOSE_PROFILE}), (
            f"Base service {name!r} declares compose profiles {sorted(profiles)}. "
            f"It must declare exactly [{INERT_COMPOSE_PROFILE!r}] so that prod, "
            "judge and lakshman render it and never start it. A lab lane opts in "
            "by overriding `profiles:` in its own overlay."
        )

    for lane in ("lakshman",):
        started = set(_carried_profiles_on_lane(lane))
        assert not (started & set(CARRIER_PROFILES)), (
            f"Lane {lane!r} starts a carrier profile this repo does not deploy "
            f"for it: {sorted(started & set(CARRIER_PROFILES))}."
        )


@pytest.mark.skipif(
    importlib.util.find_spec("omnimarket") is None,
    reason=_OMNIMARKET_ABSENT_REASON,
)
def test_carrier_profile_contracts_match_the_omnimarket_manifest() -> None:
    """Re-derive each carrier profile's contract set through the real predicate.

    The registry above is derived, not chosen, and this is what keeps it that
    way: it runs the same `discover_contracts` -> `filter_manifest_for_runtime_profile`
    path the kernel runs, so the registry cannot drift from the ownership filter
    it describes.
    """
    from omnibase_infra.runtime.auto_wiring import (
        discover_contracts,
        filter_manifest_for_runtime_profile,
    )

    manifest = discover_contracts()
    assert manifest.contracts, (
        "Contract discovery returned an empty manifest. A zero here would make "
        "every comparison below trivially pass, so it is a failure, not a skip."
    )

    for name, spec in sorted(CARRIER_PROFILES.items()):
        owned = tuple(
            sorted(
                contract.name
                for contract in filter_manifest_for_runtime_profile(
                    manifest, name
                ).manifest.contracts
            )
        )
        assert owned == spec.contracts, (
            f"Profile {name!r} owns {owned} in the live manifest but "
            f"CARRIER_PROFILES records {spec.contracts}. Amend the registry in "
            "the same change that moves a contract, so the carrier's workload is "
            "always the set the ownership filter actually hands it."
        )
