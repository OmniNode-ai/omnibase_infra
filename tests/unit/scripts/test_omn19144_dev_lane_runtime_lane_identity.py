# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19144 -- a runtime that cannot name its lane emits an unkeyable verdict.

``ModelRuntimeHealthCheckEvent.lane`` is read from ``ONEX_RUNTIME_LANE`` at emit
time (:func:`omnibase_infra.runtime.health.runtime_lane_identity.resolve_runtime_lane`,
called at ``service_runtime_health_monitor.py`` where the event is built). The
lab lane-health projection keys its row on that value and DROPS an event that
carries ``None`` -- deliberately, because attributing an unnamed lane's verdict
to a lane by proximity is worse than holding no row.

OMN-18769 shipped the field, the resolver and the consumer. It shipped no
deployment that declares the variable, on any lane. So every runtime-health
event the dev lane has ever emitted carried ``lane: null``, the projection's
health arm returned before its upsert on every one of them, and the offsets
committed over it. Measured read-only on the ``.201`` dev lane 2026-09-22:
``docker inspect omninode-runtime`` returned 109 environment entries, 22 of them
``ONEX_`` prefixed, and ``ONEX_RUNTIME_LANE`` was not among them; the projection
row ``omninode_internal.lab_lane_health`` carried a current census fact and a
current receipt fact beside two NULL health columns.

Nothing raised, because nothing was wrong with the writer. The event simply had
no lane on it.

This module is the gate for that class. It asserts the DEPLOYMENT declares the
lane and that the declared spelling is one the emitter's own resolver accepts,
so a drift between the compose file and the lane's overlay scope -- a rename, a typo, an
anchor dropped from a merge list -- fails here rather than as another silent
total loss on the lab.

Every constant this compares against is spelled in THIS repository: the compose
overlays it parses and the core overlay scope model. Nothing is
read out of a sibling clone, so the module collects and runs on a CI runner
rather than skipping there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.models.config_overlay import ModelConfigOverlayScope
from omnibase_infra.runtime.health.runtime_lane_identity import ENV_RUNTIME_LANE

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
DEV_LANE_OVERLAY = DOCKER_DIR / "docker-compose.dev-lane.yml"

#: The lane this overlay IS. The census emitter spells it ``dev`` and the
#: receipt emitter spells it ``compose-dev``; the resolver's vocabulary is the
#: receipt emitter's, which is the join key the projection rows are keyed on.
DEV_LANE_VALUE = "compose-dev"

#: The one dev-lane service that speaks for the lane's health.
#:
#: ``omninode-runtime`` and ``runtime-effects`` BOTH run the health monitor and
#: both emit on the health topic -- verified read-only on the lab 2026-09-22,
#: each logging four monitor lines in the same 40-minute window. Only one of
#: them may claim the lane. The projection's health arm is an upsert guarded on
#: ``observed_at``, not a fold across processes, so two services declaring the
#: same lane would make the lane's stored verdict whichever process emitted
#: last rather than a statement about the lane.
#:
#: ``omninode-runtime`` is the one that speaks, because it is the lane's health
#: surface: port 8085 is the endpoint the lane's own acceptance falsifier reads.
#: ``runtime-effects`` keeps no lane and its health events stay unkeyable and
#: dropped -- an honest "this process cannot speak for the lane", not an
#: oversight. A worst-of fold across a lane's runtimes is a different design and
#: belongs to the parent ticket, not to this repair.
LANE_SPEAKING_SERVICE = "omninode-runtime"

#: Overlays for lanes the lab lane-health vocabulary deliberately excludes
#: (OMN-18769 AC6), mapped to the ONE lane each may declare, if any.
#:
#: Until OMN-19408 these had to declare nothing at all. That left the
#: stability-test runtime unable to say what it was, so a lab-only node could
#: not be kept off it, attached there, and held the lane unhealthy. A lane may
#: now name ITSELF -- a registered, non-lab lane -- because the health emitter
#: keys only a lab lane: the declaration controls what may attach, never which
#: lab row a verdict lands on. Naming a LAB lane from one of these overlays is
#: still refused, and so is naming another lane.
EXCLUDED_LANE_OVERLAYS = {
    DOCKER_DIR / "docker-compose.stability-test.yml": "stability-test",
    DOCKER_DIR / "docker-compose.judge.yml": "judge",
    DOCKER_DIR / "docker-compose.lakshman.yml": "lakshman",
}


class _TolerantLoader(yaml.SafeLoader):
    """compose uses `!override` / `!!merge`, which SafeLoader refuses."""


_TolerantLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "",
    lambda loader, suffix, node: (
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else (
            loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        )
    ),
)


def _service_environments(overlay: Path) -> dict[str, dict[str, str]]:
    """Map service name -> its declared environment mapping.

    Compose merge keys are resolved by the loader, so an anchor merged into a
    service's ``environment`` reads back as that service's own key. That is the
    point: the gate must see the value the container will get, not the literal
    lines under the service.
    """
    with open(overlay, encoding="utf-8") as handle:
        # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
        # multi-constructor for compose's tags, which resolve to plain
        # mappings/sequences/scalars. No arbitrary object can be built.
        parsed = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
    services = dict(parsed.get("services") or {})
    out: dict[str, dict[str, str]] = {}
    for name, body in services.items():
        if not isinstance(body, dict):
            continue
        env = body.get("environment")
        if isinstance(env, dict):
            out[name] = {str(k): str(v) for k, v in env.items()}
    return out


def _lane_declarations(overlay: Path) -> dict[str, str]:
    """Map service name -> declared lane, for services that declare one."""
    return {
        name: env[ENV_RUNTIME_LANE]
        for name, env in _service_environments(overlay).items()
        if ENV_RUNTIME_LANE in env
    }


def test_the_parse_sees_this_services_environment_at_all() -> None:
    """Positive control for the zero the next test reports.

    Every assertion below is shaped "the variable is absent". An absent variable
    and a parse that returned nothing look identical from the assertion's side,
    which is how a broken gate reads as a clean bill of health. This proves the
    loader resolves the merge anchors and hands back real keys for the very
    service under test.
    """
    environments = _service_environments(DEV_LANE_OVERLAY)

    assert LANE_SPEAKING_SERVICE in environments, (
        f"{LANE_SPEAKING_SERVICE} has no environment mapping in "
        f"{DEV_LANE_OVERLAY.name} -- the parse, not the lane, is what failed"
    )
    assert len(environments[LANE_SPEAKING_SERVICE]) > 1, (
        f"{LANE_SPEAKING_SERVICE} resolved to "
        f"{environments[LANE_SPEAKING_SERVICE]!r}; the compose merge anchors did "
        "not resolve, so every absence this module reports is a parse artifact"
    )


def test_dev_lane_declares_the_runtime_lane_on_its_health_emitting_service() -> None:
    """The fix at cause: the deployment states which lane it is."""
    declared = _lane_declarations(DEV_LANE_OVERLAY)

    assert LANE_SPEAKING_SERVICE in declared, (
        f"{DEV_LANE_OVERLAY.name} service {LANE_SPEAKING_SERVICE} declares no "
        f"{ENV_RUNTIME_LANE}, so every runtime-health event it emits carries "
        "lane=null. The lab lane-health projection drops an event it cannot "
        "key, returns before its upsert and commits the offset anyway, so "
        "consumer lag reads 0 over a total loss of the health arm (OMN-19144). "
        f"Merge *dev_lane_identity_env into that service's environment in "
        f"docker/{DEV_LANE_OVERLAY.name}."
    )
    assert declared[LANE_SPEAKING_SERVICE] == DEV_LANE_VALUE, (
        f"{LANE_SPEAKING_SERVICE} declares "
        f"{ENV_RUNTIME_LANE}={declared[LANE_SPEAKING_SERVICE]!r}, not "
        f"{DEV_LANE_VALUE!r}. The projection keys its row on this exact string."
    )


def test_the_declared_lane_is_a_lane_id_an_overlay_can_declare() -> None:
    """The deployment and its overlay must agree, not merely both exist.

    OMN-19747: the lane's roles come from its runtime.lane overlay document,
    stored at the scope segment this value names. A value that is not a scope
    segment could never have a document, so the runtime would refuse to start.
    """
    declared = _lane_declarations(DEV_LANE_OVERLAY)
    value = declared.get(LANE_SPEAKING_SERVICE)

    assert value is not None, (
        "no lane is declared at all -- see the sibling test in this module"
    )
    ModelConfigOverlayScope(environment="local", lane=value)
    assert value == DEV_LANE_VALUE


def test_exactly_one_dev_lane_service_speaks_for_the_lane() -> None:
    """Two claimants make the stored verdict a race, not a measurement.

    Both ``omninode-runtime`` and ``runtime-effects`` run the health monitor on
    this lane. The projection's health arm is an ``observed_at``-guarded upsert
    on a row keyed by lane alone, so a second service declaring the same lane
    does not add a fact -- it overwrites the first one whenever it emits later.
    Adding a claimant is therefore a modelling decision (a lane-level fold), not
    a configuration tweak, and it fails here until someone makes it.
    """
    declared = _lane_declarations(DEV_LANE_OVERLAY)

    assert sorted(declared) == [LANE_SPEAKING_SERVICE], (
        f"{sorted(declared)} declare {ENV_RUNTIME_LANE} on "
        f"{DEV_LANE_OVERLAY.name}; exactly one service may speak for a lane "
        "while the projection stores one row per lane with no cross-process "
        "fold. Whichever of these emits last silently becomes the lane's "
        "recorded health."
    )


@pytest.mark.parametrize(
    ("overlay", "own_lane"),
    sorted(EXCLUDED_LANE_OVERLAYS.items()),
    ids=lambda value: value.stem if isinstance(value, Path) else value,
)
def test_excluded_lanes_declare_only_their_own_non_lab_lane(
    overlay: Path, own_lane: str
) -> None:
    """A read-only surface may name itself, and may never publish a lab-keyed verdict."""
    assert overlay.exists(), (
        f"{overlay.name} is gone, so this arm of the gate proves nothing -- "
        "re-point it at the lane overlays that exist"
    )
    declared = _lane_declarations(overlay)

    for service, lane in declared.items():
        assert lane == own_lane, (
            f"{overlay.name} service {service} declares {ENV_RUNTIME_LANE}="
            f"{lane!r}; this overlay may declare only its own lane {own_lane!r}"
        )


def test_the_stability_test_main_runtime_names_its_lane() -> None:
    """OMN-19408: the lane a lab-only node is kept off must be able to say so.

    The auto-wiring ownership filter fails closed on a runtime that owns a
    lane-scoped contract and declares no lane -- so without this declaration
    the lab lane-health projection would stop attaching here only by turning
    the runtime DEGRADED with a discovery error.
    """
    overlay = DOCKER_DIR / "docker-compose.stability-test.yml"
    environments = _service_environments(overlay)
    assert len(environments.get(LANE_SPEAKING_SERVICE, {})) > 1, (
        "positive control: the stability-test main runtime's environment did "
        "not parse, so an absence below would be a parse artifact"
    )

    declared = _lane_declarations(overlay)

    assert declared == {LANE_SPEAKING_SERVICE: "stability-test"}, (
        f"{overlay.name} declares {declared!r}; the main runtime must declare "
        f"{ENV_RUNTIME_LANE}=stability-test (OMN-19408)"
    )
