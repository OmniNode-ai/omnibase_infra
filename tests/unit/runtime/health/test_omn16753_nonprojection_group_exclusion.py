# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A non-projection consumer's flow is not a projection's to account for (OMN-16753).

RED-first reproduction of stability refresh attempt 3 of 3, receipt
``~/.omnibase/state/stability_lane_refresh/history/20260908T235241Z-915a10446a8d.json``
on the ``.201`` host, at omnibase_infra dev ``915a10446`` (#3334, merged
2026-09-08T23:47Z). Every other leg of that gate passed --
``digest_changed=true``, ``manifest_ok=true`` (301 contracts over a floor of
288, the shared typed retry budget having recovered the effects fetch after
three ``Connection reset by peer`` attempts), ``revision_match=true`` on all
four core services, cluster healthy, consumer-group coverage 531/536 = 99.07%,
partition headroom 47.9%, and ``dlq_saturated_projections`` EMPTY. The gate
failed on one string::

    projection_dlq_saturation: No projection is fully DLQ-routed over 8 flow
    window(s) (2 topic(s) carried flow no single declaring projection could be
    attributed, and are excluded from every ratio:
    onex.evt.omniclaude.session-ended.v1,
    onex.evt.omniclaude.session-started.v1)

The lane was rolled back onto the previous image and ``18085`` returned 503.

The cause, read off the stability broker (``rpk group list``, read-only,
2026-09-09; positive control 688 groups total, negative control 0 on a nonsense
infix): **twelve** groups carry those two topics. Eleven are projection groups
(``projection_hook_ledger``, ``projection_session_replay`` at both
``consume.1.0.0`` and ``consume.1.1.0``, ``projection_work_events``). The
twelfth pair is::

    stability-test.omnimarket.node_session_phase_reducer.consume.1.1.0
        .__i.stability-test-main.__t.onex.evt.omniclaude.session-started.v1
    ... .__t.onex.evt.omniclaude.session-ended.v1

``node_session_phase_reducer`` is a REDUCER: ``node_type: reducer``, it
subscribes to both topics, and it declares no ``db_io.db_tables`` at all
(verified against
``omnimarket/src/omnimarket/nodes/node_session_phase_reducer/contract.yaml``;
positive control -- the same grep finds ``db_tables`` in the hook-ledger
projection contract). So ``select_projection_contracts`` never scopes it,
``select_projection_group_suffixes`` derives no infix for it, and
``_attribute_delta`` fell through: no projection infix matched, the topic has
three in-scope declarers so the sole-declarer fallback was correctly guarded,
and the delta was recorded as UNATTRIBUTABLE -- which #3334 had just made
degrade the dimension.

Both halves of that were right on their own terms and wrong together. Flow a
non-projection consumer took was never a projection's to take, and says
nothing about projection liveness. ``unattributable`` stays reserved for the
genuine cannot-tell case: flow a PROJECTION group took that cannot be pinned to
one declaring projection.

The exclusion is recorded on the verdict and rendered on the dimension detail
with the owning contract and the reason, because an exclusion nobody can see is
the same false all-clear this ticket exists to remove, wearing a third costume.

Related Tickets:
    - OMN-16753: this ticket
    - OMN-16994: the dimension being corrected
    - OMN-15837: the stability refresh gate that rolled the lane back
    - OMN-17562: the "a topic is not an attribution" finding this extends
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.models.observability import (
    ModelConsumerFlowDelta,
    ModelNodeFlowWindow,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.projection_liveness import (
    describe_dlq_saturation,
    dlq_saturation_status,
    evaluate_projection_liveness,
    select_nonprojection_group_infixes,
    select_projection_contracts,
    select_projection_group_suffixes,
)

SESSION_STARTED = "onex.evt.omniclaude.session-started.v1"
SESSION_ENDED = "onex.evt.omniclaude.session-ended.v1"
BOTH_TOPICS = (SESSION_STARTED, SESSION_ENDED)

# The three in-scope projections that co-declare the two topics on the lane.
HOOK_LEDGER = "projection_hook_ledger"
SESSION_REPLAY = "projection_session_replay"
WORK_EVENTS = "projection_work_events"

# The twelfth group's owner: a reducer, not a projection.
PHASE_REDUCER = "node_session_phase_reducer"


def _table(name: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema="omninode_internal",
        migration="0001_init.sql",
        access="read_write",
        role="projection_target",
    )


def _projection(
    name: str,
    *,
    topics: tuple[str, ...] = BOTH_TOPICS,
    package_name: str = "omnimarket",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name=package_name,
        event_bus=ModelEventBusWiring(subscribe_topics=topics, publish_topics=()),
        db_io=ModelDbOwnershipSubcontract(db_tables=[_table("projection_target")]),
    )


def _nonprojection(
    name: str,
    *,
    topics: tuple[str, ...] = BOTH_TOPICS,
    package_name: str = "omnimarket",
) -> ModelDiscoveredContract:
    """A contract that subscribes but declares no projection target table.

    The single discriminator ``select_projection_contracts`` admits on, absent:
    that is what makes this a reducer/effect/forwarder rather than a
    projection, and it is derived from the contract, never from a name list.
    """
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=1, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name=package_name,
        event_bus=ModelEventBusWiring(subscribe_topics=topics, publish_topics=()),
    )


def _manifest(*contracts: ModelDiscoveredContract) -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(contracts=tuple(contracts), errors=())


def _group(
    name: str,
    topic: str,
    *,
    env: str = "stability-test",
    version: str = "1.1.0",
    instance: str = "stability-test-main",
) -> str:
    """The full live group name the broker prints, suffixes included."""
    return f"{env}.omnimarket.{name}.consume.{version}.__i.{instance}.__t.{topic}"


def _window(
    deltas: tuple[tuple[str, str, int, int], ...],
    *,
    sequence: int = 1,
) -> ModelNodeFlowWindow:
    """One closed window. Each delta is (consumer_group, topic, in, dlq)."""
    node_id = uuid4()
    start = datetime(2026, 9, 8, 23, 52, tzinfo=UTC) + timedelta(seconds=30 * sequence)
    end = start + timedelta(seconds=30)
    return ModelNodeFlowWindow(
        node_id=node_id,
        window_start=start,
        window_end=end,
        window_sequence=sequence,
        consumer_deltas=tuple(
            ModelConsumerFlowDelta(
                consumer_group=group,
                topic=topic,
                node_id=node_id,
                window_start=start,
                window_end=end,
                window_sequence=sequence,
                messages_in=messages_in,
                messages_out=0,
                messages_dlq=messages_dlq,
                handler_errors=messages_dlq,
            )
            for group, topic, messages_in, messages_dlq in deltas
        ),
    )


def _attempt_three_manifest() -> ModelAutoWiringManifest:
    """The live shape: three co-declaring projections plus the reducer."""
    return _manifest(
        _projection(HOOK_LEDGER),
        _projection(SESSION_REPLAY),
        _projection(WORK_EVENTS),
        _nonprojection(PHASE_REDUCER),
    )


# Flow on all four groups over both topics, the way the lane actually ran:
# every projection consuming cleanly, and the reducer consuming too.
_ATTEMPT_THREE_DELTAS = tuple(
    (
        _group(name, topic, version="1.0.0" if name != PHASE_REDUCER else "1.1.0"),
        topic,
        240,
        0,
    )
    for name in (HOOK_LEDGER, SESSION_REPLAY, WORK_EVENTS, PHASE_REDUCER)
    for topic in BOTH_TOPICS
)

# The minimal cause, isolated: ONLY the reducer took flow. This is the shape
# that has to read HEALTHY -- nothing about a projection is in evidence at all.
_REDUCER_ONLY_DELTAS = tuple(
    (_group(PHASE_REDUCER, topic), topic, 240, 0) for topic in BOTH_TOPICS
)


def _evaluate(
    manifest: ModelAutoWiringManifest,
    deltas: tuple[tuple[str, str, int, int], ...],
):
    projections = select_projection_contracts(manifest)
    suffixes = select_projection_group_suffixes(manifest, projections)
    return evaluate_projection_liveness(
        projections=projections,
        attached_topics=frozenset(BOTH_TOPICS),
        flow_windows=[_window(deltas)],
        projection_group_suffixes=suffixes,
        nonprojection_group_infixes=select_nonprojection_group_infixes(
            manifest, suffixes
        ),
    )


@pytest.mark.unit
class TestReducerFlowIsExcludedNotUnattributable:
    """The attempt-3 shape. Today: DEGRADED. Fixed: HEALTHY, and visibly so."""

    def test_the_attempt_three_shape_is_healthy(self) -> None:
        verdict = _evaluate(_attempt_three_manifest(), _ATTEMPT_THREE_DELTAS)
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == ()
        assert dlq_saturation_status(verdict) == "HEALTHY"

    def test_reducer_only_flow_does_not_degrade_the_lane(self) -> None:
        """The isolated cause: no projection consumed, so nothing is proven.

        Pre-fix this is the exact receipt string -- two topics carried flow no
        single declaring projection could be attributed -- over traffic no
        projection ever touched.
        """
        verdict = _evaluate(_attempt_three_manifest(), _REDUCER_ONLY_DELTAS)
        assert verdict.unattributable_flow_topics == ()
        assert dlq_saturation_status(verdict) == "HEALTHY"

    def test_the_excluded_group_is_named_with_its_reason(self) -> None:
        """Never silent. An invisible exclusion is a false all-clear too."""
        verdict = _evaluate(_attempt_three_manifest(), _ATTEMPT_THREE_DELTAS)
        excluded = verdict.excluded_nonprojection_flow
        assert {entry.owner_contract for entry in excluded} == {PHASE_REDUCER}
        assert {entry.consumer_group for entry in excluded} == {
            _group(PHASE_REDUCER, topic) for topic in BOTH_TOPICS
        }
        for entry in excluded:
            assert entry.outcome == "EXCLUDED"
            assert entry.projection is None
            assert entry.reason

    def test_the_dimension_detail_names_the_excluded_group(self) -> None:
        verdict = _evaluate(_attempt_three_manifest(), _ATTEMPT_THREE_DELTAS)
        detail = describe_dlq_saturation(verdict)
        assert PHASE_REDUCER in detail
        assert "not a declared projection" in detail

    def test_the_reducers_flow_never_reaches_a_projection_ratio(self) -> None:
        """A reducer DLQing everything is not a projection's saturation.

        The reducer takes 2000 events and quarantines all of them while the
        three projections are clean. If its counters leaked into any
        projection's ratio, that projection would be named as 100% DLQ-routed.
        """
        deltas = tuple(
            (_group(name, topic, version="1.0.0"), topic, 240, 0)
            for name in (HOOK_LEDGER, SESSION_REPLAY, WORK_EVENTS)
            for topic in BOTH_TOPICS
        ) + tuple(
            (_group(PHASE_REDUCER, topic), topic, 1000, 1000) for topic in BOTH_TOPICS
        )
        verdict = _evaluate(_attempt_three_manifest(), deltas)
        assert verdict.dlq_saturated_projections == ()
        assert dlq_saturation_status(verdict) == "HEALTHY"


@pytest.mark.unit
class TestTheGenuineCannotTellCaseIsUntouched:
    """Controls. The round-2 rule and the measured signal both stay."""

    def test_an_unrecognised_group_on_a_shared_topic_still_degrades(self) -> None:
        """A group with NO derivable owner is still the cannot-tell case.

        It is not shown to be a non-projection consumer, so it is not excluded:
        exclusion requires positive proof, and its absence fails closed to
        DEGRADED exactly as before.
        """
        verdict = _evaluate(
            _attempt_three_manifest(),
            (("an-unrecognised-group", SESSION_STARTED, 1855, 1855),),
        )
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == (SESSION_STARTED,)
        assert dlq_saturation_status(verdict) == "DEGRADED"

    def test_an_ambiguous_two_projection_match_still_degrades(self) -> None:
        """Round 2's rule, intact: two matching infixes attribute to neither."""
        manifest = _attempt_three_manifest()
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        ambiguous = "".join(suffixes)
        verdict = _evaluate(manifest, ((ambiguous, SESSION_STARTED, 900, 900),))
        assert verdict.unattributable_flow_topics == (SESSION_STARTED,)
        assert dlq_saturation_status(verdict) == "DEGRADED"

    def test_a_genuinely_saturated_projection_still_degrades(self) -> None:
        """The measured failure the dimension exists for, beside the reducer."""
        deltas = (
            (
                _group(HOOK_LEDGER, SESSION_STARTED, version="1.0.0"),
                SESSION_STARTED,
                1855,
                1855,
            ),
            (_group(PHASE_REDUCER, SESSION_STARTED), SESSION_STARTED, 240, 0),
        )
        verdict = _evaluate(_attempt_three_manifest(), deltas)
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)
        assert dlq_saturation_status(verdict) == "DEGRADED"
        assert HOOK_LEDGER in describe_dlq_saturation(verdict)

    def test_the_sole_declarer_fallback_still_attributes(self) -> None:
        """A projection group with no infix match on a single-declarer topic.

        The fallback is what makes a one-declarer topic attributable at all,
        and the exclusion must not swallow it: the group carries no
        non-projection infix, so nothing about it is excluded.
        """
        manifest = _manifest(
            _projection(HOOK_LEDGER, topics=(SESSION_STARTED,)),
            _nonprojection(PHASE_REDUCER, topics=(SESSION_ENDED,)),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset(BOTH_TOPICS),
            flow_windows=[
                _window((("some-unrecognised-group", SESSION_STARTED, 500, 500),))
            ],
            projection_group_suffixes=suffixes,
            nonprojection_group_infixes=select_nonprojection_group_infixes(
                manifest, suffixes
            ),
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)
        assert dlq_saturation_status(verdict) == "DEGRADED"

    def test_a_reducer_group_does_not_take_the_sole_declarer_fallback(self) -> None:
        """The exclusion has to be decided BEFORE the fallback, not after.

        On a topic exactly one projection declares, the pre-fix fallback would
        hand the reducer's counters to that projection -- the misattribution
        class this whole ticket removes, in its most damaging form: a reducer
        DLQing everything would name a healthy projection as 100% DLQ-routed.
        """
        manifest = _manifest(
            _projection(HOOK_LEDGER, topics=(SESSION_STARTED,)),
            _nonprojection(PHASE_REDUCER, topics=(SESSION_STARTED,)),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({SESSION_STARTED}),
            flow_windows=[
                _window(
                    (
                        (
                            _group(PHASE_REDUCER, SESSION_STARTED),
                            SESSION_STARTED,
                            900,
                            900,
                        ),
                    )
                )
            ],
            projection_group_suffixes=suffixes,
            nonprojection_group_infixes=select_nonprojection_group_infixes(
                manifest, suffixes
            ),
        )
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == ()
        assert dlq_saturation_status(verdict) == "HEALTHY"
        assert {e.owner_contract for e in verdict.excluded_nonprojection_flow} == {
            PHASE_REDUCER
        }


@pytest.mark.unit
class TestSelectNonprojectionGroupInfixes:
    """The exclusion map is contract-derived and refuses an ambiguous entry."""

    def test_a_projection_is_not_in_the_map(self) -> None:
        manifest = _attempt_three_manifest()
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        infixes = select_nonprojection_group_infixes(manifest, suffixes)
        assert set(infixes.values()) == {PHASE_REDUCER}
        assert infixes == {f".omnimarket.{PHASE_REDUCER}.consume.": PHASE_REDUCER}

    def test_a_contract_with_no_subscription_is_not_in_the_map(self) -> None:
        """Nothing it could ever produce a consumer-flow delta under."""
        publisher = ModelDiscoveredContract(
            name="node_publisher_only",
            node_type="EFFECT",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=__file__,
            entry_point_name="node_publisher_only",
            package_name="omnimarket",
            event_bus=ModelEventBusWiring(
                subscribe_topics=(), publish_topics=(SESSION_STARTED,)
            ),
        )
        manifest = _manifest(_projection(HOOK_LEDGER), publisher)
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert select_nonprojection_group_infixes(manifest, suffixes) == {}

    def test_two_contracts_deriving_one_infix_are_both_dropped(self) -> None:
        """Same posture as the projection map: ambiguity drops, never guesses.

        An infix two contracts derive cannot name the owner of an exclusion, so
        the delta falls through to the normal attribution path instead.
        """
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _nonprojection("node_a_b"),
            _nonprojection("node_a__b"),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert select_nonprojection_group_infixes(manifest, suffixes) == {}

    def test_an_infix_colliding_with_a_projection_infix_is_dropped(self) -> None:
        """A group the projection map can match is never excluded instead.

        ``normalize_kafka_identifier`` collapses separators, so a
        non-projection named ``projection__x`` mints the same infix as a
        projection named ``projection_x``. Excluding on it would silence a real
        projection's flow, which is strictly worse than the DEGRADED it
        replaces.
        """
        manifest = _manifest(
            _projection("projection_x"),
            _nonprojection("projection__x"),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert select_nonprojection_group_infixes(manifest, suffixes) == {}

    def test_an_infix_nested_in_a_projection_infix_is_dropped(self) -> None:
        """Containment either way is a match either way, so it drops."""
        manifest = _manifest(
            _projection("node_x", package_name="pkg.omnimarket"),
            _nonprojection("node_x", package_name="omnimarket"),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert select_nonprojection_group_infixes(manifest, suffixes) == {}
