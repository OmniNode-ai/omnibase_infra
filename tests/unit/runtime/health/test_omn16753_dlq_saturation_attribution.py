# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The DLQ-saturation dimension names the projection that actually failed (OMN-16753).

RED-first reproduction of a misattribution measured live on the
``.201`` stability-test lane at 2026-09-08T17:31:44.731789Z. The runtime health
monitor published ``projection_dlq_saturation`` DEGRADED naming
``projection_work_events``. That projection was writing: its target table held
148,912 rows with a ``max(emitted_at)`` seconds old, and its consumer group was
Stable and advancing. The projection that was actually losing every event was
``projection_hook_ledger``, whose dispatch raised ``AttributeError`` on every
record and boundary-DLQ'd it with the offset committed.

The mechanism is a dict collision, not a counting error. ``evaluate_projection_
liveness`` folded the per-``(consumer_group, topic)`` flow deltas through::

    topic_to_projection = {topic: ref.name for ref in projections
                           for topic in ref.subscribe_topics}

``projections`` is ordered by contract name, so for a topic declared by more
than one projection the alphabetically LAST declarer silently absorbs every
sibling's counters. Three projections declare the same four
``onex.evt.omniclaude.*`` topics on this lane -- ``projection_hook_ledger`` <
``projection_session_replay`` < ``projection_work_events`` -- so the healthy
projection was named for its failing peer's loss, and the failing peer could
not be named at all.

Cost of the misattribution: the refresh gate rolled the lane back and the
diagnosis pointed at a projection with nothing wrong with it.

Related Tickets:
    - OMN-16753: this ticket
    - OMN-16994: the dimension being corrected
    - OMN-17562: the same "a topic is not an attribution" finding, on the
      attachment half; this is that finding applied to the saturation half
    - OMN-15837: the stability refresh gate that reported the wrong name
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
    select_projection_contracts,
    select_projection_group_suffixes,
)

TOOL_EXECUTED = "onex.evt.omniclaude.tool-executed.v1"
PROMPT_SUBMITTED = "onex.evt.omniclaude.prompt-submitted.v1"

# The three co-declaring projections, in the alphabetical order that decides
# which one the pre-fix dict collision hands the whole topic to.
HOOK_LEDGER = "projection_hook_ledger"
SESSION_REPLAY = "projection_session_replay"
WORK_EVENTS = "projection_work_events"


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
    topics: tuple[str, ...] = (TOOL_EXECUTED, PROMPT_SUBMITTED),
    package_name: str = "omnimarket",
    version: ModelContractVersion | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=version or ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name=package_name,
        event_bus=ModelEventBusWiring(subscribe_topics=topics, publish_topics=()),
        db_io=ModelDbOwnershipSubcontract(db_tables=[_table("projection_target")]),
    )


def _manifest(*contracts: ModelDiscoveredContract) -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(contracts=tuple(contracts), errors=())


def _group(name: str, *, env: str = "stability-test", version: str = "1.0.0") -> str:
    """The bare base group id the wiring seam binds the flow counters with.

    ``handler_wiring`` registers the counters with
    ``compute_consumer_group_id(identity, CONSUME)`` -- BEFORE the bus appends
    its ``.__i.<instance>`` and ``.__t.<topic>`` suffixes -- so this, not the
    live rpk group name, is what a flow delta carries.
    """
    return f"{env}.omnimarket.{name}.consume.{version}"


def _window(
    deltas: tuple[tuple[str, str, int, int], ...],
    *,
    sequence: int = 1,
) -> ModelNodeFlowWindow:
    """One closed window. Each delta is (consumer_group, topic, in, dlq)."""
    node_id = uuid4()
    start = datetime(2026, 9, 8, 17, 0, tzinfo=UTC) + timedelta(seconds=30 * sequence)
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


# The live shape, reproduced exactly. ``projection_hook_ledger`` was the only
# one of the three co-declarers this process actually consumed on, and every
# record it took failed at dispatch and was boundary-DLQ'd with the offset
# committed (1855 tool-executed + 17 prompt-submitted in the sampled window).
# Pre-fix, the topic key resolved to the alphabetically last declarer and the
# dimension published:
#
#   "1 projection(s) routed 100% of consumed events to a DLQ/quarantine sink
#    over N flow window(s) - offsets commit, so lag reads 0 over a total loss:
#    projection_work_events"
#
# which is the verbatim string recovered off
# ``onex.evt.omnibase-infra.runtime-health-check.v1`` for the
# 2026-09-08T17:31:44.731789Z verdict.
_LIVE_DELTAS = (
    (_group(HOOK_LEDGER), TOOL_EXECUTED, 1855, 1855),
    (_group(HOOK_LEDGER), PROMPT_SUBMITTED, 17, 17),
)

# The same three projections with all of them consuming: the failing one still
# has to be the one named, and the collision must not dilute its ratio either.
_ALL_THREE_CONSUMING_DELTAS = _LIVE_DELTAS + (
    (_group(SESSION_REPLAY), TOOL_EXECUTED, 1855, 0),
    (_group(SESSION_REPLAY), PROMPT_SUBMITTED, 17, 0),
    (_group(WORK_EVENTS), TOOL_EXECUTED, 1855, 0),
    (_group(WORK_EVENTS), PROMPT_SUBMITTED, 17, 0),
)


@pytest.mark.unit
class TestSaturationAttribution:
    """A shared topic is not an attribution; a consumer group is."""

    def test_names_the_failing_projection_not_its_alphabetically_last_peer(
        self,
    ) -> None:
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[_window(_LIVE_DELTAS)],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)

    def test_the_healthy_peers_are_not_named(self) -> None:
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[_window(_LIVE_DELTAS)],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        assert WORK_EVENTS not in verdict.dlq_saturated_projections
        assert SESSION_REPLAY not in verdict.dlq_saturated_projections

    def test_a_healthy_peer_consuming_the_same_topic_is_still_not_named(
        self,
    ) -> None:
        """The collision hurts in both directions, and both are fixed here.

        With every co-declarer consuming, the pre-fix fold summed all three
        groups' counters under one name -- which DILUTES the failing
        projection's ratio below 1.0 and reports the lane clean. Attribution by
        group is what makes the failing projection's own ratio readable.
        """
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[_window(_ALL_THREE_CONSUMING_DELTAS)],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)

    def test_the_full_live_group_name_also_attributes(self) -> None:
        """A delta may carry the bare base id or the full live name.

        ``handler_wiring`` registers the counters with the base id, but the bus
        appends ``.__i.<instance>.__t.<topic>`` when it joins, and the live
        broker prints that longer form. Both have to resolve.
        """
        manifest = _manifest(_projection(HOOK_LEDGER), _projection(WORK_EVENTS))
        projections = select_projection_contracts(manifest)
        live_group = (
            f"{_group(HOOK_LEDGER)}.__i.stability-test-main.__t.{TOOL_EXECUTED}"
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED}),
            flow_windows=[_window(((live_group, TOOL_EXECUTED, 600, 600),))],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)

    def test_env_prefix_of_the_live_group_is_not_part_of_the_match(self) -> None:
        """The gate must attribute on a lane whose env differs from this process.

        The suffix carries only contract facts (package, name, purpose,
        version). Binding the env in would make attribution depend on a
        process-wide variable that no contract declares, and a mismatch would
        fail silently back to "nobody is saturated".
        """
        manifest = _manifest(_projection(HOOK_LEDGER), _projection(WORK_EVENTS))
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[
                _window(
                    (
                        (
                            _group(HOOK_LEDGER, env="some-other-lane"),
                            TOOL_EXECUTED,
                            600,
                            600,
                        ),
                        (
                            _group(WORK_EVENTS, env="some-other-lane"),
                            TOOL_EXECUTED,
                            600,
                            0,
                        ),
                    )
                )
            ],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)

    def test_sole_declarer_still_attributes_with_no_group_map(self) -> None:
        """Regression guard: the single-declarer path is unchanged by the fix."""
        manifest = _manifest(_projection(HOOK_LEDGER, topics=(TOOL_EXECUTED,)))
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED}),
            flow_windows=[
                _window((("some-unrecognised-group", TOOL_EXECUTED, 500, 500),))
            ],
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)

    def test_shared_topic_with_no_group_map_names_nobody_and_says_so(self) -> None:
        """Unattributable is reported as unattributable, never as a name.

        With no suffix map an unrecognised group on a topic THREE projections
        declare carries no attribution at all. Naming the alphabetically last
        declarer is what this ticket removes; naming nobody silently would
        replace a wrong answer with a false all-clear, so the topic is carried
        onto the verdict and into the dimension detail instead.
        """
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[
                _window((("an-unrecognised-group", TOOL_EXECUTED, 1855, 1855),))
            ],
        )
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == (TOOL_EXECUTED,)
        assert TOOL_EXECUTED in describe_dlq_saturation(verdict)

    def test_detail_names_the_failing_projection(self) -> None:
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[_window(_LIVE_DELTAS)],
            projection_group_suffixes=select_projection_group_suffixes(
                manifest, projections
            ),
        )
        detail = describe_dlq_saturation(verdict)
        assert HOOK_LEDGER in detail
        assert WORK_EVENTS not in detail


@pytest.mark.unit
class TestUnattributableFlowDegradesTheVerdict:
    """MAJOR (adversarial review, omnibase_infra#3334).

    The first revision of this fix carried ``unattributable_flow_topics`` onto
    the model and appended a parenthetical to the prose, and stopped there. The
    dimension status was still ``"DEGRADED" if dlq_saturated_projections else
    "HEALTHY"`` at the call site, so a lane whose ENTIRE flow could not be
    attributed published a **HEALTHY** ``projection_dlq_saturation`` -- the
    excluded topics annotated in a detail string nothing gates on.

    A lane whose flow cannot be attributed is not a clean lane. It is an
    unanswered question on the one dimension that exists because a total loss
    reads green on lag, on group state and on process health.
    """

    @staticmethod
    def _unattributable_verdict():
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        return evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[
                _window((("an-unrecognised-group", TOOL_EXECUTED, 1855, 1855),))
            ],
        )

    def test_an_unattributable_topic_is_not_healthy(self) -> None:
        verdict = self._unattributable_verdict()
        # The premise: nothing was PROVEN saturated ...
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == (TOOL_EXECUTED,)
        # ... and the dimension still refuses to say the lane is fine.
        assert dlq_saturation_status(verdict) == "DEGRADED"

    def test_a_measured_saturation_is_still_degraded(self) -> None:
        """The original signal is untouched by the new one."""
        manifest = _manifest(_projection(HOOK_LEDGER, topics=(TOOL_EXECUTED,)))
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED}),
            flow_windows=[_window(((_group(HOOK_LEDGER), TOOL_EXECUTED, 500, 500),))],
        )
        assert verdict.dlq_saturated_projections == (HOOK_LEDGER,)
        assert dlq_saturation_status(verdict) == "DEGRADED"

    def test_a_clean_attributed_lane_is_still_healthy(self) -> None:
        """The control. Degrading everything would be its own false signal."""
        manifest = _manifest(_projection(HOOK_LEDGER, topics=(TOOL_EXECUTED,)))
        projections = select_projection_contracts(manifest)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED}),
            flow_windows=[_window(((_group(HOOK_LEDGER), TOOL_EXECUTED, 500, 0),))],
        )
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == ()
        assert dlq_saturation_status(verdict) == "HEALTHY"

    def test_the_status_and_the_prose_come_from_the_same_verdict(self) -> None:
        """They drifted apart once; the status is now derived, not restated."""
        verdict = self._unattributable_verdict()
        assert dlq_saturation_status(verdict) == "DEGRADED"
        assert TOOL_EXECUTED in describe_dlq_saturation(verdict)

    def test_the_monitor_publishes_the_degraded_dimension_and_verdict(self) -> None:
        """End to end through the monitor's own dimension construction.

        Asserted on the monitor rather than only on the pure function, because
        the defect WAS at the call site: the pure half was already right.
        """
        from omnibase_infra.models.health.model_runtime_health_dimension import (
            ModelRuntimeHealthDimension,
        )
        from omnibase_infra.services.service_runtime_health_monitor import _worst

        verdict = self._unattributable_verdict()
        dimension = ModelRuntimeHealthDimension(
            name="projection_dlq_saturation",
            status=dlq_saturation_status(verdict),
            detail=describe_dlq_saturation(verdict),
        )
        assert dimension.status == "DEGRADED"
        # And it carries the aggregate with it, so the refresh gate -- which
        # requires a PASS verdict -- refuses rather than proving the lane.
        assert _worst(["HEALTHY", "HEALTHY", dimension.status]) == "DEGRADED"
        assert TOOL_EXECUTED in dimension.detail

    def test_the_status_is_produced_in_the_dimension_vocabulary(self) -> None:
        """No narrowing step sits between the function and the dimension.

        The first revision passed the status through a string-narrowing helper
        that mapped anything unrecognised to DEGRADED. That is fail-closed, but
        it also silently regrades a typo. The function now returns the
        dimension's own Literal, so a wrong value is a type error where it is
        written instead of a quiet downgrade at the call site.
        """
        import omnibase_infra.services.service_runtime_health_monitor as monitor

        assert not hasattr(monitor, "_as_health_status")
        for verdict in (
            self._unattributable_verdict(),
            evaluate_projection_liveness(
                projections=(), attached_topics=frozenset(), flow_windows=[]
            ),
        ):
            assert dlq_saturation_status(verdict) in {
                "HEALTHY",
                "DEGRADED",
                "CRITICAL",
            }


@pytest.mark.unit
class TestAmbiguousGroupMatchDropsRatherThanGuesses:
    """MINOR (adversarial round 2): first-match iteration was order-dependent.

    ``_attribute_delta`` returned on the FIRST suffix contained in the consumer
    group. The guards in ``select_projection_group_suffixes`` remove infixes
    that are ambiguous or nested within the map, but they cannot prove a real
    group id contains no two of them. Where two match, the answer depended on
    manifest insertion order -- a guess wearing the costume of a fact, which is
    the defect this whole ticket removes.
    """

    def test_two_matching_infixes_yield_no_attribution(self) -> None:
        from omnibase_infra.runtime.health.projection_liveness import _attribute_delta

        declarers = {TOOL_EXECUTED: frozenset({HOOK_LEDGER, SESSION_REPLAY})}
        # A group id that genuinely contains both infixes.
        group = f".pkg.{HOOK_LEDGER}.consume..pkg.{SESSION_REPLAY}.consume."
        suffixes = {
            f".pkg.{HOOK_LEDGER}.consume.": HOOK_LEDGER,
            f".pkg.{SESSION_REPLAY}.consume.": SESSION_REPLAY,
        }
        assert (
            _attribute_delta(
                group,
                TOOL_EXECUTED,
                group_suffixes=suffixes,
                topic_declarers=declarers,
                nonprojection_infixes={},
            ).outcome
            == "UNATTRIBUTABLE"
        )
        # Reversing the map's insertion order gives the same answer, which is
        # the property the first-match form could not hold.
        assert (
            _attribute_delta(
                group,
                TOOL_EXECUTED,
                group_suffixes=dict(reversed(list(suffixes.items()))),
                topic_declarers=declarers,
                nonprojection_infixes={},
            ).outcome
            == "UNATTRIBUTABLE"
        )

    def test_exactly_one_matching_infix_still_attributes(self) -> None:
        from omnibase_infra.runtime.health.projection_liveness import _attribute_delta

        declarers = {TOOL_EXECUTED: frozenset({HOOK_LEDGER, SESSION_REPLAY})}
        suffixes = {
            f".pkg.{HOOK_LEDGER}.consume.": HOOK_LEDGER,
            f".pkg.{SESSION_REPLAY}.consume.": SESSION_REPLAY,
        }
        attribution = _attribute_delta(
            f"env.pkg.{HOOK_LEDGER}.consume.1.0.0",
            TOOL_EXECUTED,
            group_suffixes=suffixes,
            topic_declarers=declarers,
            nonprojection_infixes={},
        )
        assert attribution.outcome == "ATTRIBUTED"
        assert attribution.projection == HOOK_LEDGER

    def test_an_ambiguous_match_reaches_the_verdict_as_unattributable(self) -> None:
        """And therefore degrades the dimension, rather than vanishing."""
        manifest = _manifest(
            _projection(HOOK_LEDGER),
            _projection(SESSION_REPLAY),
            _projection(WORK_EVENTS),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        ambiguous = "".join(suffixes)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({TOOL_EXECUTED, PROMPT_SUBMITTED}),
            flow_windows=[_window(((ambiguous, TOOL_EXECUTED, 900, 900),))],
            projection_group_suffixes=suffixes,
        )
        assert verdict.dlq_saturated_projections == ()
        assert verdict.unattributable_flow_topics == (TOOL_EXECUTED,)
        assert dlq_saturation_status(verdict) == "DEGRADED"


@pytest.mark.unit
class TestSelectProjectionGroupSuffixes:
    """The infix map is contract-derived and refuses an ambiguous entry."""

    def test_key_is_the_contract_half_of_the_group_id(self) -> None:
        manifest = _manifest(_projection(HOOK_LEDGER))
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert suffixes == {f".omnimarket.{HOOK_LEDGER}.consume.": HOOK_LEDGER}

    def test_the_version_is_not_part_of_the_key(self) -> None:
        """Measured on the lab 2026-09-08: the running lane's group carried
        ``consume.1.1.0`` for ``projection_session_replay`` while the manifest
        read beside it declared a different version. A version-bound key
        matched nothing, silently, which is a false all-clear."""
        manifest = _manifest(
            _projection(
                HOOK_LEDGER, version=ModelContractVersion(major=1, minor=1, patch=0)
            )
        )
        projections = select_projection_contracts(manifest)
        assert select_projection_group_suffixes(manifest, projections) == {
            f".omnimarket.{HOOK_LEDGER}.consume.": HOOK_LEDGER
        }

    def test_a_key_contained_in_another_key_is_dropped(self) -> None:
        """A group matching the longer key must not also match the shorter one.

        The short key is the ambiguous half, so it is the half that goes; the
        long one still discriminates and is kept. A group carrying only the
        short form then attributes to nothing, which is the intended trade --
        sensitivity, never a wrong name.
        """
        manifest = _manifest(
            _projection("projection_x", package_name="omnimarket"),
            _projection("projection_x", package_name="pkg.omnimarket"),
        )
        projections = select_projection_contracts(manifest)
        assert select_projection_group_suffixes(manifest, projections) == {
            ".pkg.omnimarket.projection_x.consume.": "projection_x"
        }

    def test_only_in_scope_projections_are_mapped(self) -> None:
        """A contract the selector excluded must not be attributable either.

        Otherwise a name no dimension can report reaches the totals and can
        push an in-scope sibling's ratio around.
        """
        manifest = _manifest(_projection(HOOK_LEDGER), _projection(WORK_EVENTS))
        projections = tuple(
            ref
            for ref in select_projection_contracts(manifest)
            if ref.name != WORK_EVENTS
        )
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert set(suffixes.values()) == {HOOK_LEDGER}

    def test_two_contracts_deriving_one_suffix_are_both_dropped(self) -> None:
        """A suffix that maps to two contracts attributes to neither.

        ``normalize_kafka_identifier`` collapses consecutive separators, so
        ``projection_a_b`` and ``projection_a__b`` mint the same group suffix.
        Same posture as ``attributable_subscribe_topics``: an ambiguous key
        yields no attribution rather than an arbitrary one.
        """
        manifest = _manifest(
            _projection("projection_a_b"),
            _projection("projection_a__b"),
        )
        projections = select_projection_contracts(manifest)
        suffixes = select_projection_group_suffixes(manifest, projections)
        assert suffixes == {}
