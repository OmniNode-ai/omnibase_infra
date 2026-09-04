# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``projection_write_path`` must judge per contract, not per shared topic (OMN-17562).

RED-first reproduction of the measurement defect found on the ``.201`` dev and
stability lanes on 2026-09-04, after omnibase_infra #3156 + omnimarket #2278 +
the two compose-writer PRs merged and both lanes were redeployed. At 09:26Z the
runtime reported ``projection_count 22``, ``nonwriting 8``, ``status DEGRADED``
on all four lane ports with **12 writers healthy** — the writers were deployed
and consuming, so the DEGRADED word was wrong.

The defect
----------
``evaluate_projection_liveness`` computed ``nonwriting_attached_projections``
as::

    ref.name for ref in kernel_nonwriting
    if any(topic in attached_topics for topic in ref.subscribe_topics)

``attached_topics`` is a **topic-keyed** set derived from the live bus registry.
A topic is in it when *any* contract in this process subscribed it. So the
expression asks

    "is this projection's declared subscribe topic subscribed in this process?"

when the fact the dimension needs is

    "did this process subscribe **on behalf of this projection**?"

Those differ exactly when a topic has more than one declaring contract, which
is the common case on these lanes:

* ``onex.evt.omninode.node-introspection.v1`` is declared by
  ``node_ledger_projection_compute`` and ``node_registration_orchestrator``
  as well as by the kernel-nonwriting projections;
* ``onex.evt.omninode.node-heartbeat.v1`` by four more contracts.

The kernel correctly withheld the subscription for every kernel-nonwriting
projection (OMN-17562's wiring half), but the topics stayed in the registry
because their *other* owners subscribed them. Three projections were named as
still-attached silent-loss sites on that basis alone —
``projection_llm_cost`` and ``projection_registration`` on the ``main``
profile, ``projection_live_events`` on ``effects``.

What is fixed, and what is deliberately NOT
-------------------------------------------
Attribution is resolved against the manifest's per-topic declarer census,
combined with the per-contract dispatch facts already recorded by
``projection_dispatch_ledger`` (``dispatch_is_noop`` per handler entry ->
``projections_with_no_live_dispatcher``). A topic is attributable to a
kernel-nonwriting projection only when every OTHER contract declaring it is
also kernel-nonwriting — i.e. no contract with a live in-process dispatcher
could have put that topic in the registry.

The real fault stays red: a projection whose topic this process subscribed for
it, and which dispatches nothing, is still named. The sole loss is sensitivity
on a topic shared with a live-dispatching contract, where a topic-keyed
registry cannot attribute the subscription at all — declining to name is the
only truthful answer there, and the wiring seam's withholding (which IS
per-contract) is what actually prevents the loss.

Related Tickets:
    - OMN-17562: this ticket — per-contract write-path judgement
    - OMN-17557: the DEGRADED lane measurement this closes
    - OMN-17448: the dimension itself
    - OMN-16994: the two masks it sits alongside
"""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.projection_liveness import (
    describe_projection_write_path,
    evaluate_projection_liveness,
    select_kernel_nonwriting_projections,
    select_projection_contracts,
)
from omnibase_infra.runtime.projection_dispatch_ledger import (
    projections_with_no_live_dispatcher,
    record_dispatch_skipped_projection,
    record_live_projection_dispatch,
    reset_dispatch_skipped_projections,
)

# The live shape, verbatim from the 2026-09-04 09:26Z lane read.
SHARED_TOPIC = "onex.evt.omninode.node-introspection.v1"
SOLE_TOPIC = "onex.tenant.events"
LLM_COST = "node_projection_llm_cost"
REGISTRATION = "node_projection_registration"
TENANT_REGISTRY = "node_projection_tenant_registry"
LEDGER_COMPUTE = "node_ledger_projection_compute"
REGISTRATION_ORCHESTRATOR = "node_registration_orchestrator"


@pytest.fixture(autouse=True)
def _clean_ledger() -> Any:
    reset_dispatch_skipped_projections()
    yield
    reset_dispatch_skipped_projections()


def _table(name: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema="omninode_internal",
        migration="0001_init.sql",
        access="write",
        role="projection_target",
    )


def _contract(
    *, name: str, topic: str, with_db_io: bool = True
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name="omnimarket" if with_db_io else "omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        db_io=(
            ModelDbOwnershipSubcontract(db_tables=[_table(f"{name}_mirror")])
            if with_db_io
            else None
        ),
    )


def _shared_topic_manifest() -> ModelAutoWiringManifest:
    """Two kernel-nonwriting projections and two live co-owners on one topic.

    ``node_ledger_projection_compute`` is a raw-event projection (no
    ``db_io.db_tables``) and ``node_registration_orchestrator`` is not a
    projection at all — neither is selected by the projection discriminator.
    They are in the manifest precisely because the declarer census must read
    EVERY contract, not only the ones the health dimension scopes.
    """
    return ModelAutoWiringManifest(
        contracts=(
            _contract(name=LLM_COST, topic=SHARED_TOPIC),
            _contract(name=REGISTRATION, topic=SHARED_TOPIC),
            _contract(name=LEDGER_COMPUTE, topic=SHARED_TOPIC, with_db_io=False),
            _contract(
                name=REGISTRATION_ORCHESTRATOR,
                topic=SHARED_TOPIC,
                with_db_io=False,
            ),
        ),
        errors=(),
    )


def _sole_owner_manifest() -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(
        contracts=(_contract(name=TENANT_REGISTRY, topic=SOLE_TOPIC),), errors=()
    )


@pytest.mark.unit
class TestSharedTopicIsNotAttribution:
    """A co-owner's subscription must not be read as this projection's."""

    def test_a_shared_topic_subscribed_for_another_contract_is_not_the_defect(
        self,
    ) -> None:
        """RED before this ticket: the three lane false positives.

        ``projection_llm_cost`` dispatches nothing here and the kernel
        therefore withheld its subscription. ``node-introspection.v1`` is in the
        registry anyway because ``node_ledger_projection_compute`` and
        ``node_registration_orchestrator`` subscribed it. Nothing was consumed
        on this projection's behalf, so nothing was destroyed.
        """
        manifest = _shared_topic_manifest()
        kernel_nonwriting = frozenset({LLM_COST, REGISTRATION})

        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({SHARED_TOPIC}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )

        assert verdict.nonwriting_projections == (LLM_COST, REGISTRATION)
        assert verdict.nonwriting_attached_projections == (), (
            "a topic subscribed on behalf of a different live contract is not "
            "evidence that THIS projection is consuming"
        )

    def test_the_detail_does_not_claim_events_are_destroyed(self) -> None:
        manifest = _shared_topic_manifest()
        kernel_nonwriting = frozenset({LLM_COST, REGISTRATION})
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({SHARED_TOPIC}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        detail = describe_projection_write_path(verdict)
        assert "not subscribed" in detail
        assert "destroyed" not in detail

    def test_the_ledger_facts_drive_the_same_verdict_end_to_end(self) -> None:
        """The per-contract dispatch facts, not a hand-built name set.

        ``projection_llm_cost`` has only a no-op entry; ``projection_
        registration`` pairs one no-op entry with a live in-process one, so it
        keeps its dispatcher and stays in the attachment scope.
        """
        record_dispatch_skipped_projection(LLM_COST, "LlmCostProjectionRunner")
        record_dispatch_skipped_projection(REGISTRATION, "RegistrationRunner")
        record_live_projection_dispatch(REGISTRATION, "HandlerProjectionRegistration")

        manifest = _shared_topic_manifest()
        kernel_nonwriting = projections_with_no_live_dispatcher()
        assert kernel_nonwriting == frozenset({LLM_COST})

        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({SHARED_TOPIC}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        assert verdict.nonwriting_projections == (LLM_COST,)
        assert verdict.nonwriting_attached_projections == ()

    def test_a_co_owner_that_is_itself_nonwriting_does_not_block_attribution(
        self,
    ) -> None:
        """Attribution only needs to exclude a LIVE dispatcher.

        With every declaring contract kernel-nonwriting, the kernel withheld
        every subscription for this topic — so a topic still in the registry
        can only have come from a re-subscription of one of them, which is the
        loss this dimension exists to catch.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(
                _contract(name=LLM_COST, topic=SHARED_TOPIC),
                _contract(name=REGISTRATION, topic=SHARED_TOPIC),
            ),
            errors=(),
        )
        kernel_nonwriting = frozenset({LLM_COST, REGISTRATION})
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({SHARED_TOPIC}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        assert verdict.nonwriting_attached_projections == (LLM_COST, REGISTRATION)


@pytest.mark.unit
class TestTheRealFaultStaysRed:
    """The narrowing must not disarm the dimension."""

    def test_a_sole_owner_still_attached_and_never_dispatching_is_named(self) -> None:
        """The OMN-17448 silent-loss state, unchanged.

        ``node_projection_tenant_registry`` is the only declarer of
        ``onex.tenant.events``, so the subscription in the registry can only be
        its own: this process consumes, acks and destroys every event while no
        handler runs.
        """
        manifest = _sole_owner_manifest()
        kernel_nonwriting = frozenset({TENANT_REGISTRY})
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({SOLE_TOPIC}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        assert verdict.nonwriting_attached_projections == (TENANT_REGISTRY,)
        assert "destroyed" in describe_projection_write_path(verdict)

    def test_a_sole_owner_that_is_not_subscribed_is_not_the_defect(self) -> None:
        manifest = _sole_owner_manifest()
        kernel_nonwriting = frozenset({TENANT_REGISTRY})
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset({"onex.evt.some.other.topic.v1"}),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        assert verdict.nonwriting_projections == (TENANT_REGISTRY,)
        assert verdict.nonwriting_attached_projections == ()

    def test_unreadable_registry_stays_unknown(self) -> None:
        """Empty ``attached_topics`` means unobservable, never "none attached"."""
        manifest = _sole_owner_manifest()
        kernel_nonwriting = frozenset({TENANT_REGISTRY})
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            ),
            attached_topics=frozenset(),
            flow_windows=(),
            kernel_nonwriting=select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            ),
        )
        assert verdict.attachment_evaluated is False
        assert verdict.nonwriting_attached_projections == ()
