# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Environment-resolved topic replication policy (OMN-15395).

Why this exists
---------------
AWS Health raised ``AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE`` against
``omninode-dev-msk``: 519 of 1,610 topics sat at replication factor 1. The
broker default is RF2 and broker-side auto-create is disabled, so every one of
those topics was created by an explicit ``CreateTopics`` call from this repo's
provisioner that *overrode the broker default down to 1* — a module-level
``DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1`` applied silently whenever the
owning contract declared nothing.

This module removes the implicit default and replaces it with one explicit,
typed resolution seam:

* ``ModelTopicSpec.replication_factor is None`` now means **the owning contract
  declared nothing** — it is no longer silently 1.
* :class:`ModelTopicProvisioningPolicy` is the *only* place a ``None`` becomes
  a concrete number, the only place a concrete number is checked against the
  environment's durability floor, and the only place a value is reduced to what
  the target broker can physically host.
* Against a managed cluster, a *declared* replication factor below the floor
  (the RF1 case) is rejected before any ``CreateTopics`` is issued. An
  *undeclared* replication factor resolves to the managed durability floor —
  never to 1, and never below the MSK broker's own RF2 default.

The capacity ceiling is MEASURED, never assumed
-----------------------------------------------
A policy constructed from configuration alone carries **no** capacity ceiling
(``capacity_replication_factor is None``, ``broker_count is None``). The ceiling
is installed only by :meth:`ModelTopicProvisioningPolicy.with_broker_capacity`,
from a live ``describe_cluster`` broker count read off the same admin client
that will issue the ``CreateTopics``
(:mod:`omnibase_infra.topics.broker_capacity_probe`).

This is load-bearing, not a refinement. The first revision of this policy set
``capacity_replication_factor = 1`` unconditionally for every cluster whose
``sasl_mechanism`` was not ``AWS_MSK_IAM``, and
:meth:`resolve_replication_factor` then silently reduced every declared value
down to it. ``ModelKafkaEventBusConfig`` accepts PLAIN / SCRAM-SHA-256 /
SCRAM-SHA-512 / OAUTHBEARER as well, so *any* multi-broker cluster not reached
over MSK IAM — including an MSK cluster fronted by SCRAM — had its
contract-declared RF2/RF3 clamped to RF1: the exact
``AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE`` condition this module exists to
eliminate, reintroduced by the mechanism meant to prevent it. A ceiling that is
an assumption about the broker rather than a measurement of it is a durability
downgrade wearing a capacity argument.

Two invariants make the ceiling safe:

* it may only ever *reduce* a value, never raise one; and
* it is never installed below the profile's durability floor — a measured
  broker count under the floor leaves the ceiling unset so that resolution
  *refuses* instead of silently creating an under-replicated topic.

A durability requirement is expressed in the CONTRACT. The ceiling exists only
so a contract-declared RF2 does not make provisioning impossible on a broker
that physically has one node, and it says so out loud (``logger.warning``) every
time it fires.

Why an undeclared RF resolves rather than refuses (deviation from a literal
reading of OMN-15395 acceptance criterion (a), recorded here because it is a
judgement call)
---------------------------------------------------------------------------
A refuse-on-undeclared policy was implemented first and measured against the
real contract tree: **168 of 168** provisioned topics carry no contract-declared
replication factor, and **75 of those have no producing declaration anywhere in
this repository** — they appear only in ``event_bus.subscribe_topics``, i.e.
they are produced by omniclaude / omnimarket / CLI relays. Refusing every
undeclared topic therefore makes provisioning a permanent 100% no-op on MSK,
with a third of the topic universe having no in-repo contract that *could* be
fixed. That is strictly worse than the bug being repaired.

What is actually forbidden by (a) is a *module-level constant of 1* silently
overriding the broker's own default. This policy is the opposite of that: the
value is profile-scoped, equal to the managed cluster's own RF2 default, and
identical to the number the managed-staging namespace catalog already declares
(``managed_staging_canary_catalog_namespace.yaml`` →
``default_replication_factor: 2``). Acceptance criterion (c) names the
*divergence* between that catalog's RF2 and the manager's RF1 as the defect;
converging both onto :data:`MANAGED_MINIMUM_REPLICATION_FACTOR` is the fix. A
contract that declares its own replication factor always wins, and a declared
value below the floor always fails closed.

The policy is resolved from the live Kafka client configuration, not from a
lane label or a caller argument: ``sasl_mechanism == "AWS_MSK_IAM"`` is how
this codebase talks to MSK, so it is an un-forgeable discriminator for "this is
the managed cluster".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.topics.enum_topic_provisioning_profile import (
    EnumTopicProvisioningProfile,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

if TYPE_CHECKING:
    from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
        ModelKafkaEventBusConfig,
    )

logger = logging.getLogger(__name__)

# The managed-staging durability floor, and the value an undeclared replication
# factor resolves to on the managed profile. RF1 on MSK is unrecoverable data
# loss on a single broker failure and blocks broker update operations; the MSK
# broker default is RF2, which is also what the managed-staging canary namespace
# declares (``managed_staging_canary_catalog_namespace.yaml`` →
# ``default_replication_factor``, bound to this constant in
# ``model_canary_namespace.py``). One constant, both paths — the RF2-here /
# RF1-there divergence is exactly what OMN-15395 (c) calls the defect.
MANAGED_MINIMUM_REPLICATION_FACTOR: int = 2

# What an UNDECLARED replication factor resolves to on a self-hosted broker
# whose node count has not been measured yet. It is a floor-of-last-resort for
# the unmeasured case only: once
# :meth:`ModelTopicProvisioningPolicy.with_broker_capacity` binds a live broker
# count, an undeclared RF on a multi-node self-hosted cluster resolves to
# ``MANAGED_MINIMUM_REPLICATION_FACTOR`` instead — a 3-broker Redpanda has no
# more business minting RF1 topics than MSK does.
#
# This constant is NOT a capacity ceiling. Nothing reduces a declared value to
# it; only a measured broker count can install a ceiling.
SELF_HOSTED_REPLICATION_FACTOR: int = 1

# The SASL mechanism this codebase uses to authenticate to AWS MSK.
MANAGED_SASL_MECHANISM: str = "AWS_MSK_IAM"


class ModelTopicProvisioningPolicy(BaseModel):
    """Resolves and validates the replication factor for a topic being created.

    Attributes:
        profile: Durability class of the target broker.
        minimum_replication_factor: Hard floor. A spec resolving below this is
            rejected fail-closed — never clamped, never warned-and-continued.
        default_replication_factor: The value an *undeclared* replication
            factor resolves to. ``None`` means there is no default and an
            undeclared replication factor is refused.
        capacity_replication_factor: The most replicas the target broker can
            physically host, **as measured** from a live ``describe_cluster``
            broker count. A declared value ABOVE this is reduced to it (at
            ``logger.warning`` — a durability downgrade is never emitted below
            WARNING) because a ``CreateTopics`` carrying RF > broker count is
            rejected outright with ``INVALID_REPLICATION_FACTOR``. ``None``
            means **unmeasured, therefore no ceiling** — nothing is reduced.
            This only ever *reduces*, and never below
            ``minimum_replication_factor``: the validator forbids a ceiling
            under the floor, so a durability floor can never be silently
            undercut by a capacity ceiling.
        broker_count: The live broker count this policy was bound to, or
            ``None`` when the cluster has not been probed. Provenance for
            ``capacity_replication_factor`` — an unmeasured policy is
            structurally incapable of reducing anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    profile: EnumTopicProvisioningProfile
    minimum_replication_factor: int = Field(ge=1)
    default_replication_factor: int | None = Field(default=None, ge=1)
    capacity_replication_factor: int | None = Field(default=None, ge=1)
    broker_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _bounds_are_coherent(self) -> Self:
        """Floor <= ceiling, and the default must sit inside both bounds."""
        if (
            self.capacity_replication_factor is not None
            and self.capacity_replication_factor < self.minimum_replication_factor
        ):
            raise ValueError(
                f"capacity_replication_factor="
                f"{self.capacity_replication_factor} is below "
                f"minimum_replication_factor="
                f"{self.minimum_replication_factor}; a capacity ceiling may "
                f"never undercut a durability floor — that would silently "
                f"create topics the policy is supposed to reject"
            )
        if self.default_replication_factor is None:
            return self
        if self.default_replication_factor < self.minimum_replication_factor:
            raise ValueError(
                f"default_replication_factor="
                f"{self.default_replication_factor} is below "
                f"minimum_replication_factor="
                f"{self.minimum_replication_factor}; a policy cannot default to "
                f"a value it would itself reject"
            )
        if (
            self.capacity_replication_factor is not None
            and self.default_replication_factor > self.capacity_replication_factor
        ):
            raise ValueError(
                f"default_replication_factor="
                f"{self.default_replication_factor} exceeds "
                f"capacity_replication_factor="
                f"{self.capacity_replication_factor}; a policy cannot default to "
                f"a value the broker cannot host"
            )
        return self

    @property
    def is_managed(self) -> bool:
        """``True`` when the target broker is a managed (MSK) cluster."""
        return self.profile is EnumTopicProvisioningProfile.MANAGED

    @classmethod
    def self_hosted(
        cls, *, broker_count: int | None = None
    ) -> ModelTopicProvisioningPolicy:
        """Policy for a self-hosted broker (local Redpanda, ``.201``, CI).

        With ``broker_count`` unset the policy is **unmeasured**: there is no
        capacity ceiling, so a contract-declared RF2/RF3 reaches
        ``CreateTopics`` exactly as declared. Only a measured node count may
        reduce it — see :meth:`with_broker_capacity`.

        An undeclared replication factor resolves to
        :data:`SELF_HOSTED_REPLICATION_FACTOR` while unmeasured, and to the
        durable :data:`MANAGED_MINIMUM_REPLICATION_FACTOR` once a measurement
        proves the cluster has the nodes for it.

        Args:
            broker_count: Live node count, when already known.
        """
        policy = cls(
            profile=EnumTopicProvisioningProfile.SELF_HOSTED,
            minimum_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
            default_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
            capacity_replication_factor=None,
        )
        if broker_count is None:
            return policy
        return policy.with_broker_capacity(broker_count)

    @classmethod
    def managed(
        cls, *, broker_count: int | None = None
    ) -> ModelTopicProvisioningPolicy:
        """Policy for a managed (MSK) cluster: RF1 rejected fail-closed.

        An undeclared replication factor resolves to
        :data:`MANAGED_MINIMUM_REPLICATION_FACTOR` — the cluster's own broker
        default and the value the managed-staging namespace catalog declares —
        never to 1. See the module docstring for why this resolves rather than
        refuses.

        Args:
            broker_count: Live node count, when already known. A managed
                cluster with FEWER nodes than the durability floor gets no
                ceiling at all, so resolution refuses rather than clamping.
        """
        policy = cls(
            profile=EnumTopicProvisioningProfile.MANAGED,
            minimum_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            default_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            capacity_replication_factor=None,
        )
        if broker_count is None:
            return policy
        return policy.with_broker_capacity(broker_count)

    def with_broker_capacity(self, broker_count: int) -> ModelTopicProvisioningPolicy:
        """Bind this policy to a MEASURED live broker count.

        This is the only way a capacity ceiling is ever installed. The
        durability floor and the profile are carried through untouched; the
        measurement may move exactly two things:

        * the ceiling, to ``broker_count`` — but only when the cluster has at
          least ``minimum_replication_factor`` nodes. A measurement *below* the
          floor leaves the ceiling unset (with a warning) so resolution refuses
          rather than clamping a topic under the durability floor.
        * the undeclared-RF default, UP toward
          :data:`MANAGED_MINIMUM_REPLICATION_FACTOR` when the measured cluster
          can host it. A 3-node self-hosted broker defaults undeclared topics
          to RF2, not RF1 — the unmeasured RF1 default is a conservative
          placeholder for "we have not looked", not a statement about the
          cluster.

        Args:
            broker_count: Live node count from ``describe_cluster``.

        Returns:
            A new policy bound to the measurement.

        Raises:
            ValueError: ``broker_count`` is not a positive node count.
        """
        if broker_count < 1:
            raise ValueError(
                f"broker_count={broker_count} is not a live node count; a "
                "capacity ceiling may only be installed from a real "
                "measurement (OMN-15395)"
            )

        capacity: int | None = broker_count
        if broker_count < self.minimum_replication_factor:
            logger.warning(
                "Cluster reports %d broker(s), below the %s durability floor "
                "of %d; NOT installing a capacity ceiling that would undercut "
                "the floor — under-replicated specs will be refused rather "
                "than silently created (OMN-15395)",
                broker_count,
                self.profile.value,
                self.minimum_replication_factor,
            )
            capacity = None

        default = self.default_replication_factor
        if default is not None:
            # Raise a conservative unmeasured default up to the durable value
            # the measured cluster can actually host, then hold it under the
            # ceiling. Never below the floor: `capacity` is either None or
            # >= minimum_replication_factor by the branch above.
            default = max(
                default, min(MANAGED_MINIMUM_REPLICATION_FACTOR, broker_count)
            )
            if capacity is not None:
                default = min(default, capacity)
            default = max(default, self.minimum_replication_factor)

        return ModelTopicProvisioningPolicy(
            profile=self.profile,
            minimum_replication_factor=self.minimum_replication_factor,
            default_replication_factor=default,
            capacity_replication_factor=capacity,
            broker_count=broker_count,
        )

    @classmethod
    def from_kafka_config(
        cls, config: ModelKafkaEventBusConfig
    ) -> ModelTopicProvisioningPolicy:
        """Derive the policy from the live Kafka client configuration.

        MSK IAM auth (``sasl_mechanism == "AWS_MSK_IAM"``) is how this codebase
        reaches the managed cluster, so it is the discriminator. It is read from
        the same config object the admin client authenticates with, which is why
        a caller cannot declare itself self-hosted while pointed at MSK.

        The discriminator selects the *durability floor*, and nothing else. It
        deliberately does NOT imply a node count: ``ModelKafkaEventBusConfig``
        also accepts PLAIN / SCRAM-SHA-256 / SCRAM-SHA-512 / OAUTHBEARER, so a
        non-IAM cluster may well be multi-broker. The returned policy is
        therefore **unmeasured** — no capacity ceiling — until
        :meth:`with_broker_capacity` binds a live ``describe_cluster`` count.
        """
        if config.sasl_mechanism == MANAGED_SASL_MECHANISM:
            return cls.managed()
        return cls.self_hosted()

    @classmethod
    def from_env(cls) -> ModelTopicProvisioningPolicy:
        """Derive the policy from the standard runtime Kafka environment."""
        from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
            ModelKafkaEventBusConfig,
        )

        return cls.from_kafka_config(ModelKafkaEventBusConfig.default())

    def resolve_replication_factor(self, *, topic: str, declared: int | None) -> int:
        """Return the explicit replication factor to create ``topic`` with.

        Args:
            topic: Topic name (for the error message).
            declared: The owning contract's declared replication factor, or
                ``None`` when the contract declared none.

        Returns:
            The resolved, explicit replication factor.

        Raises:
            TopicReplicationPolicyError: When ``declared`` is ``None`` and the
                policy has no default, or when the resolved value is below the
                policy's durability floor (the RF1-on-MSK case).
        """
        if declared is None:
            if self.default_replication_factor is None:
                raise self._violation(
                    topic=topic,
                    detail=(
                        "no replication_factor declared by the owning contract "
                        f"and the {self.profile.value} policy has no default. "
                        "Declare topic_config.replication_factor >= "
                        f"{self.minimum_replication_factor} in the contract that "
                        "owns this topic (OMN-13238 seam) — refusing to create a "
                        "topic whose durability nobody declared"
                    ),
                    declared=declared,
                )
            resolved = self.default_replication_factor
        else:
            resolved = declared

        # Capacity ceiling: only ever reduces, only ever from a MEASURED broker
        # count, and the validator guarantees it cannot reduce below the
        # durability floor. A declared RF2 on a broker measured at one node
        # becomes RF1 here rather than failing CreateTopics with
        # INVALID_REPLICATION_FACTOR.
        #
        # WARNING, not INFO: this is a durability downgrade of a value some
        # contract explicitly asked for. Emitting it below WARNING makes it
        # invisible under normal log filtering, which is how a silent RF
        # reduction stops being auditable.
        if (
            self.capacity_replication_factor is not None
            and resolved > self.capacity_replication_factor
        ):
            logger.warning(
                "Reducing replication_factor %d -> %d for topic %s: the %s "
                "cluster measured %s broker(s) and cannot host more replicas "
                "than it has nodes (OMN-15395)",
                resolved,
                self.capacity_replication_factor,
                topic,
                self.profile.value,
                self.broker_count,
            )
            resolved = self.capacity_replication_factor

        if resolved < self.minimum_replication_factor:
            raise self._violation(
                topic=topic,
                detail=(
                    f"replication_factor={resolved} is below the "
                    f"{self.profile.value} floor of "
                    f"{self.minimum_replication_factor}. RF1 on a managed "
                    "cluster is unrecoverable data loss on a single broker "
                    "failure and blocks broker updates "
                    "(AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE). Not clamping, "
                    "not warning — refusing to create"
                ),
                declared=declared,
            )
        return resolved

    def resolve_spec(self, spec: ModelTopicSpec) -> ModelTopicSpec:
        """Return ``spec`` with an explicit, policy-approved replication factor.

        Partitions and ``kafka_config`` are carried through untouched — this
        seam only ever *adds* the resolved replication factor, it never
        re-defaults a value the contract declared.
        """
        resolved = self.resolve_replication_factor(
            topic=spec.suffix, declared=spec.replication_factor
        )
        if spec.replication_factor == resolved:
            return spec
        return spec.model_copy(update={"replication_factor": resolved})

    def _violation(
        self, *, topic: str, detail: str, declared: int | None
    ) -> TopicReplicationPolicyError:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="resolve_topic_replication_factor",
            target_name=topic,
        )
        return TopicReplicationPolicyError(
            f"Refusing to provision topic {topic!r}: {detail}",
            context=context,
            topic=topic,
            declared_replication_factor=declared,
            profile=self.profile.value,
            minimum_replication_factor=self.minimum_replication_factor,
        )


def resolve_specs_for_creation(
    policy: ModelTopicProvisioningPolicy,
    specs: Sequence[ModelTopicSpec],
) -> tuple[ModelTopicSpec, ...]:
    """Resolve EVERY spec before any ``CreateTopics`` is issued.

    Fail-closed and batch-scoped: a single spec that violates the environment
    replication policy — the RF1-on-MSK case — aborts the whole batch with ZERO
    creates issued. Not a warning, not a clamp-and-continue, and not a per-topic
    skip that lets the rest of the pass proceed while a durability defect sits
    unfixed in a contract we own. Every violation is collected first so one run
    surfaces every offending contract instead of one per redeploy.

    Module-level rather than a method (matching ``build_provisioning_diff``'s
    shape in the sibling diff module) because there is more than one live
    ``CreateTopics`` path in this repository: the runtime provisioner
    (:class:`~omnibase_infra.event_bus.service_topic_manager.TopicProvisioner`),
    the managed-staging canary checker, and the operator CLI
    ``scripts/create_kafka_topics.py``. A batch resolver owned by one of them is
    a resolver the others silently do without — which is exactly how the CLI
    shipped a flat ``--replication-factor 1`` default that discarded every
    contract's declared ``topic_config.replication_factor`` (OMN-15395 D2).

    Args:
        policy: The environment policy to resolve against.
        specs: The specs about to be created.

    Returns:
        The resolved specs, each carrying an explicit replication factor, in
        input order.

    Raises:
        TopicReplicationPolicyError: Any spec violates the policy. The message
            enumerates up to ten violations and counts the rest.
    """
    resolved: list[ModelTopicSpec] = []
    violations: list[str] = []
    for spec in specs:
        try:
            resolved.append(policy.resolve_spec(spec))
        except TopicReplicationPolicyError as exc:
            violations.append(str(exc))
    if violations:
        shown = violations[:10]
        suffix = (
            f" (+{len(violations) - len(shown)} more)"
            if len(violations) > len(shown)
            else ""
        )
        raise TopicReplicationPolicyError(
            f"Refusing to provision {len(violations)} topic(s) under the "
            f"{policy.profile.value} replication policy; no CreateTopics was "
            "issued. Violations: " + " | ".join(shown) + suffix
        )
    return tuple(resolved)


__all__: list[str] = [
    "MANAGED_MINIMUM_REPLICATION_FACTOR",
    "MANAGED_SASL_MECHANISM",
    "SELF_HOSTED_REPLICATION_FACTOR",
    "ModelTopicProvisioningPolicy",
    "resolve_specs_for_creation",
]
