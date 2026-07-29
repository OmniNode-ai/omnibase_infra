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

# Self-hosted brokers (local Redpanda, `.201` lanes, CI sandboxes) are
# single-node, so RF1 is both legal and the only *creatable* value there: a
# CreateTopics carrying RF > broker count is rejected outright with
# INVALID_REPLICATION_FACTOR. This is the self-hosted capacity ceiling as well
# as its default, which is what lets a contract declare the production-durable
# RF2 without breaking local and CI provisioning.
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
            physically host. A declared value ABOVE this is reduced to it (with
            a warning) because a ``CreateTopics`` carrying RF > broker count is
            rejected outright. ``None`` means no ceiling. This only ever
            *reduces*, and never below ``minimum_replication_factor`` — the
            validator forbids a ceiling under the floor, so a durability floor
            can never be silently undercut by a capacity ceiling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    profile: EnumTopicProvisioningProfile
    minimum_replication_factor: int = Field(ge=1)
    default_replication_factor: int | None = Field(default=None, ge=1)
    capacity_replication_factor: int | None = Field(default=None, ge=1)

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
    def self_hosted(cls) -> ModelTopicProvisioningPolicy:
        """Policy for a single-node self-hosted broker: everything resolves to RF1.

        Both the default and the capacity ceiling are 1. The ceiling is what
        allows a contract to declare the production-durable RF2 while local
        Redpanda, CI sandboxes, and the ``.201`` lanes keep provisioning: those
        brokers are single-node, and a CreateTopics carrying RF2 against one
        broker fails with ``INVALID_REPLICATION_FACTOR``.
        """
        return cls(
            profile=EnumTopicProvisioningProfile.SELF_HOSTED,
            minimum_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
            default_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
            capacity_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
        )

    @classmethod
    def managed(cls) -> ModelTopicProvisioningPolicy:
        """Policy for a managed (MSK) cluster: RF1 rejected, no capacity ceiling.

        An undeclared replication factor resolves to
        :data:`MANAGED_MINIMUM_REPLICATION_FACTOR` — the cluster's own broker
        default and the value the managed-staging namespace catalog declares —
        never to 1. See the module docstring for why this resolves rather than
        refuses.
        """
        return cls(
            profile=EnumTopicProvisioningProfile.MANAGED,
            minimum_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            default_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            capacity_replication_factor=None,
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

        # Capacity ceiling: only ever reduces, and the validator guarantees it
        # cannot reduce below the durability floor. A declared RF2 on a
        # single-node self-hosted broker becomes RF1 here rather than failing
        # CreateTopics with INVALID_REPLICATION_FACTOR.
        if (
            self.capacity_replication_factor is not None
            and resolved > self.capacity_replication_factor
        ):
            logger.info(
                "Reducing replication_factor %d -> %d for topic %s: the %s "
                "broker cannot host more replicas than it has nodes "
                "(OMN-15395)",
                resolved,
                self.capacity_replication_factor,
                topic,
                self.profile.value,
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


__all__: list[str] = [
    "MANAGED_MINIMUM_REPLICATION_FACTOR",
    "MANAGED_SASL_MECHANISM",
    "SELF_HOSTED_REPLICATION_FACTOR",
    "ModelTopicProvisioningPolicy",
]
