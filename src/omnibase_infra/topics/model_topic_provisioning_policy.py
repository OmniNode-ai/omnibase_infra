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
  a concrete number, and the only place a concrete number is checked against
  the environment's durability floor.
* Against a managed cluster the policy has **no default at all**: an undeclared
  replication factor fails closed, and RF1 is rejected before any
  ``CreateTopics`` is issued.

The policy is resolved from the live Kafka client configuration, not from a
lane label or a caller argument: ``sasl_mechanism == "AWS_MSK_IAM"`` is how
this codebase talks to MSK, so it is an un-forgeable discriminator for "this is
the managed cluster".
"""

from __future__ import annotations

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

# The managed-staging durability floor. RF1 on MSK is unrecoverable data loss on
# a single broker failure and blocks broker update operations; the MSK broker
# default is RF2, which is also what the managed-staging canary namespace
# declares (``managed_staging_canary_catalog_namespace.yaml``).
MANAGED_MINIMUM_REPLICATION_FACTOR: int = 2

# Self-hosted brokers (local Redpanda, `.201` lanes, CI sandboxes) are routinely
# single-node, so RF1 is both legal and the only creatable value there.
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
            undeclared replication factor is refused (managed profile).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    profile: EnumTopicProvisioningProfile
    minimum_replication_factor: int = Field(ge=1)
    default_replication_factor: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _default_satisfies_floor(self) -> Self:
        """A policy may not declare a default below its own durability floor."""
        if (
            self.default_replication_factor is not None
            and self.default_replication_factor < self.minimum_replication_factor
        ):
            raise ValueError(
                f"default_replication_factor="
                f"{self.default_replication_factor} is below "
                f"minimum_replication_factor="
                f"{self.minimum_replication_factor}; a policy cannot default to "
                f"a value it would itself reject"
            )
        return self

    @property
    def is_managed(self) -> bool:
        """``True`` when the target broker is a managed (MSK) cluster."""
        return self.profile is EnumTopicProvisioningProfile.MANAGED

    @classmethod
    def self_hosted(cls) -> ModelTopicProvisioningPolicy:
        """Policy for a self-hosted broker: RF1 allowed, declared default of 1."""
        return cls(
            profile=EnumTopicProvisioningProfile.SELF_HOSTED,
            minimum_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
            default_replication_factor=SELF_HOSTED_REPLICATION_FACTOR,
        )

    @classmethod
    def managed(cls) -> ModelTopicProvisioningPolicy:
        """Policy for a managed (MSK) cluster: RF1 rejected, no implicit default."""
        return cls(
            profile=EnumTopicProvisioningProfile.MANAGED,
            minimum_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            default_replication_factor=None,
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
                policy has no default (managed profile), or when the resolved
                value is below the policy's durability floor.
        """
        if declared is None:
            if self.default_replication_factor is None:
                raise self._violation(
                    topic=topic,
                    detail=(
                        "no replication_factor declared by the owning contract "
                        "and the managed-staging policy has no implicit default. "
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
