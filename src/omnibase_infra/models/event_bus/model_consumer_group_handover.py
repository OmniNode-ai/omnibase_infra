# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared consumer-group offset handover (OMN-21753, under OMN-21288).

The one-runtime fold retires a standalone writer container and lets the main
runtime consume the writer's contract instead. The two consume under different
consumer groups:

* the container: ``{env}.omnimarket-projections.<writer>-writer.consume.v1``
  (its ``KAFKA_CONSUMER_GROUP``), one group across every topic;
* the runtime: ``compute_consumer_group_id`` of the contract identity
  (``{env}.{package}.{contract name}.consume.{contract version}``), which the
  Kafka bus scopes per topic.

A group with no committed offset starts at the bus's ``auto_offset_reset``,
``latest`` by default, so every event published between the container stopping
and the runtime joining was never projected. This declaration names the
retired consumer, and the runtime copies its committed offsets into its own
group before that group first joins.

Shape, as a TOP-LEVEL key in the node's ``contract.yaml``::

    consumer_group_handover:
      from_service: omnimarket-projections
      from_node: live-events-writer
      from_version: v1
      from_auto_offset_reset: earliest

A retired consumer whose group is a bare name, not built from service, node and
version, declares the name itself in place of the three segments::

    consumer_group_handover:
      from_group_id: agent-observability-postgres
      from_auto_offset_reset: earliest

``from_group_id`` is the group as it is, with no environment prefix and no
derivation, so it serves every lane that retired consumer ran in. A declaration
carries either the three segments or ``from_group_id``, never both.

The source group is derived with the same ``compute_consumer_group_id`` and the
runtime's own environment, so one declaration serves every lane whose retired
container used ``{env}.<service>.<node>.consume.<version>`` (the dev, stability,
pre-PR and collaborator compose files all do). ``from_auto_offset_reset`` is
what the retired consumer did on a partition it never committed (the
omnimarket projection runner uses ``earliest``); the handover applies the same
rule there so the runtime continues where the container would have.

The key is read by ``handler_consumer_group_offset_handover`` and not by
``ModelEventBusSubcontract`` in ``omnibase_core``, for the reason
``model_consume_concurrency`` gives: it is a runtime-local concern, and
``runtime/auto_wiring/discovery.py`` hand-picks the keys it knows, so an
unrecognised top-level key cannot break parsing.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.utils import compute_consumer_group_id

__all__ = ["ModelConsumerGroupHandover"]


class ModelConsumerGroupHandover(BaseModel):
    """The retired consumer whose committed offsets the runtime group resumes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_service: str | None = Field(
        default=None, min_length=1, description="Service segment of the retired group."
    )
    from_node: str | None = Field(
        default=None, min_length=1, description="Node segment of the retired group."
    )
    from_version: str | None = Field(
        default=None, min_length=1, description="Version segment of the retired group."
    )
    from_group_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z0-9._-]+$",
        description=(
            "The retired consumer group by name, for a group that is not built "
            "from service, node and version. Used as it is in every environment."
        ),
    )
    from_auto_offset_reset: Literal["earliest", "latest"] = Field(
        ...,
        description=(
            "Where the retired consumer started on a partition it never "
            "committed; applied to such partitions so the runtime group "
            "continues exactly where the retired one would have."
        ),
    )

    @model_validator(mode="after")
    def _one_form(self) -> Self:
        """Either ``from_group_id`` or all of service, node and version."""
        segments = (self.from_service, self.from_node, self.from_version)
        if self.from_group_id is not None:
            if any(segment is not None for segment in segments):
                raise ValueError(
                    "from_group_id names the retired group by itself; it cannot "
                    "be combined with from_service, from_node or from_version"
                )
        elif any(segment is None for segment in segments):
            raise ValueError(
                "a handover declares either from_group_id or all of "
                "from_service, from_node and from_version"
            )
        return self

    def source_group(self, environment: str) -> str:
        """The retired consumer group in ``environment``."""
        if self.from_group_id is not None:
            return self.from_group_id
        if (
            self.from_service is None
            or self.from_node is None
            or self.from_version is None
        ):
            raise ValueError(
                "a handover declares either from_group_id or all of "
                "from_service, from_node and from_version"
            )
        return compute_consumer_group_id(
            ModelNodeIdentity(
                env=environment,
                service=self.from_service,
                node_name=self.from_node,
                version=self.from_version,
            ),
            EnumConsumerGroupPurpose.CONSUME,
        )
