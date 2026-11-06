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

A retired group whose id is not ``{env}.<service>.<node>.consume.<version>`` (a
consumer that set its own ``KAFKA_CONSUMER_GROUP``) is named by its literal id
through ``from_group_id`` (OMN-21932), which may stand alone in place of the
three segments (OMN-21920)::

    consumer_group_handover:
      from_group_id: agent-observability-postgres
      from_auto_offset_reset: earliest

The source group is derived with the same ``compute_consumer_group_id`` and the
runtime's own environment, so one declaration serves every lane whose retired
container used ``{env}.<service>.<node>.consume.<version>`` (the dev, stability,
pre-PR and collaborator compose files all do). ``from_auto_offset_reset`` is
what the retired consumer did on a partition it never committed (the
omnimarket projection runner uses ``earliest``); the handover applies the same
rule there so the runtime continues where the container would have.

``handover_rule`` says which offsets win when both groups hold a commit
(one-runtime plan T8, the TLA+ model ``formal/consumer_group_handover``):

* ``copy_if_uncommitted`` (the default, the writers' rule): a partition the
  runtime group already committed keeps its commit. A projection writer UPSERTs
  on a deterministic key, so the replay a rollback or a second fold causes is
  absorbed.
* ``last_ran`` (effects and the worker, whose handlers cannot take a repeat):
  the destination takes the commit of the side that last ran, over whatever it
  holds. Every cutover copies that commit onto the destination and only the
  running side then advances, so on each partition the side that last ran holds
  the later of the two commits; the planner reads it that way and needs no
  record of which side was deployed. A second fold therefore never trusts the
  runtime group's commit from before a rollback (the model's SecondFoldFresh),
  and a retry after a crash copies from the side that ran last, not from the
  side the fold started on (its mut_recopy_fixed). The rollback half, the
  retired side taking the runtime group's commit before it starts, applies the
  same rule from that side and is not wired by this key alone.

It is declared beside the other keys, as ``handover_rule: last_ran``.

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
    from_auto_offset_reset: Literal["earliest", "latest"] = Field(
        ...,
        description=(
            "Where the retired consumer started on a partition it never "
            "committed; applied to such partitions so the runtime group "
            "continues exactly where the retired one would have."
        ),
    )

    handover_rule: Literal["copy_if_uncommitted", "last_ran"] = Field(
        default="copy_if_uncommitted",
        description=(
            "Which commit wins on a partition both groups committed: the "
            "runtime group's own (copy_if_uncommitted, projection writers) or "
            "the later of the two, the side that last ran (last_ran, consumers "
            "whose handler cannot take a repeat)."
        ),
    )

    from_group_id: str | None = Field(
        default=None,
        min_length=1,
        strict=True,
        # At least one character that is not a dot, so "." and ".." are refused.
        pattern=r"^\.*[A-Za-z0-9_-][A-Za-z0-9._-]*$",
        description=(
            "The retired consumer group's literal id, for a group the derived "
            "name cannot express. Returned verbatim by ``source_group`` in every "
            "environment; when unset the group is derived from the three "
            "segments above."
        ),
    )

    @model_validator(mode="after")
    def _segments_complete_or_group_id(self) -> Self:
        """Service, node and version are all declared, or ``from_group_id`` is."""
        segments = (self.from_service, self.from_node, self.from_version)
        if any(segment is None for segment in segments) and (
            self.from_group_id is None
            or any(segment is not None for segment in segments)
        ):
            raise ValueError(
                "a handover declares all of from_service, from_node and "
                "from_version, or from_group_id on its own"
            )
        return self

    def source_group(self, environment: str) -> str:
        """The retired consumer group in ``environment``.

        ``from_group_id`` verbatim when declared, otherwise the group derived
        from the service, node and version segments.
        """
        if self.from_group_id is not None:
            return self.from_group_id
        if (
            self.from_service is None
            or self.from_node is None
            or self.from_version is None
        ):
            raise ValueError(
                "a handover declares all of from_service, from_node and "
                "from_version, or from_group_id"
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
