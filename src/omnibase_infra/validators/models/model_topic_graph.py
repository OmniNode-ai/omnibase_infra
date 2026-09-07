# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The contract graph: nodes are contracts, edges are topics (OMN-18013)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validators.models.model_contract_node import ModelContractNode


class ModelTopicGraph(BaseModel):
    """The contract graph: nodes are contracts, edges are topics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[ModelContractNode, ...]
    producers: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    consumers: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # Topics an external, non-contract actor legitimately publishes (the skill
    # CLI, a GitHub webhook, the omniclaude hook daemon). Declared, never assumed.
    external_producers: dict[str, str] = Field(default_factory=dict)
    # Mirror of external_producers for the consume side: topics whose ONLY
    # consumer lives outside the checked contract corpus (e.g. the onex-api
    # gateway projection in private omninode_infra). The producing contract's
    # own externally_consumed_topics is the preferred declaration, but a
    # PINNED foreign contract cannot be edited from this repo — this is the
    # declared-never-assumed escape for that case. Declared, never assumed.
    external_consumers: dict[str, str] = Field(default_factory=dict)

    @property
    def topics(self) -> set[str]:
        return set(self.producers) | set(self.consumers)

    def edges(self) -> list[tuple[str, str, str]]:
        """Every (producer_node, topic, consumer_node) triple."""
        out: list[tuple[str, str, str]] = []
        for topic, prods in self.producers.items():
            for consumer in self.consumers.get(topic, ()):
                for producer in prods:
                    out.append((producer, topic, consumer))
        return out

    def is_reachable(self, topic: str) -> bool:
        """Can anything, anywhere, ever put a message on this topic?"""
        if self.producers.get(topic):
            return True
        if topic in self.external_producers:
            return True
        # A node's runtime_dispatch.command_topic is the CLI's publish target,
        # so declaring one IS an entry point even with no contract producer.
        return any(n.command_topic == topic for n in self.nodes)


__all__ = ["ModelTopicGraph"]
