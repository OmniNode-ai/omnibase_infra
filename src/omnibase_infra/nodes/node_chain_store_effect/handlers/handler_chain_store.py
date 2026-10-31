# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that stores a verified chain trajectory to Qdrant.

This is an EFFECT handler -- it performs I/O (Qdrant upsert).
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainStoreResult,
)
from omnibase_infra.nodes.node_chain_orchestrator.models.protocol_chain_vector_client import (
    ProtocolChainVectorClient,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "onex_chain_trajectories"


class HandlerChainStore:
    """Stores a verified chain entry and its prompt embedding to Qdrant."""

    def __init__(
        self,
        qdrant_client: ProtocolChainVectorClient | None = None,
        vector_size: int = 4096,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._vector_size = vector_size

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        chain_entry: ModelChainEntry,
        prompt_embedding: list[float],
        correlation_id: UUID,
    ) -> ModelChainStoreResult:
        """Store a verified chain to Qdrant.

        Args:
            chain_entry: The verified chain trajectory to store.
            prompt_embedding: The embedding vector for the prompt.
            correlation_id: Workflow correlation ID.

        Returns:
            ModelChainStoreResult with success/failure status.
        """
        if self._qdrant_client is None:
            return ModelChainStoreResult(
                correlation_id=correlation_id,
                chain_id=chain_entry.chain_id,
                success=False,
                error_message="HandlerChainStore: qdrant_client not configured",
            )

        logger.info(
            "Storing chain %s (correlation_id=%s, steps=%d)",
            chain_entry.chain_id,
            correlation_id,
            len(chain_entry.chain_steps),
        )

        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            logger.exception("qdrant-client not installed")
            return ModelChainStoreResult(
                correlation_id=correlation_id,
                chain_id=chain_entry.chain_id,
                success=False,
                error_message="qdrant-client is required for chain storage",
            )

        # Serialize chain entry to Qdrant payload
        payload = chain_entry.model_dump(mode="json")

        point = PointStruct(
            id=str(chain_entry.chain_id),
            vector=prompt_embedding,
            payload=payload,
        )

        try:
            await asyncio.to_thread(
                self._qdrant_client.upsert,
                collection_name=COLLECTION_NAME,
                points=[point],
            )
        except Exception:
            logger.exception(
                "Failed to store chain %s (correlation_id=%s)",
                chain_entry.chain_id,
                correlation_id,
            )
            return ModelChainStoreResult(
                correlation_id=correlation_id,
                chain_id=chain_entry.chain_id,
                success=False,
                error_message="Qdrant upsert failed",
            )

        logger.info(
            "Chain %s stored successfully (correlation_id=%s)",
            chain_entry.chain_id,
            correlation_id,
        )

        return ModelChainStoreResult(
            correlation_id=correlation_id,
            chain_id=chain_entry.chain_id,
            success=True,
        )
