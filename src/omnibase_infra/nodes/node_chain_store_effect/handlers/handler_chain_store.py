# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that stores a verified chain trajectory to Qdrant.

This is an EFFECT handler -- it performs I/O (Qdrant upsert).
"""

from __future__ import annotations

import asyncio
import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainStoreRequest,
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

    async def handle(self, request: ModelChainStoreRequest) -> ModelChainStoreResult:
        """Store a verified chain to Qdrant.

        Args:
            request: The verified chain trajectory, its prompt embedding and the
                workflow correlation ID.

        Returns:
            ModelChainStoreResult with success/failure status.
        """
        if self._qdrant_client is None:
            return ModelChainStoreResult(
                correlation_id=request.correlation_id,
                chain_id=request.chain_entry.chain_id,
                success=False,
                error_message="HandlerChainStore: qdrant_client not configured",
            )

        logger.info(
            "Storing chain %s (correlation_id=%s, steps=%d)",
            request.chain_entry.chain_id,
            request.correlation_id,
            len(request.chain_entry.chain_steps),
        )

        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            logger.exception("qdrant-client not installed")
            return ModelChainStoreResult(
                correlation_id=request.correlation_id,
                chain_id=request.chain_entry.chain_id,
                success=False,
                error_message="qdrant-client is required for chain storage",
            )

        # Serialize chain entry to Qdrant payload
        payload = request.chain_entry.model_dump(mode="json")

        point = PointStruct(
            id=str(request.chain_entry.chain_id),
            vector=request.prompt_embedding,
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
                request.chain_entry.chain_id,
                request.correlation_id,
            )
            return ModelChainStoreResult(
                correlation_id=request.correlation_id,
                chain_id=request.chain_entry.chain_id,
                success=False,
                error_message="Qdrant upsert failed",
            )

        logger.info(
            "Chain %s stored successfully (correlation_id=%s)",
            request.chain_entry.chain_id,
            request.correlation_id,
        )

        return ModelChainStoreResult(
            correlation_id=request.correlation_id,
            chain_id=request.chain_entry.chain_id,
            success=True,
        )
