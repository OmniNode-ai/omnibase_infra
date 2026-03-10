# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ChunkingGateway — producer-side: splits large envelopes into wire chunks."""

from omnibase_core.models.chunking.model_chunk_policy import ModelChunkPolicy
from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker
from omnibase_spi.protocols.chunking.protocol_chunkable_envelope import (
    ProtocolChunkableEnvelope,
)


class ChunkingGateway:
    """Producer-side gateway that evaluates chunk policy and splits large envelopes.

    Usage:
        gateway = ChunkingGateway(chunker=DefaultEnvelopeChunker(), policy=policy)
        chunks = gateway.evaluate_and_chunk(envelope)
        if chunks:
            # publish each chunk individually
        else:
            # publish envelope as-is (below threshold or chunking disabled)
    """

    def __init__(
        self,
        chunker: DefaultEnvelopeChunker,
        policy: ModelChunkPolicy,
    ) -> None:
        self._chunker = chunker
        self._policy = policy

    def evaluate_and_chunk(
        self,
        envelope: ProtocolChunkableEnvelope,
    ) -> list[ModelChunkedEnvelope]:
        """Evaluate policy and optionally chunk the envelope.

        Args:
            envelope: Logical envelope to evaluate.

        Returns:
            Non-empty list of ModelChunkedEnvelope if chunking was triggered.
            Empty list if payload is within threshold or chunking is disabled
            (caller publishes the original envelope as-is).
        """
        if not self._policy.enabled:
            return []

        payload_size = len(envelope.to_bytes())
        if payload_size <= self._policy.max_payload_size_bytes:
            return []

        return self._chunker.chunk(
            envelope,
            max_chunk_size=self._policy.chunk_target_size_bytes,
        )
