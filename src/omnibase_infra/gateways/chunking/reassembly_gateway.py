# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ReassemblyGateway — consumer-side: buffers chunks and reassembles envelopes."""

from datetime import UTC, datetime, timezone
from uuid import UUID

from omnibase_core.models.chunking.model_chunk_series_failed import (
    EnumChunkFailureReason,
    ModelChunkSeriesFailed,
)
from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker
from omnibase_spi.protocols.chunking.protocol_chunkable_envelope import (
    ProtocolChunkableEnvelope,
)


class ReassemblyGateway:
    """Consumer-side gateway that buffers incoming chunks and triggers reassembly.

    One ReassemblyGateway instance should be shared across all chunk series in a
    given consumer context. It maintains a buffer keyed by ``chunk_series_id``.

    Call ``receive(chunk, envelope_factory)`` for each incoming chunk. It returns:
    - ``None`` if the series is still incomplete.
    - A reconstructed envelope instance if all chunks have arrived and checksums pass.
    - ``ModelChunkSeriesFailed`` on checksum mismatch or expiry.

    The buffer is cleared for a series once it is resolved (success or failure).
    """

    def __init__(self, chunker: DefaultEnvelopeChunker) -> None:
        self._chunker = chunker
        # series_id -> list of received chunks
        self._buffer: dict[UUID, list[ModelChunkedEnvelope]] = {}

    def receive(
        self,
        chunk: ModelChunkedEnvelope,
        envelope_factory: type[ProtocolChunkableEnvelope],
    ) -> ProtocolChunkableEnvelope | ModelChunkSeriesFailed | None:
        """Receive a single chunk and attempt reassembly if all chunks are present.

        Args:
            chunk: Incoming wire-format chunk.
            envelope_factory: Class satisfying ProtocolChunkableEnvelope with a
                ``from_bytes(data: bytes)`` classmethod.

        Returns:
            - Reassembled envelope instance when all chunks received and valid.
            - ``ModelChunkSeriesFailed`` on checksum mismatch or series expiry.
            - ``None`` when the series is still incomplete.
        """
        meta = chunk.chunk_metadata
        series_id = meta.chunk_series_id

        # Check expiry on each chunk arrival
        if meta.expiry_timestamp is not None:
            now = datetime.now(tz=UTC)
            if now > meta.expiry_timestamp:
                self._buffer.pop(series_id, None)
                return ModelChunkSeriesFailed(
                    chunk_series_id=series_id,
                    reason=EnumChunkFailureReason.TIMEOUT,
                    received_chunk_count=len(self._buffer.get(series_id, [])),
                    expected_chunk_count=meta.chunk_count,
                    failed_at=now,
                    detail=f"Series expired at {meta.expiry_timestamp.isoformat()}",
                )

        if series_id not in self._buffer:
            self._buffer[series_id] = []
        self._buffer[series_id].append(chunk)

        received = self._buffer[series_id]
        if len(received) < meta.chunk_count:
            return None

        # All chunks received — attempt reassembly
        all_chunks = self._buffer.pop(series_id)
        try:
            result = self._chunker.reassemble(all_chunks, envelope_factory)
        except ValueError as exc:
            return ModelChunkSeriesFailed(
                chunk_series_id=series_id,
                reason=EnumChunkFailureReason.CHECKSUM_MISMATCH,
                received_chunk_count=len(all_chunks),
                expected_chunk_count=meta.chunk_count,
                failed_at=datetime.now(tz=UTC),
                detail=str(exc),
            )
        return result
