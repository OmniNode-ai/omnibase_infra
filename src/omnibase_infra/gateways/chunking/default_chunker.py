# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DefaultEnvelopeChunker — SHA-256 based implementation of ProtocolEnvelopeChunker."""

import hashlib
import math
from typing import Protocol, runtime_checkable
from uuid import uuid4

from omnibase_core.models.chunking.model_chunk_metadata import ModelChunkMetadata
from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_spi.protocols.chunking.protocol_chunkable_envelope import (
    ProtocolChunkableEnvelope,
)


def _sha256_hex(data: bytes) -> str:
    """Return 'sha256:<hex_digest>' checksum string for data."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


@runtime_checkable
class EnvelopeFactory(Protocol):
    """Factory protocol for reconstructing envelopes from raw bytes."""

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProtocolChunkableEnvelope":
        """Reconstruct an envelope from its byte representation."""
        ...


class DefaultEnvelopeChunker:
    """Concrete ProtocolEnvelopeChunker using SHA-256 checksums.

    Splits a ProtocolChunkableEnvelope into ordered ModelChunkedEnvelope slices
    and reassembles them back into the original envelope. Checksum validation is
    performed on each individual chunk payload during reassembly.

    This class is intentionally dependency-light; it has no external I/O and
    can be tested without any infrastructure.
    """

    def chunk(
        self,
        envelope: ProtocolChunkableEnvelope,
        max_chunk_size: int,
    ) -> list[ModelChunkedEnvelope]:
        """Split a logical envelope into wire-format chunks.

        Args:
            envelope: Logical envelope to split.
            max_chunk_size: Maximum bytes per chunk payload.

        Returns:
            Ordered list of ModelChunkedEnvelope.
        """
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
        payload = envelope.to_bytes()
        payload_checksum = _sha256_hex(payload)
        total_size = len(payload)
        chunk_count = max(1, math.ceil(total_size / max_chunk_size))
        series_id = uuid4()

        chunks: list[ModelChunkedEnvelope] = []
        for index in range(chunk_count):
            start = index * max_chunk_size
            end = min(start + max_chunk_size, total_size)
            chunk_payload = payload[start:end]
            metadata = ModelChunkMetadata(
                chunk_series_id=series_id,
                chunk_index=index,
                chunk_count=chunk_count,
                chunk_size=len(chunk_payload),
                total_size=total_size,
                payload_checksum=payload_checksum,
                chunk_checksum=_sha256_hex(chunk_payload),
                reassembly_strategy="any_order",
            )
            chunks.append(
                ModelChunkedEnvelope(
                    envelope_headers={},
                    chunk_metadata=metadata,
                    chunk_payload=chunk_payload,
                )
            )
        return chunks

    def reassemble(
        self,
        chunks: list[ModelChunkedEnvelope],
        envelope_factory: EnvelopeFactory,
    ) -> ProtocolChunkableEnvelope:
        """Reassemble a list of chunks into the original logical envelope.

        Args:
            chunks: All chunks for a single chunk series (any order).
            envelope_factory: An object satisfying the ``EnvelopeFactory`` protocol
                (has a ``from_bytes(data: bytes)`` classmethod). Used to reconstruct
                the domain envelope from raw bytes.

        Returns:
            Reconstructed envelope via ``envelope_factory.from_bytes``.

        Raises:
            ValueError: If any chunk checksum fails validation.
        """
        if not chunks:
            raise ValueError("Cannot reassemble empty chunk list.")

        # Validate each chunk's individual checksum
        for chunk in chunks:
            expected = _sha256_hex(chunk.chunk_payload)
            if chunk.chunk_metadata.chunk_checksum != expected:
                raise ValueError(
                    f"checksum mismatch on chunk {chunk.chunk_metadata.chunk_index}: "
                    f"expected {expected!r}, got "
                    f"{chunk.chunk_metadata.chunk_checksum!r}"
                )

        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_metadata.chunk_index)
        full_payload = b"".join(c.chunk_payload for c in sorted_chunks)

        # Validate full payload checksum using the first chunk's metadata
        expected_payload_checksum = sorted_chunks[0].chunk_metadata.payload_checksum
        if _sha256_hex(full_payload) != expected_payload_checksum:
            raise ValueError("Full payload checksum mismatch after reassembly.")

        return envelope_factory.from_bytes(full_payload)
