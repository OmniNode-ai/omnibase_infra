# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DefaultEnvelopeChunker."""

import hashlib

import pytest

from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker
from omnibase_spi.protocols.chunking.protocol_chunkable_envelope import (
    ProtocolChunkableEnvelope,
)


class _FakeEnvelope:
    """Minimal ProtocolChunkableEnvelope implementation for testing."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def to_bytes(self) -> bytes:
        return self._payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "_FakeEnvelope":
        return cls(data)


@pytest.mark.unit
class TestDefaultEnvelopeChunker:
    def test_single_chunk_when_payload_fits(self) -> None:
        chunker = DefaultEnvelopeChunker()
        envelope = _FakeEnvelope(b"small payload")
        chunks = chunker.chunk(envelope, max_chunk_size=1024)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ModelChunkedEnvelope)

    def test_multiple_chunks_when_payload_exceeds_max(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"x" * 1000
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=300)
        assert len(chunks) == 4  # ceil(1000/300)

    def test_chunk_count_matches_metadata(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"y" * 500
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=200)
        for chunk in chunks:
            assert chunk.chunk_metadata.chunk_count == len(chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"z" * 600
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=200)
        indices = [c.chunk_metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_all_chunks_share_same_series_id(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"a" * 600
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=200)
        series_ids = {c.chunk_metadata.chunk_series_id for c in chunks}
        assert len(series_ids) == 1

    def test_payload_checksum_is_sha256_of_full_payload(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"checksum test payload"
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=1024)
        expected = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert chunks[0].chunk_metadata.payload_checksum == expected

    def test_chunk_checksum_matches_chunk_payload(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"b" * 600
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=200)
        for chunk in chunks:
            expected = "sha256:" + hashlib.sha256(chunk.chunk_payload).hexdigest()
            assert chunk.chunk_metadata.chunk_checksum == expected

    def test_concatenated_payloads_equal_original(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"reconstruct me " * 100
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=256)
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_metadata.chunk_index)
        reconstructed = b"".join(c.chunk_payload for c in sorted_chunks)
        assert reconstructed == payload

    def test_total_size_equals_payload_length(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"size check" * 50
        envelope = _FakeEnvelope(payload)
        chunks = chunker.chunk(envelope, max_chunk_size=200)
        for chunk in chunks:
            assert chunk.chunk_metadata.total_size == len(payload)

    def test_satisfies_chunkable_envelope_protocol(self) -> None:
        assert isinstance(_FakeEnvelope(b"test"), ProtocolChunkableEnvelope)

    def test_reassemble_reconstructs_original_bytes(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"reassemble this " * 50
        original = _FakeEnvelope(payload)
        chunks = chunker.chunk(original, max_chunk_size=128)
        # reassemble returns a ProtocolChunkableEnvelope
        # We pass a factory to DefaultEnvelopeChunker via reassemble_with
        result = chunker.reassemble(chunks, _FakeEnvelope)
        assert result.to_bytes() == payload

    def test_reassemble_out_of_order_chunks(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"ordered " * 100
        original = _FakeEnvelope(payload)
        chunks = chunker.chunk(original, max_chunk_size=100)
        shuffled = list(reversed(chunks))
        result = chunker.reassemble(shuffled, _FakeEnvelope)
        assert result.to_bytes() == payload

    def test_reassemble_raises_on_checksum_mismatch(self) -> None:
        chunker = DefaultEnvelopeChunker()
        payload = b"corrupt me " * 50
        original = _FakeEnvelope(payload)
        chunks = chunker.chunk(original, max_chunk_size=200)
        # Corrupt the first chunk's payload via a replaced model
        first = chunks[0]
        bad_chunk = first.model_copy(update={"chunk_payload": b"corrupted data"})
        bad_chunks = [bad_chunk] + list(chunks[1:])
        with pytest.raises(ValueError, match="checksum"):
            chunker.reassemble(bad_chunks, _FakeEnvelope)
