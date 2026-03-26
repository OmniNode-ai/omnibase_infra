# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ReassemblyGateway."""

from datetime import UTC, datetime

import pytest

from omnibase_core.models.chunking.model_chunk_series_failed import (
    EnumChunkFailureReason,
    ModelChunkSeriesFailed,
)
from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker
from omnibase_infra.gateways.chunking.reassembly_gateway import ReassemblyGateway


class _FakeEnvelope:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def to_bytes(self) -> bytes:
        return self._payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "_FakeEnvelope":
        return cls(data)


@pytest.mark.unit
class TestReassemblyGateway:
    def _make_gateway(self) -> ReassemblyGateway:
        return ReassemblyGateway(chunker=DefaultEnvelopeChunker())

    def _make_chunks(
        self,
        payload: bytes,
        chunk_size: int = 200,
    ) -> list[ModelChunkedEnvelope]:
        chunker = DefaultEnvelopeChunker()
        return chunker.chunk(_FakeEnvelope(payload), max_chunk_size=chunk_size)

    def test_partial_chunks_returns_none(self) -> None:
        """Receiving fewer than chunk_count chunks → returns None."""
        gateway = self._make_gateway()
        chunks = self._make_chunks(b"partial " * 100, chunk_size=200)
        assert len(chunks) > 1
        # Feed all but the last chunk
        result = None
        for chunk in chunks[:-1]:
            result = gateway.receive(chunk, _FakeEnvelope)
        assert result is None

    def test_all_chunks_returns_reassembled_envelope(self) -> None:
        """Receiving all chunks → returns reconstructed envelope."""
        gateway = self._make_gateway()
        payload = b"complete series " * 50
        chunks = self._make_chunks(payload, chunk_size=200)
        result = None
        for chunk in chunks:
            result = gateway.receive(chunk, _FakeEnvelope)
        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload

    def test_out_of_order_chunks_reassembled(self) -> None:
        """Out-of-order delivery → correct reassembly."""
        gateway = self._make_gateway()
        payload = b"out of order " * 60
        chunks = self._make_chunks(payload, chunk_size=150)
        shuffled = list(reversed(chunks))
        result = None
        for chunk in shuffled:
            result = gateway.receive(chunk, _FakeEnvelope)
        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload

    def test_checksum_mismatch_returns_series_failed(self) -> None:
        """Corrupted chunk payload → returns ModelChunkSeriesFailed."""
        gateway = self._make_gateway()
        payload = b"will be corrupted " * 50
        chunks = self._make_chunks(payload, chunk_size=200)
        first = chunks[0]
        bad_chunk = first.model_copy(update={"chunk_payload": b"BAD DATA"})
        bad_list = [bad_chunk] + list(chunks[1:])
        result = None
        for chunk in bad_list:
            result = gateway.receive(chunk, _FakeEnvelope)
        assert isinstance(result, ModelChunkSeriesFailed)
        assert result.reason == EnumChunkFailureReason.CHECKSUM_MISMATCH

    def test_expired_series_returns_series_failed(self) -> None:
        """Chunks arriving after expiry_timestamp → returns ModelChunkSeriesFailed."""
        gateway = self._make_gateway()
        payload = b"expired series " * 50
        chunks = self._make_chunks(payload, chunk_size=200)
        # Rewrite chunk_metadata with a past expiry timestamp
        past = datetime(2000, 1, 1, tzinfo=UTC)
        expired_chunks: list[ModelChunkedEnvelope] = []
        for chunk in chunks:
            new_meta = chunk.chunk_metadata.model_copy(
                update={"expiry_timestamp": past}
            )
            expired_chunks.append(chunk.model_copy(update={"chunk_metadata": new_meta}))

        result = None
        for chunk in expired_chunks:
            result = gateway.receive(chunk, _FakeEnvelope)
        assert isinstance(result, ModelChunkSeriesFailed)
        assert result.reason == EnumChunkFailureReason.TIMEOUT

    def test_multiple_independent_series_buffered_separately(self) -> None:
        """Two concurrent chunk series are buffered independently."""
        gateway = self._make_gateway()
        payload_a = b"series A " * 50
        payload_b = b"series B " * 50
        chunks_a = self._make_chunks(payload_a, chunk_size=200)
        chunks_b = self._make_chunks(payload_b, chunk_size=200)

        # Interleave chunks from both series
        interleaved = []
        for a, b in zip(chunks_a, chunks_b, strict=False):
            interleaved.extend([a, b])
        # Append any remainder
        for extra in chunks_a[len(chunks_b) :]:
            interleaved.append(extra)
        for extra in chunks_b[len(chunks_a) :]:
            interleaved.append(extra)

        results = []
        for chunk in interleaved:
            r = gateway.receive(chunk, _FakeEnvelope)
            if r is not None:
                results.append(r)

        assert len(results) == 2
        payloads = {r.to_bytes() for r in results}
        assert payload_a in payloads
        assert payload_b in payloads

    def test_single_chunk_series_reassembles(self) -> None:
        """Single chunk (payload below chunk size) → reassembles immediately."""
        gateway = self._make_gateway()
        payload = b"tiny"
        chunks = self._make_chunks(payload, chunk_size=1024)
        assert len(chunks) == 1
        result = gateway.receive(chunks[0], _FakeEnvelope)
        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload
