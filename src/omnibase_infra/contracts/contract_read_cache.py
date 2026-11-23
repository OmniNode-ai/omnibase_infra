# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stat-keyed memo of parsed contract YAML and contract content hashes (OMN-26359).

One runtime start read and parsed each contract file eight to twelve times:
discovery, the applier loop, handler preparation, the subscribe interleave and the
per-topic subscribe reads all go through :func:`read_contract_yaml`. Every one of
those reads sat on the event loop before ``/health`` could be answered.

:class:`ContractReadCache` keys an entry by the file's path and by its
``(st_mtime_ns, st_size)``. A file that is written again changes that key, so the
next reader parses it again; an unchanged file is parsed once per process.

The parsed object is stored as pickled bytes and every hit unpickles a fresh copy.
Callers mutate what they read (several overlay environment values into the mapping),
so sharing one mutable object would let one reader change what the next one sees.
Failures are never stored: an absent, unreadable or malformed file raises on every
read exactly as it did before.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

__all__ = ["ContractReadCache"]

_StatKey = tuple[int, int]


class ContractReadCache:
    """Process-wide memo of one parse and one content hash per contract file."""

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _parsed: ClassVar[dict[str, tuple[_StatKey, bytes]]] = {}
    _hashed: ClassVar[dict[str, tuple[_StatKey, str]]] = {}
    _parse_runs: ClassVar[list[int]] = [0]

    @staticmethod
    def _stat_key(contract_path: Path) -> tuple[str, _StatKey]:
        stat = contract_path.stat()
        return str(contract_path), (stat.st_mtime_ns, stat.st_size)

    @classmethod
    def read_parsed(
        cls, contract_path: Path, parse: Callable[[Path], object]
    ) -> object:
        """Return a private copy of the parsed contract, parsing only when changed.

        ``parse`` is called with the path and returns the parsed object; its errors
        propagate and nothing is stored for them.
        """
        contract_path = Path(contract_path)
        key, stat_key = cls._stat_key(contract_path)
        with cls._lock:
            hit = cls._parsed.get(key)
        if hit is not None and hit[0] == stat_key:
            return pickle.loads(hit[1])
        raw = parse(contract_path)
        payload = pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL)
        with cls._lock:
            cls._parse_runs[0] += 1
            cls._parsed[key] = (stat_key, payload)
        return pickle.loads(payload)

    @classmethod
    def read_hash(cls, contract_path: Path, normalise: Callable[[bytes], bytes]) -> str:
        """Return the SHA-256 of the normalised file bytes, hashing only on change.

        ``normalise`` is the canonical-form step applied to the bytes as read.
        """
        contract_path = Path(contract_path)
        key, stat_key = cls._stat_key(contract_path)
        with cls._lock:
            hit = cls._hashed.get(key)
        if hit is not None and hit[0] == stat_key:
            return hit[1]
        value = hashlib.sha256(normalise(contract_path.read_bytes())).hexdigest()
        with cls._lock:
            cls._hashed[key] = (stat_key, value)
        return value

    @classmethod
    def parse_runs(cls) -> int:
        """How many parses have actually run in this process."""
        return cls._parse_runs[0]

    @classmethod
    def clear(cls) -> None:
        """Forget every stored parse and hash (tests that rewrite files)."""
        with cls._lock:
            cls._parsed.clear()
            cls._hashed.clear()
