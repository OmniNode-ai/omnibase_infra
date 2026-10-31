# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TTL secret cache and its counters, held outside the handler (OMN-21536)."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelSecretCache(BaseModel):
    """Cached secrets by key with the hit, miss and fetch counters of the handler.

    The model takes no lock: the handler holds its own lock around every call.
    Storing past ``max_size`` first purges expired entries, then the entries that
    expire earliest, so the incoming entry always fits.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    max_size: int = Field(..., gt=0, description="Entries held before eviction.")
    entries: dict[str, tuple[SecretStr, float]] = Field(
        default_factory=dict, description="Secret and its monotonic expiry by key."
    )
    hits: int = Field(default=0, ge=0, description="Lookups served from the cache.")
    misses: int = Field(
        default=0, ge=0, description="Lookups the cache could not serve."
    )
    fetches: int = Field(default=0, ge=0, description="Reads sent to the secret store.")

    def lookup(self, key: str) -> SecretStr | None:
        """Return the unexpired secret for ``key`` and count a hit, else count a miss."""
        entry = self.entries.get(key)
        if entry is not None and time.monotonic() < entry[1]:
            self.hits += 1
            return entry[0]
        self.misses += 1
        return None

    def record_fetch(self) -> None:
        """Count one read sent to the secret store."""
        self.fetches += 1

    def store(self, key: str, value: SecretStr, ttl: float) -> None:
        """Keep ``value`` for ``ttl`` seconds, evicting first when the cache is full."""
        self._evict()
        self.entries[key] = (value, time.monotonic() + ttl)

    def invalidate(self, secret_name: str | None) -> int:
        """Drop every entry, or those of ``secret_name``; return how many went."""
        if secret_name is None:
            dropped = len(self.entries)
            self.entries.clear()
            return dropped
        doomed = [k for k in self.entries if k.endswith(f"::{secret_name}")]
        for key in doomed:
            del self.entries[key]
        return len(doomed)

    def discard_all(self) -> None:
        """Drop every entry; the counters stay."""
        self.entries.clear()

    def _evict(self) -> None:
        if len(self.entries) < self.max_size:
            return
        now = time.monotonic()
        for key in [
            k for k, (_, expires_at) in self.entries.items() if now >= expires_at
        ]:
            del self.entries[key]
        if len(self.entries) < self.max_size:
            return
        by_expiry = sorted(self.entries, key=lambda k: self.entries[k][1])
        for key in by_expiry[: len(self.entries) - self.max_size + 1]:
            del self.entries[key]


__all__: list[str] = ["ModelSecretCache"]
