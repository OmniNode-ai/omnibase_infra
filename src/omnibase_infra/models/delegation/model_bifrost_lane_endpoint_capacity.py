# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Capacity of one unified-KV-pool model endpoint a lane overlay binds (OMN-20490).

A llama.cpp server started with ``--kv-unified --parallel N`` holds ONE token
pool for all N slots, and ``/props`` and ``/slots`` report that pool as every
slot's ``n_ctx``. A request is therefore not bounded by the ``n_ctx`` it reads
back: when every slot is busy, a request that grows past ``pool / N`` tokens
(prompt plus generated output) exhausts the pool and the server answers HTTP
500 ``Context size has been exceeded``. The fair per-slot share is the only
bound that holds at full concurrency, so it is what a binding's ``max_tokens``
must be sized against.

This model declares the three numbers that bound is computed from, so that no
test or renderer restates them: the pool, the slot count and the prompt
headroom reserved inside one slot's share. The overlay file is the authority on
them, the same way it is on every endpoint and ceiling it binds.
"""

from __future__ import annotations

from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEFAULT_PORTS = {"http": 80, "https": 443}


class ModelBifrostLaneEndpointCapacity(BaseModel):
    """The token capacity of one unified-pool endpoint, keyed by ``host:port``."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    #: ``host:port`` of the model server, exactly as ``authority_of`` renders a
    #: binding's ``endpoint_url``. One entry covers every backend bound to it.
    endpoint: str = Field(min_length=1)
    #: The server's whole token pool (``-c`` under ``--kv-unified``).
    pool_tokens: int = Field(gt=0)
    #: The server's concurrent slot count (``--parallel``).
    parallel_slots: int = Field(gt=0)
    #: Tokens of one slot's share reserved for the PROMPT. What is left is the
    #: most a binding's ``max_tokens`` may ask for.
    prompt_headroom_tokens: int = Field(gt=0)

    @property
    def per_slot_tokens(self) -> int:
        """The fair share of the pool one request keeps at full concurrency."""
        return self.pool_tokens // self.parallel_slots

    @property
    def max_output_tokens(self) -> int:
        """The largest ``max_tokens`` that fits the share beside the headroom."""
        return self.per_slot_tokens - self.prompt_headroom_tokens

    @staticmethod
    def authority_of(endpoint_url: str) -> str | None:
        """The ``host:port`` of a complete endpoint URL, or None if it has no host."""
        parsed = urlsplit(endpoint_url)
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme)
        return f"{host}:{port}" if port is not None else None

    def serves(self, endpoint_url: str) -> bool:
        """Whether ``endpoint_url`` is a URL on this capacity's server."""
        return self.authority_of(endpoint_url) == self.endpoint

    @model_validator(mode="after")
    def _validate_capacity(self) -> Self:
        host, separator, port = self.endpoint.rpartition(":")
        if not separator or not host or not port.isdecimal() or int(port) == 0:
            raise ValueError(
                f"endpoint {self.endpoint!r} must be host:port with a numeric port"
            )
        if self.pool_tokens < self.parallel_slots:
            raise ValueError(
                f"endpoint {self.endpoint!r}: pool_tokens {self.pool_tokens} "
                f"cannot be split across {self.parallel_slots} slots"
            )
        if self.prompt_headroom_tokens >= self.per_slot_tokens:
            raise ValueError(
                f"endpoint {self.endpoint!r}: prompt_headroom_tokens "
                f"{self.prompt_headroom_tokens} leaves no output room in a "
                f"per-slot share of {self.per_slot_tokens}"
            )
        return self


__all__ = ["ModelBifrostLaneEndpointCapacity"]
