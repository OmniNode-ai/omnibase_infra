# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HTTP seam for the local-model doctor probe."""

from typing import Protocol

from omnibase_core.protocols.http.protocol_http_client import ProtocolHttpResponse


class ProtocolModelsTransport(Protocol):
    """The single OpenAI-compatible models request used by the doctor."""

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProtocolHttpResponse: ...


__all__ = ["ProtocolModelsTransport"]
