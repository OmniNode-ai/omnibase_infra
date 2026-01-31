# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovered Contract Model.

Internal model for contracts discovered by sources. Using a Pydantic model
instead of tuples prevents position bugs and makes error reporting cleaner.

This model is populated in stages:
1. Discovery: origin, ref, text are set
2. Parsing: handler_id is extracted from YAML
3. Hashing: content_hash (SHA-256) is computed

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ModelDiscoveredContract(BaseModel):
    """Internal model for discovered contracts.

    Represents a contract discovered by a source (filesystem or package)
    before validation and publishing. This model prevents tuple position
    bugs and provides a clear structure for error reporting.

    Population Stages:
        1. Discovery: origin, ref, text are set by the source
        2. Parsing: handler_id is extracted from YAML (may fail)
        3. Hashing: content_hash is computed for dedup/conflict detection

    Attributes:
        origin: Source type that discovered this contract
        ref: Path (filesystem) or resource path (package)
        text: Raw YAML content
        handler_id: Extracted from contract YAML, None if parsing failed
        content_hash: SHA-256 hash of text, for dedup/conflict detection

    Example:
        >>> contract = ModelDiscoveredContract(
        ...     origin="filesystem",
        ...     ref=Path("/app/contracts/foo/contract.yaml"),
        ...     text="handler_id: foo.handler\\n...",
        ... )
        >>> contract = contract.with_parsed_data(
        ...     handler_id="foo.handler",
        ... )
        >>> contract = contract.with_content_hash()

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: Literal["filesystem", "package"] = Field(
        description="Source type that discovered this contract"
    )
    ref: Path | str = Field(description="Path (filesystem) or resource path (package)")
    text: str = Field(description="Raw YAML content")
    handler_id: str | None = Field(
        default=None,
        description="Handler ID extracted from YAML, None if parsing failed",
    )
    content_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of text for dedup/conflict detection",
    )

    def with_parsed_data(self, handler_id: str) -> ModelDiscoveredContract:
        """Return new instance with handler_id populated.

        Args:
            handler_id: Handler ID extracted from contract YAML

        Returns:
            New instance with handler_id set
        """
        return self.model_copy(update={"handler_id": handler_id})

    def extract_handler_id(self) -> ModelDiscoveredContract:
        """Extract handler_id from YAML and return new instance with it populated.

        Parses the YAML text content to extract handler_id early, enabling
        proper deterministic sorting and deduplication before validation.

        Returns:
            New instance with handler_id if extraction succeeded,
            self unchanged if parsing failed (will fail validation later).
        """
        # Skip if handler_id already extracted
        if self.handler_id is not None:
            return self

        try:
            data = yaml.safe_load(self.text)
            if isinstance(data, dict) and "handler_id" in data:
                handler_id = data["handler_id"]
                if isinstance(handler_id, str) and handler_id:
                    return self.with_parsed_data(handler_id=handler_id)
        except yaml.YAMLError:
            # YAML parse errors will be caught during validation
            logger.debug(
                "Failed to extract handler_id from %s:%s (YAML parse error)",
                self.origin,
                self.ref,
            )
        return self

    def with_content_hash(self) -> ModelDiscoveredContract:
        """Return new instance with content_hash computed.

        Computes SHA-256 hash of the text content for use in
        deduplication and conflict detection.

        Returns:
            New instance with content_hash set
        """
        hash_value = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return self.model_copy(update={"content_hash": hash_value})

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash of text content.

        Args:
            text: Raw YAML content

        Returns:
            Hexadecimal SHA-256 hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sort_key(self) -> tuple[str, str, str]:
        """Return sort key for deterministic ordering.

        Sorts by (handler_id, origin, ref) to ensure consistent
        ordering across runs regardless of filesystem order.

        Returns:
            Tuple for sorting: (handler_id or "", origin, str(ref))
        """
        return (self.handler_id or "", self.origin, str(self.ref))


__all__ = ["ModelDiscoveredContract"]
