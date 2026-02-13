# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry Fingerprint Model.

Frozen Pydantic model representing a deterministic SHA-256 fingerprint of
all event registrations in an ``EventRegistry``.  The fingerprint is used as
a startup hard gate: if the live registry does not match the expected
manifest artifact, the process must terminate before emitting any events.

Related:
    - OMN-2088: Handshake hardening -- Event registry fingerprint + startup assertion
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint_element import (
    ModelEventRegistryFingerprintElement,
)


class ModelEventRegistryFingerprint(BaseModel):
    """Deterministic fingerprint manifest for an entire event registry.

    Contains the overall SHA-256 fingerprint plus per-element breakdown
    for diff computation when a mismatch is detected.

    Attributes:
        version: Manifest format version (always 1 for this schema).
        fingerprint_sha256: Overall SHA-256 hex digest of all element hashes.
        elements: Tuple of per-registration fingerprint elements.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(default=1, description="Manifest format version")
    fingerprint_sha256: str = Field(
        ..., description="Overall SHA-256 hex digest of all element hashes"
    )
    elements: tuple[ModelEventRegistryFingerprintElement, ...] = Field(
        ..., description="Per-registration fingerprint elements"
    )

    @classmethod
    def from_json_path(cls, path: Path) -> ModelEventRegistryFingerprint:
        """Load a fingerprint manifest from a JSON artifact file.

        Args:
            path: Path to the JSON artifact file.

        Returns:
            Parsed ``ModelEventRegistryFingerprint`` instance.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
            pydantic.ValidationError: If the JSON does not match the schema.
        """
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return cls.model_validate(data)

    def to_json(self, path: Path) -> None:
        """Write the fingerprint manifest to a JSON artifact file.

        Uses sorted keys and 2-space indentation for human-readable output.

        Args:
            path: Destination path for the JSON artifact.
        """
        # model_dump_json converts tuples to lists for JSON serialization
        serializable = json.loads(self.model_dump_json())
        path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__: list[str] = [
    "ModelEventRegistryFingerprint",
    "ModelEventRegistryFingerprintElement",
]
