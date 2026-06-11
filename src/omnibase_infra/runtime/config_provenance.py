# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Config provenance for runtime-rendered contracts (OMN-12958).

The Bifrost delegation contract is rendered once at container startup and written
to a Docker named volume at ``/app/data/delegation/bifrost_delegation.yaml``. The
volume survives ``docker compose build``/recreate, so the deployed (volume) config
can silently diverge from the packaged source shipped in the image. Two competing
authorities — the packaged source and the volume copy — is the defect this module
addresses.

This module computes a deterministic provenance record (path + sha256 of both the
deployed volume contract and the packaged source) and classifies drift between
them. The record is:

* logged at runtime startup (path + sha),
* exposed via the runtime health endpoint
  (``health/health_config_provenance.py``),
* written next to the deployed contract as a sidecar JSON so the sweep
  (omnimarket ``node_volume_config_drift_sweep``) and proof packets can read it
  without re-deriving the source path.

This module is **read-only** with respect to the contract content itself: it never
rewrites the deployed contract. The re-seed (overwriting a drifted volume copy from
packaged source) is an operator deploy-procedure action, not a runtime side effect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Sidecar file written next to the rendered contract. Named with a leading dot so
# it never collides with a contract-discovery glob.
PROVENANCE_SIDECAR_NAME = ".config_provenance.json"


def compute_sha256(path: Path) -> str | None:
    """Return the hex sha256 of ``path`` content, or ``None`` when absent.

    Hashes raw bytes (not parsed YAML) so the digest is byte-exact and stable
    across machines. Returns ``None`` for a missing file rather than raising,
    because an absent deployed contract is itself a valid provenance state that
    the caller classifies.
    """
    if not path.exists():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ModelConfigProvenance(BaseModel):
    """Provenance of a runtime-rendered config relative to its packaged source.

    ``deployed`` is the volume copy actually loaded by the runtime; ``source`` is
    the packaged contract shipped in the image. ``has_drifted`` is ``True`` only
    when both digests are known and differ — an absent deployed copy (first boot)
    or absent source is reported via the digests being ``None``, not as drift.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    config_name: str = Field(
        ...,
        description="Logical config identifier (e.g. 'bifrost_delegation')",
        min_length=1,
    )
    deployed_path: str = Field(
        ...,
        description="Absolute path of the deployed (volume) contract",
        min_length=1,
    )
    deployed_sha256: str | None = Field(
        default=None,
        description="sha256 of the deployed contract bytes, or None if absent",
    )
    source_path: str = Field(
        ...,
        description="Absolute path of the packaged source contract",
        min_length=1,
    )
    source_sha256: str | None = Field(
        default=None,
        description="sha256 of the packaged source contract bytes, or None if absent",
    )

    @property
    def deployed_present(self) -> bool:
        """Whether the deployed (volume) contract exists."""
        return self.deployed_sha256 is not None

    @property
    def source_present(self) -> bool:
        """Whether the packaged source contract exists."""
        return self.source_sha256 is not None

    @property
    def has_drifted(self) -> bool:
        """Whether the deployed copy diverges from packaged source.

        ``True`` only when both digests are known and differ. Unknown digests
        (missing file) are not drift — the caller inspects ``*_present``.
        """
        if self.deployed_sha256 is None or self.source_sha256 is None:
            return False
        return self.deployed_sha256 != self.source_sha256

    def provenance_line(self) -> str:
        """Single-line, log-friendly provenance summary (path + sha + drift)."""
        deployed = self.deployed_sha256 or "absent"
        source = self.source_sha256 or "absent"
        drift = "DRIFT" if self.has_drifted else "in-sync"
        return (
            f"config_provenance config={self.config_name} status={drift} "
            f"deployed_path={self.deployed_path} deployed_sha256={deployed} "
            f"source_path={self.source_path} source_sha256={source}"
        )


def build_config_provenance(
    *,
    config_name: str,
    deployed_path: Path,
    source_path: Path,
) -> ModelConfigProvenance:
    """Compute provenance for a deployed contract against its packaged source."""
    return ModelConfigProvenance(
        config_name=config_name,
        deployed_path=str(deployed_path),
        deployed_sha256=compute_sha256(deployed_path),
        source_path=str(source_path),
        source_sha256=compute_sha256(source_path),
    )


def write_provenance_sidecar(
    provenance: ModelConfigProvenance,
    *,
    deployed_path: Path,
) -> Path:
    """Write the provenance record as a JSON sidecar next to the deployed contract.

    The sidecar lets the drift sweep and proof packets read provenance without
    re-resolving the packaged source path (which depends on import-time package
    layout inside the container). Returns the sidecar path.
    """
    sidecar = deployed_path.parent / PROVENANCE_SIDECAR_NAME
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sidecar


__all__: list[str] = [
    "PROVENANCE_SIDECAR_NAME",
    "ModelConfigProvenance",
    "build_config_provenance",
    "compute_sha256",
    "write_provenance_sidecar",
]
