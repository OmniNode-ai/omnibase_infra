# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded parsing and provider-backed verification for model provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from omnibase_core.crypto.crypto_ed25519_signer import verify_base64
from omnibase_core.protocols.crypto.protocol_multi_key_provider import (
    ProtocolMultiKeyProvider,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_rsd_model_artifact_provenance import (
    ModelRsdModelArtifactProvenance,
)

_AUTHORITY_DOMAIN = b"omninode-rsd.model-artifact-provenance.v2\x00"
_MAX_PROVENANCE_BYTES = 64 * 1024
_MAX_NESTING_DEPTH = 32
_MAX_VALIDITY_WINDOW_SECONDS = 300


class NoDuplicateSafeLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting aliases, duplicate keys, and deep nesting."""

    _nesting_depth: int = 0

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        if self.check_event(AliasEvent):
            event = self.get_event()
            raise ConstructorError(
                None,
                None,
                "YAML aliases are not permitted in model provenance",
                event.start_mark,
            )
        if self._nesting_depth >= _MAX_NESTING_DEPTH:
            raise ConstructorError(
                None, None, "YAML nesting exceeds the model provenance limit", None
            )
        self._nesting_depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._nesting_depth -= 1


def _construct_unique_mapping(
    loader: NoDuplicateSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key})",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


NoDuplicateSafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _is_exact_json(value: object) -> bool:
    if type(value) in (str, int, float, bool) or value is None:
        return True
    if type(value) is list:
        return all(_is_exact_json(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _is_exact_json(item) for key, item in value.items()
        )
    return False


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _strict_model(value: object) -> ModelRsdModelArtifactProvenance | None:
    if type(value) is not ModelRsdModelArtifactProvenance:
        return None
    try:
        dumped = value.model_dump(mode="json")
        if not _is_exact_json(dumped):
            return None
        return ModelRsdModelArtifactProvenance.model_validate_json(
            _canonical_json(dumped), strict=True
        )
    except (TypeError, ValidationError):
        return None


def load_rsd_model_artifact_provenance(
    path: Path,
) -> ModelRsdModelArtifactProvenance:
    """Load one bounded provenance document without filesystem fallbacks."""
    try:
        with path.open("rb") as stream:
            raw_bytes = stream.read(_MAX_PROVENANCE_BYTES + 1)
        if len(raw_bytes) > _MAX_PROVENANCE_BYTES:
            raise ProtocolConfigurationError("RSD model artifact provenance is invalid")
        raw_text = raw_bytes.decode("utf-8")
        raw = yaml.load(
            raw_text,
            Loader=NoDuplicateSafeLoader,  # noqa: S506
        )
    except (
        MemoryError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance is invalid"
        ) from None
    if type(raw) is not dict or not _is_exact_json(raw):
        raise ProtocolConfigurationError("RSD model artifact provenance is invalid")
    try:
        return ModelRsdModelArtifactProvenance.model_validate_json(
            _canonical_json(raw), strict=True
        )
    except (TypeError, ValidationError):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance is invalid"
        ) from None


def provenance_signing_preimage(
    provenance: ModelRsdModelArtifactProvenance,
) -> bytes:
    """Return the domain-separated bytes covered by the Ed25519 signature."""
    checked = _strict_model(provenance)
    if checked is None:
        raise ProtocolConfigurationError("RSD model artifact provenance is invalid")
    payload = checked.model_dump(mode="json", exclude={"signature_base64"})
    if not _is_exact_json(payload):
        raise ProtocolConfigurationError("RSD model artifact provenance is invalid")
    return _AUTHORITY_DOMAIN + _canonical_json(payload)


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _is_utc_datetime(value: object) -> bool:
    return (
        type(value) is datetime
        and value.tzinfo is not None
        and value.utcoffset() == UTC.utcoffset(value)
    )


def validate_rsd_model_artifact_provenance(
    provenance: object,
    *,
    now: datetime,
) -> ModelRsdModelArtifactProvenance:
    """Validate the unapproved contract and injected-clock freshness window."""
    checked = _strict_model(provenance)
    if checked is None:
        raise ProtocolConfigurationError("RSD model artifact provenance is invalid")
    if not _is_utc_datetime(now):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance requires a UTC clock"
        )
    issued_at = _parse_utc(checked.issued_at)
    expires_at = _parse_utc(checked.expires_at)
    if issued_at is None or expires_at is None:
        raise ProtocolConfigurationError(
            "RSD model artifact provenance is outside its validity window"
        )
    validity_seconds = (expires_at - issued_at).total_seconds()
    if (
        issued_at > now
        or now >= expires_at
        or validity_seconds <= 0
        or validity_seconds > _MAX_VALIDITY_WINDOW_SECONDS
    ):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance is outside its validity window"
        )
    return checked


def verify_rsd_model_artifact_provenance(
    provenance: object,
    key_provider: ProtocolMultiKeyProvider,
    *,
    now: datetime,
) -> ModelRsdModelArtifactProvenance:
    """Verify the exact contract with an injected public-key provider."""
    checked = validate_rsd_model_artifact_provenance(provenance, now=now)
    if not isinstance(key_provider, ProtocolMultiKeyProvider):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance authority is unavailable or substituted"
        )
    try:
        public_key = key_provider.get_node_identity_key(str(checked.signer_key_id))
    except Exception:  # noqa: BLE001 -- provider boundary is untrusted.
        raise ProtocolConfigurationError(
            "RSD model artifact provenance authority is unavailable or substituted"
        ) from None
    if (
        type(public_key) is not bytes
        or len(public_key) != 32
        or hashlib.sha256(public_key).hexdigest()
        != checked.signer_public_key_fingerprint_sha256
    ):
        raise ProtocolConfigurationError(
            "RSD model artifact provenance authority is unavailable or substituted"
        )
    try:
        valid = verify_base64(
            public_key,
            provenance_signing_preimage(checked),
            checked.signature_base64,
        )
    except Exception:  # noqa: BLE001 -- crypto boundary is untrusted.
        raise ProtocolConfigurationError(
            "RSD model artifact provenance signature is invalid"
        ) from None
    if not valid:
        raise ProtocolConfigurationError(
            "RSD model artifact provenance signature is invalid"
        )
    return checked


__all__ = [
    "load_rsd_model_artifact_provenance",
    "provenance_signing_preimage",
    "validate_rsd_model_artifact_provenance",
    "verify_rsd_model_artifact_provenance",
]
