# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical root verification and inert preflight for the RSD lane overlay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeGuard
from weakref import WeakKeyDictionary

import yaml
from pydantic import BaseModel, ValidationError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.resolver import BaseResolver

from omnibase_core.crypto.crypto_ed25519_signer import verify_base64
from omnibase_core.protocols.crypto.protocol_multi_key_provider import (
    ProtocolMultiKeyProvider,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.enums.enum_rsd_live_delegation_preflight_failure import (
    EnumRsdLiveDelegationPreflightFailure,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_authority_envelope import (
    ModelRsdLiveDelegationAuthorityEnvelope,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_observation import (
    ModelRsdLiveDelegationObservation,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_overlay import (
    ModelRsdLiveDelegationOverlay,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_preflight_result import (
    ModelRsdLiveDelegationPreflightResult,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_result_anchor import (
    ModelRsdLiveDelegationResultAnchor,
)
from omnibase_infra.runtime.models.model_rsd_postgres_acceptance_overlay import (
    ModelRsdPostgresAcceptanceOverlay,
)

_AUTHORITY_DOMAIN = b"omninode-rsd.live-delegation-authority-envelope.v1\x00"
_MAX_OVERLAY_BYTES = 64 * 1024
_MAX_OVERLAY_NESTING_DEPTH = 64
_MAX_PREPARED_AUTHORITY_AGE_SECONDS = 300


class NoDuplicateSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous mappings at every depth."""

    _nesting_depth: int = 0

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        if self.check_event(AliasEvent):
            event = self.get_event()
            raise ConstructorError(
                None,
                None,
                "YAML aliases are not permitted in RSD live delegation overlays",
                event.start_mark,
            )
        if self._nesting_depth >= _MAX_OVERLAY_NESTING_DEPTH:
            raise ConstructorError(
                None,
                None,
                "YAML nesting exceeds the RSD live delegation overlay limit",
                None,
            )
        self._nesting_depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._nesting_depth -= 1


def _construct_unique_mapping(
    loader: NoDuplicateSafeLoader,
    node: yaml.MappingNode,
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


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


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


def _strict_model[ModelT: BaseModel](
    value: object, expected_type: type[ModelT]
) -> ModelT | None:
    if type(value) is not expected_type:
        return None
    try:
        dumped = value.model_dump(mode="json")
        if not _is_exact_json(dumped):
            return None
        return expected_type.model_validate_json(_canonical_json(dumped), strict=True)
    except (TypeError, ValidationError):
        return None


def _canonical_observation_bytes(
    observation: ModelRsdLiveDelegationObservation,
) -> bytes:
    """Canonicalize set-backed injected facts before binding them to a receipt."""
    payload = observation.model_dump(mode="json")
    for name in ("present_capability_refs", "healthy_capability_refs"):
        value = payload[name]
        if type(value) is list:
            payload[name] = sorted(value)
    return _canonical_json(payload)


def overlay_digest_sha256(overlay: ModelRsdLiveDelegationOverlay) -> str:
    """Hash exact overlay facts excluding the signed envelope itself."""
    checked = _strict_model(overlay, ModelRsdLiveDelegationOverlay)
    if checked is None:
        raise ProtocolConfigurationError("RSD live delegation overlay is not canonical")
    payload = checked.model_dump(mode="json", exclude={"authority_envelope"})
    if not _is_exact_json(payload):
        raise ProtocolConfigurationError("RSD live delegation overlay is not canonical")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def authority_signing_preimage(
    authority: ModelRsdLiveDelegationAuthorityEnvelope,
) -> bytes:
    """Return domain-separated canonical bytes covered by Ed25519."""
    checked = _strict_model(authority, ModelRsdLiveDelegationAuthorityEnvelope)
    if checked is None:
        raise ProtocolConfigurationError(
            "RSD live delegation authority is not canonical"
        )
    payload = checked.model_dump(mode="json", exclude={"signature_base64"})
    if not _is_exact_json(payload):
        raise ProtocolConfigurationError(
            "RSD live delegation authority is not canonical"
        )
    return _AUTHORITY_DOMAIN + _canonical_json(payload)


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class SealedRsdLiveDelegationAuthorityReceipt:
    """Opaque in-process receipt; provenance is held by preparation's closure."""

    overlay_digest_sha256: str
    attestor_key_id: str
    attestor_public_key_fingerprint_sha256: str
    observed_model_id: str
    observed_model_attestation_sha256: str
    expires_at: str

    def __init__(self) -> None:
        raise TypeError(
            "SealedRsdLiveDelegationAuthorityReceipt is minted by preparation only"
        )


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


def _resolve_attestor_key(
    key_provider: ProtocolMultiKeyProvider,
    authority: ModelRsdLiveDelegationAuthorityEnvelope,
    overlay: ModelRsdLiveDelegationOverlay,
) -> bytes:
    """Resolve and pin the actual result-attestor key before sealing."""
    try:
        resolved = key_provider.get_node_identity_key(str(authority.attestor_key_id))
    except Exception:  # noqa: BLE001 -- provider boundary is untrusted.
        raise ProtocolConfigurationError(
            "RSD live delegation result attestor is unavailable or substituted"
        ) from None
    if (
        type(resolved) is not bytes
        or len(resolved) != 32
        or hashlib.sha256(resolved).hexdigest()
        != authority.attestor_public_key_fingerprint_sha256
        or authority.attestor_key_id != overlay.result_attestor_key_id
        or authority.attestor_public_key_fingerprint_sha256
        != overlay.result_attestor_public_key_fingerprint_sha256
    ):
        raise ProtocolConfigurationError(
            "RSD live delegation result attestor is unavailable or substituted"
        )
    return resolved


def _validate_signed_authority_envelope(
    overlay: ModelRsdLiveDelegationOverlay,
    authority: object,
    key_provider: ProtocolMultiKeyProvider,
    *,
    now: datetime,
) -> ModelRsdLiveDelegationAuthorityEnvelope:
    """Revalidate every root-signed authority fact against this exact overlay."""
    checked_authority = _strict_model(
        authority, ModelRsdLiveDelegationAuthorityEnvelope
    )
    if checked_authority is None:
        raise ProtocolConfigurationError(
            "RSD live delegation authority envelope is invalid"
        )
    issued_at = _parse_utc(checked_authority.issued_at)
    expires_at = _parse_utc(checked_authority.expires_at)
    if issued_at is None or expires_at is None or issued_at > now or now >= expires_at:
        raise ProtocolConfigurationError(
            "RSD live delegation authority is outside its validity window"
        )
    if checked_authority.overlay_digest_sha256 != overlay_digest_sha256(overlay):
        raise ProtocolConfigurationError(
            "RSD live delegation authority overlay digest differs"
        )
    if (
        checked_authority.route_ref != overlay.route_ref
        or checked_authority.backend_id != overlay.backend_id
        or checked_authority.model_id != overlay.model_id
        or checked_authority.delegation_policy_schema
        != overlay.delegation_policy_schema
        or checked_authority.dispatch_outcome_schema != overlay.dispatch_outcome_schema
        or checked_authority.claim_binding_schema != overlay.claim_binding_schema
        or checked_authority.observed_model_id != overlay.observed_model_id
        or checked_authority.observed_model_attestation_sha256
        != overlay.observed_model_attestation_sha256
    ):
        raise ProtocolConfigurationError("RSD live delegation authority facts differ")
    try:
        resolved_root = key_provider.get_domain_trust_root(
            checked_authority.authority_domain
        )
    except Exception:  # noqa: BLE001 -- provider boundary is untrusted.
        raise ProtocolConfigurationError(
            "RSD live delegation root is unavailable or substituted"
        ) from None
    if (
        type(resolved_root) is not bytes
        or len(resolved_root) != 32
        or hashlib.sha256(resolved_root).hexdigest()
        != checked_authority.authority_root_public_key_fingerprint_sha256
        or checked_authority.authority_root_id != overlay.root_authority_key_id
        or checked_authority.authority_root_public_key_fingerprint_sha256
        != overlay.root_authority_public_key_fingerprint_sha256
    ):
        raise ProtocolConfigurationError(
            "RSD live delegation root is unavailable or substituted"
        )
    try:
        signature_verified = verify_base64(
            resolved_root,
            authority_signing_preimage(checked_authority),
            checked_authority.signature_base64,
        )
    except Exception:  # noqa: BLE001 -- crypto boundary is untrusted.
        raise ProtocolConfigurationError(
            "RSD live delegation authority signature is invalid"
        ) from None
    if not signature_verified:
        raise ProtocolConfigurationError(
            "RSD live delegation authority signature is invalid"
        )
    return checked_authority


def _prepare_rsd_live_delegation_authority(
    overlay: object,
    observation: object,
    key_provider: ProtocolMultiKeyProvider,
    *,
    now: datetime,
    mint_receipt: Callable[
        [ModelRsdLiveDelegationAuthorityEnvelope],
        SealedRsdLiveDelegationAuthorityReceipt,
    ],
) -> tuple[SealedRsdLiveDelegationAuthorityReceipt, ModelRsdLiveDelegationObservation]:
    """Verify provider authority once and emit a sealed receipt plus facts."""
    checked = _strict_model(overlay, ModelRsdLiveDelegationOverlay)
    checked_observation = _strict_model(observation, ModelRsdLiveDelegationObservation)
    if (
        checked is None
        or checked_observation is None
        or not isinstance(key_provider, ProtocolMultiKeyProvider)
    ):
        raise ProtocolConfigurationError(
            "RSD live delegation authority input is invalid"
        )
    if not _is_utc_datetime(now):
        raise ProtocolConfigurationError(
            "RSD live delegation authority requires a UTC clock"
        )
    authority = _validate_signed_authority_envelope(
        checked, checked.authority_envelope, key_provider, now=now
    )
    _resolve_attestor_key(key_provider, authority, checked)
    receipt = mint_receipt(authority)
    prepared = ModelRsdLiveDelegationObservation.model_validate_json(
        _canonical_json(
            checked_observation.model_dump(mode="json")
            | {
                "sealed_root_provider_verified": True,
                "authority_checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "verified_result_attestor_key_id": receipt.attestor_key_id,
                "verified_result_attestor_public_key_fingerprint_sha256": receipt.attestor_public_key_fingerprint_sha256,
                "verified_overlay_digest_sha256": receipt.overlay_digest_sha256,
            }
        ),
        strict=True,
    )
    return receipt, prepared


def _authority_preparation_boundary() -> tuple[
    Callable[
        ...,
        tuple[
            SealedRsdLiveDelegationAuthorityReceipt, ModelRsdLiveDelegationObservation
        ],
    ],
    Callable[[object, bytes], TypeGuard[SealedRsdLiveDelegationAuthorityReceipt]],
]:
    """Create a same-process receipt boundary for verified authority facts.

    The closure prevents ordinary imports from minting or registering a
    receipt. This is not an isolation boundary against deliberate Python
    closure inspection, monkeypatching, or native-process introspection.
    """
    observation_bytes: WeakKeyDictionary[
        SealedRsdLiveDelegationAuthorityReceipt, bytes
    ] = WeakKeyDictionary()

    def mint(
        authority: ModelRsdLiveDelegationAuthorityEnvelope,
    ) -> SealedRsdLiveDelegationAuthorityReceipt:
        receipt = object.__new__(SealedRsdLiveDelegationAuthorityReceipt)
        object.__setattr__(
            receipt, "overlay_digest_sha256", authority.overlay_digest_sha256
        )
        object.__setattr__(receipt, "attestor_key_id", str(authority.attestor_key_id))
        object.__setattr__(
            receipt,
            "attestor_public_key_fingerprint_sha256",
            authority.attestor_public_key_fingerprint_sha256,
        )
        object.__setattr__(receipt, "observed_model_id", authority.observed_model_id)
        object.__setattr__(
            receipt,
            "observed_model_attestation_sha256",
            authority.observed_model_attestation_sha256,
        )
        object.__setattr__(receipt, "expires_at", authority.expires_at)
        return receipt

    def prepare(
        overlay: object,
        observation: object,
        key_provider: ProtocolMultiKeyProvider,
        *,
        now: datetime,
    ) -> tuple[
        SealedRsdLiveDelegationAuthorityReceipt, ModelRsdLiveDelegationObservation
    ]:
        receipt, prepared = _prepare_rsd_live_delegation_authority(
            overlay, observation, key_provider, now=now, mint_receipt=mint
        )
        observation_bytes[receipt] = _canonical_observation_bytes(prepared)
        return receipt, prepared

    def contains(
        value: object, expected_observation: bytes
    ) -> TypeGuard[SealedRsdLiveDelegationAuthorityReceipt]:
        return (
            type(value) is SealedRsdLiveDelegationAuthorityReceipt
            and observation_bytes.get(value) == expected_observation
        )

    return prepare, contains


prepare_rsd_live_delegation_authority, _is_sealed_receipt = (
    _authority_preparation_boundary()
)


def load_rsd_live_delegation_overlay(path: Path) -> ModelRsdLiveDelegationOverlay:
    try:
        if path.stat().st_size > _MAX_OVERLAY_BYTES:
            raise ProtocolConfigurationError("RSD live delegation overlay is invalid")
        raw = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=NoDuplicateSafeLoader,  # noqa: S506
        )
    except (
        MemoryError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ):
        raise ProtocolConfigurationError(
            "RSD live delegation overlay is invalid"
        ) from None
    if type(raw) is not dict:
        raise ProtocolConfigurationError("RSD live delegation overlay is invalid")
    if not _is_exact_json(raw):
        raise ProtocolConfigurationError("RSD live delegation overlay is invalid")
    try:
        return ModelRsdLiveDelegationOverlay.model_validate_json(
            _canonical_json(raw), strict=True
        )
    except ValidationError:
        raise ProtocolConfigurationError(
            "RSD live delegation overlay is invalid"
        ) from None


def preflight_rsd_live_delegation_overlay(
    overlay: object,
    observation: object,
    postgres_acceptance_overlay: object,
    sealed_authority: object,
    *,
    now: datetime,
) -> ModelRsdLiveDelegationPreflightResult:
    """Purely compare a sealed receipt, facts, and static overlay; never resolve."""
    checked_overlay = _strict_model(overlay, ModelRsdLiveDelegationOverlay)
    checked_observation = _strict_model(observation, ModelRsdLiveDelegationObservation)
    checked_postgres = _strict_model(
        postgres_acceptance_overlay, ModelRsdPostgresAcceptanceOverlay
    )
    failures: list[EnumRsdLiveDelegationPreflightFailure] = [
        EnumRsdLiveDelegationPreflightFailure.EXECUTION_DISABLED
    ]
    if (
        checked_overlay is None
        or checked_observation is None
        or checked_postgres is None
        or not _is_utc_datetime(now)
    ):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        )
        return ModelRsdLiveDelegationPreflightResult(failures=tuple(failures))
    if (
        checked_observation.installed_public_rsd_revision_sha
        != checked_overlay.public_rsd_revision_sha
    ):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.PUBLIC_RSD_REVISION_MISMATCH
        )
    required_refs = frozenset(
        (
            checked_overlay.one_shot_endpoint_capability_ref,
            checked_overlay.root_authority_capability_ref,
            checked_overlay.result_attestor_signer_capability_ref,
            checked_overlay.result_attestor_key_capability_ref,
            checked_overlay.result_attestor_fingerprint_capability_ref,
            checked_overlay.postgres_capability_ref,
            checked_overlay.observed_model_attestation_capability_ref,
        )
    )
    missing = tuple(sorted(required_refs - checked_observation.present_capability_refs))
    if missing:
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.CAPABILITY_REFERENCE_MISSING
        )
    if required_refs - checked_observation.healthy_capability_refs:
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.CAPABILITY_HEALTH_UNVERIFIED
        )
    if (
        checked_postgres.postgres_capability_ref
        != checked_overlay.postgres_capability_ref
        or checked_postgres.lane != checked_overlay.lane
        or checked_postgres.locale != checked_overlay.locale
        or checked_postgres.rsd_distribution_ref != checked_overlay.rsd_distribution_ref
    ):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.POSTGRES_ACCEPTANCE_BINDING_MISMATCH
        )
    canonical_observation = _canonical_observation_bytes(checked_observation)
    if not _is_sealed_receipt(sealed_authority, canonical_observation):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        )
        return ModelRsdLiveDelegationPreflightResult(
            failures=tuple(failures), missing_capability_refs=missing
        )
    checked_at = (
        _parse_utc(checked_observation.authority_checked_at)
        if checked_observation.authority_checked_at is not None
        else None
    )
    expires_at = _parse_utc(sealed_authority.expires_at)
    if (
        checked_at is None
        or checked_at > now
        or (now - checked_at).total_seconds() > _MAX_PREPARED_AUTHORITY_AGE_SECONDS
        or expires_at is None
        or now >= expires_at
    ):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        )
        return ModelRsdLiveDelegationPreflightResult(
            failures=tuple(failures), missing_capability_refs=missing
        )
    if (
        checked_observation.sealed_root_provider_verified is not True
        or sealed_authority.overlay_digest_sha256
        != overlay_digest_sha256(checked_overlay)
        or checked_observation.verified_overlay_digest_sha256
        != sealed_authority.overlay_digest_sha256
        or sealed_authority.attestor_key_id
        != str(checked_overlay.result_attestor_key_id)
        or checked_observation.verified_result_attestor_key_id
        != checked_overlay.result_attestor_key_id
        or sealed_authority.attestor_public_key_fingerprint_sha256
        != checked_overlay.result_attestor_public_key_fingerprint_sha256
        or checked_observation.verified_result_attestor_public_key_fingerprint_sha256
        != sealed_authority.attestor_public_key_fingerprint_sha256
        or sealed_authority.observed_model_id != checked_overlay.observed_model_id
        or checked_observation.observed_model_id != checked_overlay.observed_model_id
        or sealed_authority.observed_model_attestation_sha256
        != checked_overlay.observed_model_attestation_sha256
        or checked_observation.observed_model_attestation_sha256
        != checked_overlay.observed_model_attestation_sha256
        or checked_observation.observed_model_match_status
        != checked_overlay.model_match_status
    ):
        failures.append(
            EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        )
        return ModelRsdLiveDelegationPreflightResult(
            failures=tuple(failures), missing_capability_refs=missing
        )
    failures.append(EnumRsdLiveDelegationPreflightFailure.OBSERVED_MODEL_ID_MISMATCH)
    failures.append(EnumRsdLiveDelegationPreflightFailure.ACTIVATION_PATH_UNIMPLEMENTED)
    return ModelRsdLiveDelegationPreflightResult(
        failures=tuple(failures),
        missing_capability_refs=missing,
        result_anchor=ModelRsdLiveDelegationResultAnchor.model_validate_json(
            _canonical_json(
                {
                    "signer_key_id": sealed_authority.attestor_key_id,
                    "signer_public_key_fingerprint_sha256": sealed_authority.attestor_public_key_fingerprint_sha256,
                    "dispatch_outcome_schema": "rsd.dispatch-outcome-attestation.v1",
                    "claim_binding_schema": "rsd.delegation-claim-binding.v1",
                }
            ),
            strict=True,
        ),
    )


__all__ = [
    "EnumRsdLiveDelegationPreflightFailure",
    "ModelRsdLiveDelegationPreflightResult",
    "ModelRsdLiveDelegationResultAnchor",
    "SealedRsdLiveDelegationAuthorityReceipt",
    "authority_signing_preimage",
    "load_rsd_live_delegation_overlay",
    "overlay_digest_sha256",
    "preflight_rsd_live_delegation_overlay",
    "prepare_rsd_live_delegation_authority",
]
