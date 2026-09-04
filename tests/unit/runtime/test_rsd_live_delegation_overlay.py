# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hostile coverage for root-authorized, inert RSD live delegation."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

import omnibase_infra.runtime.rsd_live_delegation_overlay as live_delegation_overlay
from omnibase_infra.errors import ProtocolConfigurationError
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
from omnibase_infra.runtime.rsd_live_delegation_overlay import (
    EnumRsdLiveDelegationPreflightFailure,
    SealedRsdLiveDelegationAuthorityReceipt,
    authority_signing_preimage,
    load_rsd_live_delegation_overlay,
    overlay_digest_sha256,
    preflight_rsd_live_delegation_overlay,
    prepare_rsd_live_delegation_authority,
)

_ROOT = Path(__file__).parents[3]
_OVERLAY_PATH = _ROOT / "docker/lane-overlays/dev.rsd-live-delegation.yaml"
_POSTGRES_OVERLAY_PATH = _ROOT / "docker/lane-overlays/dev.rsd-postgres-acceptance.yaml"
_NOW = datetime(2026, 9, 4, tzinfo=UTC)
_STATIC_ROOT = base64.urlsafe_b64decode("H2zlhxa8wHzcxiYxsm5pPkwnd-KQP7qhF8xlXyLNqMU=")
_STATIC_ATTESTOR = base64.urlsafe_b64decode(
    "Kay64UG8yvCyLhqU000LxzYeUm0L_hLIl5S8kyKWbdc="
)


class _KeyProvider:
    def __init__(self, roots: dict[str, bytes], nodes: dict[str, bytes]) -> None:
        self._roots = roots
        self._nodes = nodes

    def get_domain_trust_root(self, domain_id: str) -> bytes | None:
        return self._roots.get(domain_id)

    def get_node_identity_key(self, node_id: str) -> bytes | None:
        return self._nodes.get(node_id)


def _overlay() -> ModelRsdLiveDelegationOverlay:
    return load_rsd_live_delegation_overlay(_OVERLAY_PATH)


def _postgres_overlay() -> ModelRsdPostgresAcceptanceOverlay:
    raw = yaml.safe_load(_POSTGRES_OVERLAY_PATH.read_text(encoding="utf-8"))
    return ModelRsdPostgresAcceptanceOverlay.model_validate(raw, strict=True)


def _observation(
    overlay: ModelRsdLiveDelegationOverlay,
) -> ModelRsdLiveDelegationObservation:
    refs = frozenset(
        (
            overlay.one_shot_endpoint_capability_ref,
            overlay.root_authority_capability_ref,
            overlay.result_attestor_signer_capability_ref,
            overlay.result_attestor_key_capability_ref,
            overlay.result_attestor_fingerprint_capability_ref,
            overlay.postgres_capability_ref,
            overlay.observed_model_attestation_capability_ref,
        )
    )
    return ModelRsdLiveDelegationObservation(
        installed_public_rsd_revision_sha=overlay.public_rsd_revision_sha,
        present_capability_refs=refs,
        healthy_capability_refs=refs,
        observed_model_id=overlay.observed_model_id,
        observed_model_attestation_sha256=overlay.observed_model_attestation_sha256,
        observed_model_match_status=overlay.model_match_status,
    )


def _static_provider(overlay: ModelRsdLiveDelegationOverlay) -> _KeyProvider:
    return _KeyProvider(
        {overlay.authority_envelope.authority_domain: _STATIC_ROOT},
        {str(overlay.result_attestor_key_id): _STATIC_ATTESTOR},
    )


def _signed_overlay() -> tuple[ModelRsdLiveDelegationOverlay, _KeyProvider]:
    """Create test-only keys; only non-secret fingerprints enter the overlay."""
    template = _overlay().model_dump(mode="json")
    root_private = Ed25519PrivateKey.generate()
    root = root_private.public_key().public_bytes_raw()
    attestor = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    root_fingerprint = hashlib.sha256(root).hexdigest()
    attestor_fingerprint = hashlib.sha256(attestor).hexdigest()
    template["root_authority_public_key_fingerprint_sha256"] = root_fingerprint
    template["result_attestor_public_key_fingerprint_sha256"] = attestor_fingerprint
    authority = dict(template["authority_envelope"])
    authority["authority_root_public_key_fingerprint_sha256"] = root_fingerprint
    authority["attestor_public_key_fingerprint_sha256"] = attestor_fingerprint
    authority["signature_base64"] = base64.urlsafe_b64encode(b"\x00" * 64).decode(
        "ascii"
    )
    template["authority_envelope"] = authority
    authority["overlay_digest_sha256"] = overlay_digest_sha256(
        ModelRsdLiveDelegationOverlay.model_validate_json(
            json.dumps(template, sort_keys=True), strict=True
        )
    )
    unsigned = ModelRsdLiveDelegationAuthorityEnvelope.model_validate_json(
        json.dumps(authority, sort_keys=True), strict=True
    )
    authority["signature_base64"] = base64.urlsafe_b64encode(
        root_private.sign(authority_signing_preimage(unsigned))
    ).decode("ascii")
    overlay = ModelRsdLiveDelegationOverlay.model_validate_json(
        json.dumps(template | {"authority_envelope": authority}, sort_keys=True),
        strict=True,
    )
    return overlay, _KeyProvider(
        {authority["authority_domain"]: root},
        {str(overlay.result_attestor_key_id): attestor},
    )


def _prepare(
    overlay: ModelRsdLiveDelegationOverlay, provider: _KeyProvider
) -> tuple[SealedRsdLiveDelegationAuthorityReceipt, ModelRsdLiveDelegationObservation]:
    return prepare_rsd_live_delegation_authority(
        overlay, _observation(overlay), provider, now=_NOW
    )


@pytest.mark.unit
def test_overlay_is_inert_pinned_and_contains_no_key_bytes_or_topology() -> None:
    overlay = _overlay()
    text = _OVERLAY_PATH.read_text(encoding="utf-8")
    assert overlay.execute_enabled is False
    assert overlay.model_match_status == "model_id_mismatch"
    assert "issuer_public_key" not in text
    assert "public_key_base64" not in text
    forbidden = re.compile(
        r"(?i)(?:https?://|(?:\d{1,3}\.){3}\d{1,3}|(?:postgres(?:ql)?|mysql)://)"
    )
    assert forbidden.search(text) is None


@pytest.mark.unit
def test_static_overlay_uses_stable_canonical_json_digest_and_uuid4_key_ids() -> None:
    overlay = _overlay()
    assert overlay_digest_sha256(overlay) == (
        "ee7a3fed3212c8dbdd387b5dcfd776b14e40c3757102cbc693a7e464b9455fbc"
    )
    assert overlay.root_authority_key_id.version == 4
    assert overlay.result_attestor_key_id.version == 4
    assert overlay.authority_envelope.authority_root_id.version == 4
    assert overlay.authority_envelope.attestor_key_id.version == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_key_id",
    [
        "not-a-uuid",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "{aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa}",
        "urn:uuid:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa",
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        ("root_authority_key_id",),
        ("result_attestor_key_id",),
        ("authority_envelope", "authority_root_id"),
        ("authority_envelope", "attestor_key_id"),
    ],
)
def test_key_ids_reject_noncanonical_uuid4_json_values(
    invalid_key_id: str, path: tuple[str, ...]
) -> None:
    candidate = _overlay().model_dump(mode="json")
    target: dict[str, object] = candidate
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = invalid_key_id
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationOverlay.model_validate_json(
            json.dumps(candidate, sort_keys=True), strict=True
        )


@pytest.mark.unit
def test_result_anchor_and_missing_refs_are_strictly_canonical() -> None:
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationResultAnchor.model_validate_json(
            json.dumps(
                {
                    "signer_key_id": "{aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa}",
                    "signer_public_key_fingerprint_sha256": "A" * 64,
                    "dispatch_outcome_schema": "forged.v1",
                    "claim_binding_schema": "forged.v1",
                }
            ),
            strict=True,
        )
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationPreflightResult(
            missing_capability_refs=("https://invalid.example",)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "signature",
    [
        base64.urlsafe_b64encode(b"x" * 63).decode("ascii"),
        base64.urlsafe_b64encode(b"\x00" * 64).decode("ascii")[:-3] + "B==",
        "a" * 86,
    ],
)
def test_authority_signature_rejects_noncanonical_or_wrong_length_values(
    signature: str,
) -> None:
    candidate = _overlay().model_dump(mode="json")
    candidate["authority_envelope"]["signature_base64"] = signature
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationOverlay.model_validate_json(
            json.dumps(candidate, sort_keys=True), strict=True
        )


@pytest.mark.unit
@pytest.mark.parametrize("domain", ["https://invalid.example", "192.0.2.1", "lab-host"])
def test_authority_domain_rejects_url_ip_and_host_like_inputs(domain: str) -> None:
    candidate = _overlay().model_dump(mode="python")
    candidate["authority_envelope"]["authority_domain"] = domain
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationOverlay.model_validate(candidate, strict=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference",
    [
        "capability://192.0.2.1/path",
        "capability://rsd/../other",
        "capability://rsd/./other",
        "capability://rsd//other",
        "capability://rsd:8000/other",
        "capability://rsd/with space",
        "https://invalid.example",
    ],
)
def test_capability_references_reject_noncanonical_or_topological_values(
    reference: str,
) -> None:
    overlay = _overlay()
    candidate = overlay.model_dump(mode="python")
    candidate["one_shot_endpoint_capability_ref"] = reference
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationOverlay.model_validate(candidate, strict=True)
    with pytest.raises(ValidationError):
        ModelRsdLiveDelegationObservation(
            installed_public_rsd_revision_sha=overlay.public_rsd_revision_sha,
            present_capability_refs=frozenset((reference,)),
        )


@pytest.mark.unit
def test_static_authority_derives_anchor_only_from_injected_provider_keys() -> None:
    overlay = _overlay()
    provider = _static_provider(overlay)
    sealed, prepared = _prepare(overlay, provider)
    result = preflight_rsd_live_delegation_overlay(
        overlay,
        prepared,
        _postgres_overlay(),
        sealed,
        now=_NOW,
    )
    assert result.ready is False
    assert result.result_anchor is not None
    assert result.result_anchor.signer_key_id == overlay.result_attestor_key_id
    assert result.failures == (
        EnumRsdLiveDelegationPreflightFailure.EXECUTION_DISABLED,
        EnumRsdLiveDelegationPreflightFailure.OBSERVED_MODEL_ID_MISMATCH,
        EnumRsdLiveDelegationPreflightFailure.ACTIVATION_PATH_UNIMPLEMENTED,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "route_ref",
        "backend_id",
        "model_id",
        "overlay_digest_sha256",
        "delegation_policy_schema",
    ],
)
def test_verifier_fails_closed_for_forged_semantic_authority(field: str) -> None:
    overlay, provider = _signed_overlay()
    value = "d" * 64 if field == "overlay_digest_sha256" else "forged"
    authority = ModelRsdLiveDelegationAuthorityEnvelope.model_construct(
        **(overlay.authority_envelope.model_dump(mode="python") | {field: value})
    )
    candidate = ModelRsdLiveDelegationOverlay.model_construct(
        **(overlay.model_dump(mode="python") | {"authority_envelope": authority})
    )
    with pytest.raises(ProtocolConfigurationError):
        _prepare(candidate, provider)


@pytest.mark.unit
def test_root_and_attestor_substitution_and_self_sign_fail() -> None:
    overlay, provider = _signed_overlay()
    attacker = Ed25519PrivateKey.generate()
    attacker_root = attacker.public_key().public_bytes_raw()
    with pytest.raises(
        ProtocolConfigurationError, match="root is unavailable or substituted"
    ):
        _prepare(
            overlay,
            _KeyProvider(
                {overlay.authority_envelope.authority_domain: attacker_root},
                provider._nodes,
            ),
        )
    with pytest.raises(ProtocolConfigurationError, match="result attestor"):
        _prepare(
            overlay,
            _KeyProvider(
                provider._roots, {str(overlay.result_attestor_key_id): attacker_root}
            ),
        )
    forged = overlay.model_copy(
        update={
            "authority_envelope": overlay.authority_envelope.model_copy(
                update={
                    "signature_base64": base64.urlsafe_b64encode(b"x" * 64).decode(
                        "ascii"
                    )
                }
            )
        }
    )
    with pytest.raises(ProtocolConfigurationError, match="signature is invalid"):
        _prepare(forged, provider)


@pytest.mark.unit
def test_pure_preflight_rejects_expired_or_unsealed_receipts() -> None:
    overlay, provider = _signed_overlay()
    sealed, prepared = _prepare(overlay, provider)
    expired = preflight_rsd_live_delegation_overlay(
        overlay,
        prepared,
        _postgres_overlay(),
        sealed,
        now=datetime(2031, 1, 1, tzinfo=UTC),
    )
    assert (
        EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        in expired.failures
    )
    fabricated = preflight_rsd_live_delegation_overlay(
        overlay,
        prepared,
        _postgres_overlay(),
        {"sealed": True},
        now=_NOW,
    )
    assert (
        EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        in fabricated.failures
    )
    with pytest.raises(TypeError, match="minted by preparation"):
        SealedRsdLiveDelegationAuthorityReceipt()  # type: ignore[call-arg]


@pytest.mark.unit
def test_preparation_rejects_current_attestor_substitution() -> None:
    overlay, provider = _signed_overlay()
    changed = _KeyProvider(
        provider._roots, {str(overlay.result_attestor_key_id): b"x" * 32}
    )
    with pytest.raises(ProtocolConfigurationError, match="result attestor"):
        _prepare(overlay, changed)


@pytest.mark.unit
def test_preparation_rejects_rotated_root_before_emitting_receipt() -> None:
    overlay, provider = _signed_overlay()
    rotated_root = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    with pytest.raises(ProtocolConfigurationError, match="root is unavailable"):
        _prepare(
            overlay,
            _KeyProvider(
                {overlay.authority_envelope.authority_domain: rotated_root},
                provider._nodes,
            ),
        )


@pytest.mark.unit
def test_preflight_rejects_a_sentinel_forged_receipt() -> None:
    overlay, provider = _signed_overlay()
    sealed, prepared = _prepare(overlay, provider)
    forged = object.__new__(SealedRsdLiveDelegationAuthorityReceipt)
    for field in (
        "overlay_digest_sha256",
        "attestor_key_id",
        "attestor_public_key_fingerprint_sha256",
        "observed_model_id",
        "observed_model_attestation_sha256",
        "expires_at",
    ):
        object.__setattr__(forged, field, getattr(sealed, field))
    result = preflight_rsd_live_delegation_overlay(
        overlay, prepared, _postgres_overlay(), forged, now=_NOW
    )
    assert result.ready is False
    assert result.result_anchor is None
    assert (
        EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        in result.failures
    )


@pytest.mark.unit
def test_preflight_uses_no_provider_and_binds_the_exact_prepared_observation() -> None:
    class _CountingProvider(_KeyProvider):
        root_calls: int = 0
        attestor_calls: int = 0

        def get_domain_trust_root(self, domain_id: str) -> bytes | None:
            self.root_calls += 1
            return super().get_domain_trust_root(domain_id)

        def get_node_identity_key(self, node_id: str) -> bytes | None:
            self.attestor_calls += 1
            return super().get_node_identity_key(node_id)

    overlay = _overlay()
    provider = _CountingProvider(
        {overlay.authority_envelope.authority_domain: _STATIC_ROOT},
        {str(overlay.result_attestor_key_id): _STATIC_ATTESTOR},
    )
    sealed, prepared = _prepare(overlay, provider)
    calls_before_preflight = (provider.root_calls, provider.attestor_calls)
    result = preflight_rsd_live_delegation_overlay(
        overlay, prepared, _postgres_overlay(), sealed, now=_NOW
    )
    assert result.result_anchor is not None
    assert (provider.root_calls, provider.attestor_calls) == calls_before_preflight

    substituted = prepared.model_copy(
        update={"verified_overlay_digest_sha256": "a" * 64}
    )
    rejected = preflight_rsd_live_delegation_overlay(
        overlay, substituted, _postgres_overlay(), sealed, now=_NOW
    )
    assert rejected.result_anchor is None
    assert (
        EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        in rejected.failures
    )


@pytest.mark.unit
def test_preflight_requires_a_fresh_preparation_snapshot() -> None:
    overlay = _overlay()
    sealed, prepared = _prepare(overlay, _static_provider(overlay))
    result = preflight_rsd_live_delegation_overlay(
        overlay,
        prepared,
        _postgres_overlay(),
        sealed,
        now=datetime(2026, 9, 4, 0, 5, 1, tzinfo=UTC),
    )
    assert result.result_anchor is None
    assert (
        EnumRsdLiveDelegationPreflightFailure.SEALED_AUTHORITY_UNVERIFIED
        in result.failures
    )


@pytest.mark.unit
def test_public_models_have_explicit_canonical_json_roundtrips() -> None:
    overlay = _overlay()
    sealed, prepared = _prepare(overlay, _static_provider(overlay))
    result = preflight_rsd_live_delegation_overlay(
        overlay, prepared, _postgres_overlay(), sealed, now=_NOW
    )
    models = (
        overlay.authority_envelope,
        prepared,
        overlay,
        result,
        result.result_anchor,
    )
    for model in models:
        assert model is not None
        model_type = type(model)
        assert (
            model_type.model_validate_json(model.model_dump_json(), strict=True)
            == model
        )
    assert not hasattr(sealed, "model_dump")


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        "\nmodel_id: Qwen/Qwen3.8-27B\n",
        "\nlane: &lane dev\nlocale: *lane\n",
        "\nauthority_envelope:\n  issued_at: 2026-01-01\n",
    ],
)
def test_loader_rejects_duplicate_keys_aliases_and_non_json_scalars(
    tmp_path: Path, body: str
) -> None:
    candidate = tmp_path / "invalid-overlay.yaml"
    if "authority_envelope:" in body:
        text = _OVERLAY_PATH.read_text(encoding="utf-8").replace(
            'issued_at: "2026-01-01T00:00:00Z"', "issued_at: 2026-01-01"
        )
    else:
        text = _OVERLAY_PATH.read_text(encoding="utf-8") + body
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        load_rsd_live_delegation_overlay(candidate)


@pytest.mark.unit
def test_loader_rejects_oversized_yaml_before_parsing(tmp_path: Path) -> None:
    candidate = tmp_path / "large-overlay.yaml"
    candidate.write_bytes(b"#" * ((64 * 1024) + 1))
    with pytest.raises(ProtocolConfigurationError, match="overlay is invalid"):
        load_rsd_live_delegation_overlay(candidate)


@pytest.mark.unit
def test_loader_rejects_deeply_nested_yaml_without_leaking_parser_errors(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "nested-overlay.yaml"
    candidate.write_text(
        "value: " + ("[" * 500) + "null" + ("]" * 500), encoding="utf-8"
    )
    assert candidate.stat().st_size < 64 * 1024
    with pytest.raises(ProtocolConfigurationError) as error:
        load_rsd_live_delegation_overlay(candidate)
    assert str(error.value) == (
        "[ONEX_CORE_041_INVALID_CONFIGURATION] RSD live delegation overlay is invalid"
    )
    assert error.value.__cause__ is None


@pytest.mark.unit
def test_loader_errors_are_exactly_safe_and_do_not_expose_caller_input(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "secret-endpoint-192.0.2.1.yaml"
    candidate.write_text("model_id: [unterminated", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError) as error:
        load_rsd_live_delegation_overlay(candidate)
    rendered = str(error.value)
    assert rendered.endswith("RSD live delegation overlay is invalid")
    assert "secret-endpoint" not in rendered
    assert "192.0.2.1" not in rendered
    assert "unterminated" not in rendered


@pytest.mark.unit
def test_preparation_normalizes_invalid_clock_and_provider_errors() -> None:
    overlay = _overlay()

    class _ExplodingProvider:
        def get_domain_trust_root(self, domain_id: str) -> bytes | None:
            raise RuntimeError("unavailable")

        def get_node_identity_key(self, node_id: str) -> bytes | None:
            raise RuntimeError("unavailable")

    with pytest.raises(ProtocolConfigurationError):
        prepare_rsd_live_delegation_authority(
            overlay,
            _observation(overlay),
            _static_provider(overlay),
            now="not-a-datetime",  # type: ignore[arg-type]
        )
    with pytest.raises(ProtocolConfigurationError):
        prepare_rsd_live_delegation_authority(
            overlay, _observation(overlay), _ExplodingProvider(), now=_NOW
        )


@pytest.mark.unit
def test_preparation_suppresses_provider_and_crypto_failure_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _overlay()
    sentinel = "secret-provider-topology-sentinel"

    class _ExplodingRootProvider(_KeyProvider):
        def get_domain_trust_root(self, domain_id: str) -> bytes | None:
            raise RuntimeError(sentinel)

    with pytest.raises(ProtocolConfigurationError) as root_error:
        _prepare(
            overlay,
            _ExplodingRootProvider(
                {}, {str(overlay.result_attestor_key_id): _STATIC_ATTESTOR}
            ),
        )
    assert str(root_error.value) == (
        "[ONEX_CORE_041_INVALID_CONFIGURATION] "
        "RSD live delegation root is unavailable or substituted"
    )
    assert root_error.value.__cause__ is None
    assert sentinel not in str(root_error.value)

    class _ExplodingAttestorProvider(_KeyProvider):
        def get_node_identity_key(self, node_id: str) -> bytes | None:
            raise RuntimeError(sentinel)

    with pytest.raises(ProtocolConfigurationError) as attestor_error:
        _prepare(
            overlay,
            _ExplodingAttestorProvider(
                {overlay.authority_envelope.authority_domain: _STATIC_ROOT}, {}
            ),
        )
    assert str(attestor_error.value) == (
        "[ONEX_CORE_041_INVALID_CONFIGURATION] "
        "RSD live delegation result attestor is unavailable or substituted"
    )
    assert attestor_error.value.__cause__ is None
    assert sentinel not in str(attestor_error.value)

    def _exploding_verify(*args: object) -> bool:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(live_delegation_overlay, "verify_base64", _exploding_verify)
    with pytest.raises(ProtocolConfigurationError) as crypto_error:
        _prepare(overlay, _static_provider(overlay))
    assert str(crypto_error.value) == (
        "[ONEX_CORE_041_INVALID_CONFIGURATION] "
        "RSD live delegation authority signature is invalid"
    )
    assert crypto_error.value.__cause__ is None
    assert sentinel not in str(crypto_error.value)
