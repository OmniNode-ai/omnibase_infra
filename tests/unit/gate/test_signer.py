# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for OmniGate Sigstore signing helpers."""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime

import pytest

from omnibase_core.gate.receipt_canonical import canonical_receipt_payload
from omnibase_core.models.gate import ModelOmniGateReceipt
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.gate.signer import OmniGateSigner


def _receipt(*, sigstore_bundle_json: str | None = None) -> ModelOmniGateReceipt:
    return ModelOmniGateReceipt(
        schema_version=ModelSemVer(major=1, minor=0, patch=0),
        project_name="omni",
        project_url="https://github.com/OmniNode-ai/omnibase_core",
        repository_id="123456789",
        base_sha="a" * 40,
        head_sha="b" * 40,
        commit_sha="b" * 40,
        diff_hash="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        receipt_schema_fingerprint="sha256:" + "e" * 64,
        branch="contrib/omnigate",
        timestamp=datetime(2026, 5, 17, 14, 0, tzinfo=UTC),
        signer_identity="https://github.com/org/repo/.github/workflows/omnigate.yml@refs/heads/main",
        signer_issuer="https://token.actions.githubusercontent.com",
        sigstore_bundle_json=sigstore_bundle_json,
    )


@pytest.mark.unit
class TestOmniGateSigner:
    def test_serialize_receipt_for_signing(self) -> None:
        signer = OmniGateSigner()
        receipt = _receipt()

        payload = signer.serialize_for_signing(receipt)

        assert isinstance(payload, bytes)
        assert payload == canonical_receipt_payload(receipt, exclude_signature=True)

    def test_serialization_is_deterministic(self) -> None:
        signer = OmniGateSigner()
        receipt = _receipt()

        assert signer.serialize_for_signing(receipt) == signer.serialize_for_signing(
            receipt,
        )

    def test_signature_bundle_is_excluded_from_payload(self) -> None:
        signer = OmniGateSigner()
        receipt = _receipt(sigstore_bundle_json='{"bundle":"signed"}')

        payload = signer.serialize_for_signing(receipt)
        decoded = json.loads(payload.decode("utf-8"))

        assert "sigstore_bundle_json" not in decoded

    def test_unicode_normalization_is_deterministic(self) -> None:
        signer = OmniGateSigner()
        composed = _receipt().model_copy(update={"project_name": "Caf\u00e9"})
        decomposed = _receipt().model_copy(update={"project_name": "Cafe\u0301"})

        assert signer.serialize_for_signing(composed) == signer.serialize_for_signing(
            decomposed,
        )

    def test_sign_lazy_imports_sigstore_and_returns_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, bytes] = {}

        class Bundle:
            def to_json(self) -> str:
                return '{"bundle":"ok"}'

        class ArtifactSigner:
            def sign_artifact(self, payload: bytes) -> Bundle:
                calls["payload"] = payload
                return Bundle()

        class SignerContext:
            def __enter__(self) -> ArtifactSigner:
                return ArtifactSigner()

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: types.TracebackType | None,
            ) -> bool | None:
                return None

        class SigningContext:
            @staticmethod
            def production() -> SigningContext:
                return SigningContext()

            def signer(self) -> SignerContext:
                return SignerContext()

        sign_module = types.ModuleType("sigstore.sign")
        sign_module.SigningContext = SigningContext
        sigstore_module = types.ModuleType("sigstore")
        monkeypatch.setitem(sys.modules, "sigstore", sigstore_module)
        monkeypatch.setitem(sys.modules, "sigstore.sign", sign_module)

        receipt = _receipt()
        signed = OmniGateSigner().sign(receipt)

        assert signed is not receipt
        assert signed.sigstore_bundle_json == '{"bundle":"ok"}'
        assert calls["payload"] == OmniGateSigner().serialize_for_signing(receipt)

    def test_verify_false_when_bundle_missing(self) -> None:
        assert (
            OmniGateSigner().verify(
                _receipt(),
                expected_identity="identity",
                expected_issuer="issuer",
            )
            is False
        )

    def test_verify_uses_exact_identity_and_issuer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, object] = {}

        class VerificationMaterials:
            @staticmethod
            def from_json(value: str) -> object:
                calls["bundle_json"] = value
                return {"bundle": value}

        class VerifierInstance:
            def verify_artifact(
                self,
                payload: bytes,
                bundle: object,
                policy: object,
            ) -> None:
                calls["payload"] = payload
                calls["bundle"] = bundle
                calls["policy"] = policy

        class Verifier:
            @staticmethod
            def production() -> VerifierInstance:
                return VerifierInstance()

        class Identity:
            def __init__(self, *, identity: str, issuer: str) -> None:
                calls["identity"] = identity
                calls["issuer"] = issuer

        verify_module = types.ModuleType("sigstore.verify")
        verify_module.Verifier = Verifier
        verify_module.VerificationMaterials = VerificationMaterials
        policy_module = types.ModuleType("sigstore.verify.policy")
        policy_module.Identity = Identity
        sigstore_module = types.ModuleType("sigstore")
        monkeypatch.setitem(sys.modules, "sigstore", sigstore_module)
        monkeypatch.setitem(sys.modules, "sigstore.verify", verify_module)
        monkeypatch.setitem(sys.modules, "sigstore.verify.policy", policy_module)

        receipt = _receipt(sigstore_bundle_json='{"bundle":"ok"}')
        result = OmniGateSigner().verify(
            receipt,
            expected_identity="https://github.com/org/repo/.github/workflows/omnigate.yml@refs/heads/main",
            expected_issuer="https://token.actions.githubusercontent.com",
        )

        assert result is True
        assert calls["bundle_json"] == '{"bundle":"ok"}'
        assert (
            calls["identity"]
            == "https://github.com/org/repo/.github/workflows/omnigate.yml@refs/heads/main"
        )
        assert calls["issuer"] == "https://token.actions.githubusercontent.com"

    def test_verify_false_on_sigstore_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class VerificationMaterials:
            @staticmethod
            def from_json(value: str) -> object:
                return {"bundle": value}

        class VerifierInstance:
            def verify_artifact(
                self,
                payload: bytes,
                bundle: object,
                policy: object,
            ) -> None:
                msg = "signature mismatch"
                raise ValueError(msg)

        class Verifier:
            @staticmethod
            def production() -> VerifierInstance:
                return VerifierInstance()

        class Identity:
            def __init__(self, *, identity: str, issuer: str) -> None:
                self.identity = identity
                self.issuer = issuer

        verify_module = types.ModuleType("sigstore.verify")
        verify_module.Verifier = Verifier
        verify_module.VerificationMaterials = VerificationMaterials
        policy_module = types.ModuleType("sigstore.verify.policy")
        policy_module.Identity = Identity
        sigstore_module = types.ModuleType("sigstore")
        monkeypatch.setitem(sys.modules, "sigstore", sigstore_module)
        monkeypatch.setitem(sys.modules, "sigstore.verify", verify_module)
        monkeypatch.setitem(sys.modules, "sigstore.verify.policy", policy_module)

        assert (
            OmniGateSigner().verify(
                _receipt(sigstore_bundle_json='{"bundle":"bad"}'),
                expected_identity="identity",
                expected_issuer="issuer",
            )
            is False
        )
