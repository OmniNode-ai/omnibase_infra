# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Sigstore signing and verification for OmniGate receipts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from typing import cast

from omnibase_core.gate.receipt_canonical import canonical_receipt_payload
from omnibase_core.models.gate.model_omnigate_receipt import ModelOmniGateReceipt


class OmniGateSigner:
    """Sign and verify OmniGate receipt payloads with Sigstore."""

    def serialize_for_signing(self, receipt: ModelOmniGateReceipt) -> bytes:
        """Return the canonical receipt bytes covered by the signature."""
        return cast(
            "bytes",
            canonical_receipt_payload(receipt, exclude_signature=True),
        )

    def sign(self, receipt: ModelOmniGateReceipt) -> ModelOmniGateReceipt:
        """Return a receipt copy with a Sigstore bundle JSON string."""
        signing_context_factory = _load_module_attr(
            "sigstore.sign",
            "SigningContext",
        )
        production = _method_no_args(signing_context_factory, "production")
        context = production()
        signer_context = cast(
            "AbstractContextManager[object]",
            _method_no_args(context, "signer")(),
        )

        payload = self.serialize_for_signing(receipt)
        with signer_context as signer:
            bundle = _method_with_bytes(signer, "sign_artifact")(payload)
        return receipt.model_copy(
            update={"sigstore_bundle_json": _method_no_args(bundle, "to_json")()},
        )

    def verify(
        self,
        receipt: ModelOmniGateReceipt,
        *,
        expected_identity: str,
        expected_issuer: str,
    ) -> bool:
        """Verify receipt signature against one exact trusted identity policy."""
        if receipt.sigstore_bundle_json is None:
            return False
        if not expected_identity or not expected_issuer:
            return False

        payload = self.serialize_for_signing(receipt)
        try:
            bundle = _method_with_string(
                _load_module_attr("sigstore.verify", "VerificationMaterials"),
                "from_json",
            )(receipt.sigstore_bundle_json)
            policy = _identity_factory()(
                identity=expected_identity,
                issuer=expected_issuer,
            )
            verifier = _method_no_args(
                _load_module_attr("sigstore.verify", "Verifier"),
                "production",
            )()
            _verify_artifact(verifier)(payload, bundle, policy)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return True


def _load_module_attr(module_name: str, attr_name: str) -> object:
    return vars(import_module(module_name))[attr_name]


def _method_no_args(target: object, method_name: str) -> Callable[[], object]:
    return cast("Callable[[], object]", getattr(target, method_name))


def _method_with_bytes(target: object, method_name: str) -> Callable[[bytes], object]:
    return cast("Callable[[bytes], object]", getattr(target, method_name))


def _method_with_string(target: object, method_name: str) -> Callable[[str], object]:
    return cast("Callable[[str], object]", getattr(target, method_name))


def _identity_factory() -> Callable[..., object]:
    return cast(
        "Callable[..., object]",
        _load_module_attr("sigstore.verify.policy", "Identity"),
    )


def _verify_artifact(
    verifier: object,
) -> Callable[[bytes, object, object], object]:
    method_name = "verify_artifact"
    return cast(
        "Callable[[bytes, object, object], object]",
        getattr(verifier, method_name),
    )


__all__ = ["OmniGateSigner"]
