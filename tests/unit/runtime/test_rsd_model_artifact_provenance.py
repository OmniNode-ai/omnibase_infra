# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hostile coverage for the topology-free RSD model provenance contract."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, cast

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from omnibase_core.protocols.crypto.protocol_multi_key_provider import (
    ProtocolMultiKeyProvider,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_rsd_model_artifact_provenance import (
    ModelRsdModelArtifactProvenance,
)
from omnibase_infra.runtime.rsd_model_artifact_provenance import (
    load_rsd_model_artifact_provenance,
    provenance_signing_preimage,
    validate_rsd_model_artifact_provenance,
    verify_rsd_model_artifact_provenance,
)

_ROOT = Path(__file__).parents[3]
_EXAMPLE_PATH = (
    _ROOT / "docker/lane-overlays/dev.rsd-model-artifact-provenance.example.yaml"
)
_KEY_ID = "00000000-0000-4000-8000-000000000003"
_NOW = datetime(2026, 9, 4, 19, 58, tzinfo=UTC)


class _KeyProvider:
    def __init__(self, public_key: bytes | None) -> None:
        self.public_key = public_key

    def get_domain_trust_root(self, domain_id: str) -> bytes | None:
        del domain_id
        return None

    def get_node_identity_key(self, node_id: str) -> bytes | None:
        return self.public_key if node_id == _KEY_ID else None


def _raw_example() -> dict[str, object]:
    raw = yaml.safe_load(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert type(raw) is dict
    return cast("dict[str, object]", raw)


def _signed_example() -> tuple[ModelRsdModelArtifactProvenance, _KeyProvider]:
    raw = _raw_example()
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw()
    raw["signer_public_key_fingerprint_sha256"] = hashlib.sha256(public_key).hexdigest()
    raw["signature_base64"] = base64.urlsafe_b64encode(b"\x00" * 64).decode("ascii")
    unsigned = ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)
    raw["signature_base64"] = base64.urlsafe_b64encode(
        private_key.sign(provenance_signing_preimage(unsigned))
    ).decode("ascii")
    return (
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True),
        _KeyProvider(public_key),
    )


@pytest.mark.unit
def test_example_is_inert_and_binds_immutable_public_identity() -> None:
    provenance = load_rsd_model_artifact_provenance(_EXAMPLE_PATH)

    assert provenance.execute_enabled is False
    assert provenance.approval_status == "unapproved"
    assert provenance.base_model_id == "Qwen/Qwen3.8-27B"
    assert provenance.base_model_revision_sha == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert provenance.artifact_revision_sha == (
        "0cc27958cefbbe231782ec8511de8c4eb5233348"
    )
    assert provenance.artifact_manifest_digest_sha256 == (
        "e46ef4e3895ed0a6db7c237d642121095629c53bd5b3e5ac799b8a8e2ae83e4f"
    )
    assert provenance.artifact_manifest_algorithm == (
        "sha256-path-size-content-sha256-v1"
    )
    assert provenance.quantization == "modelopt_nvfp4"
    assert provenance.required_hardware_capability == "nvidia.rtx5090_32gb"
    assert provenance.runtime_implementation == "vllm"
    assert provenance.runtime_version is None
    assert provenance.served_model_id == "Qwen/Qwen3.8-27B"
    assert provenance.issued_at == "2026-09-04T19:55:00Z"
    assert provenance.expires_at == "2026-09-04T20:00:00Z"


@pytest.mark.unit
def test_approval_requires_a_separate_authorized_transition() -> None:
    raw = _raw_example()
    raw["approval_status"] = "approved"

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["issued_at", "expires_at"])
def test_freshness_fields_are_required(missing: str) -> None:
    raw = _raw_example()
    del raw[missing]

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
def test_freshness_fields_require_canonical_utc_form() -> None:
    raw = _raw_example()
    raw["issued_at"] = "2026-09-04 19:55:00+00:00"

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
def test_loader_rejects_unquoted_native_datetime_nodes(tmp_path: Path) -> None:
    path_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
    unquoted = path_text.replace(
        'issued_at: "2026-09-04T19:55:00Z"',
        "issued_at: 2026-09-04T19:55:00Z",
    )
    path = tmp_path / "provenance-unquoted.yaml"
    path.write_text(unquoted, encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        load_rsd_model_artifact_provenance(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("issued_at", "expires_at", "now"),
    [
        ("2026-09-04T19:59:00Z", "2026-09-04T20:04:00Z", _NOW),
        (
            "2026-09-04T19:55:00Z",
            "2026-09-04T20:00:00Z",
            datetime(2026, 9, 4, 20, 0, tzinfo=UTC),
        ),
    ],
)
def test_freshness_rejects_future_expired_and_exact_expiry_boundary(
    issued_at: str, expires_at: str, now: datetime
) -> None:
    raw = _raw_example()
    raw["issued_at"] = issued_at
    raw["expires_at"] = expires_at
    provenance = ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)

    with pytest.raises(ProtocolConfigurationError, match="outside its validity"):
        validate_rsd_model_artifact_provenance(provenance, now=now)


@pytest.mark.unit
def test_freshness_accepts_exact_maximum_window_and_current_issue_time() -> None:
    raw = _raw_example()
    raw["issued_at"] = "2026-09-04T19:58:00Z"
    raw["expires_at"] = "2026-09-04T20:03:00Z"
    provenance = ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)

    assert validate_rsd_model_artifact_provenance(provenance, now=_NOW) == provenance


@pytest.mark.unit
def test_freshness_rejects_excess_window_and_non_utc_clock() -> None:
    raw = _raw_example()
    raw["expires_at"] = "2026-09-04T20:01:00Z"
    provenance = ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)

    with pytest.raises(ProtocolConfigurationError, match="outside its validity"):
        validate_rsd_model_artifact_provenance(provenance, now=_NOW)
    with pytest.raises(ProtocolConfigurationError, match="requires a UTC clock"):
        validate_rsd_model_artifact_provenance(
            provenance, now=datetime(2026, 9, 4, 19, 58)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "updates",
    [
        {
            "artifact_revision_sha": None,
            "artifact_manifest_digest_sha256": None,
            "artifact_manifest_algorithm": None,
        },
        {
            "artifact_manifest_digest_sha256": None,
            "artifact_manifest_algorithm": "sha256-canonical-json-v1",
        },
        {
            "artifact_revision_sha": None,
            "artifact_manifest_algorithm": None,
        },
    ],
)
def test_artifact_binding_requires_immutable_form_and_digest_algorithm_pair(
    updates: dict[str, object],
) -> None:
    raw = _raw_example()
    raw.update(updates)

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
def test_artifact_revision_and_payload_commitment_are_accepted_together() -> None:
    provenance = ModelRsdModelArtifactProvenance.model_validate(
        _raw_example(), strict=True
    )

    assert provenance.artifact_revision_sha is not None
    assert provenance.artifact_manifest_digest_sha256 is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_model_id", "localhost/Qwen3.8-27B"),
        ("artifact_id", "192.168.0.1/Qwen3.8-27B"),
        ("served_model_id", "https://registry.invalid/Qwen3.8-27B"),
        ("artifact_id", "../local-checkpoint/Qwen3.8-27B"),
    ],
)
def test_registry_identifiers_reject_topology_and_local_path_values(
    field: str, value: str
) -> None:
    raw = _raw_example()
    raw[field] = value

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
def test_schema_is_strict_and_execute_can_never_be_enabled() -> None:
    raw = _raw_example()
    raw["execute_enabled"] = True
    raw["unexpected"] = "rejected"

    with pytest.raises(ValidationError):
        ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "algorithm",
    ["sha256-canonical-json-v1", "sha256-path-size-content-sha256-v1"],
)
def test_manifest_digest_variant_is_accepted(algorithm: str) -> None:
    raw = _raw_example()
    raw["artifact_revision_sha"] = None
    raw["artifact_manifest_digest_sha256"] = "0" * 64
    raw["artifact_manifest_algorithm"] = algorithm

    provenance = ModelRsdModelArtifactProvenance.model_validate(raw, strict=True)
    assert provenance.artifact_revision_sha is None
    assert provenance.artifact_manifest_digest_sha256 == "0" * 64


@pytest.mark.unit
def test_signature_verification_is_provider_bound_and_fail_closed() -> None:
    provenance, provider = _signed_example()

    assert (
        verify_rsd_model_artifact_provenance(provenance, provider, now=_NOW)
        == provenance
    )

    changed = provenance.model_copy(update={"served_model_id": "Qwen/changed"})
    with pytest.raises(ProtocolConfigurationError, match="signature is invalid"):
        verify_rsd_model_artifact_provenance(changed, provider, now=_NOW)

    missing = _KeyProvider(None)
    with pytest.raises(ProtocolConfigurationError, match="authority is unavailable"):
        verify_rsd_model_artifact_provenance(provenance, missing, now=_NOW)

    with pytest.raises(ProtocolConfigurationError, match="authority is unavailable"):
        verify_rsd_model_artifact_provenance(
            provenance, cast("ProtocolMultiKeyProvider", object()), now=_NOW
        )


@pytest.mark.unit
def test_signing_preimage_is_stable_and_excludes_signature() -> None:
    provenance, _ = _signed_example()
    changed_signature = provenance.model_copy(
        update={
            "signature_base64": base64.urlsafe_b64encode(b"\x01" * 64).decode("ascii")
        }
    )

    assert provenance_signing_preimage(provenance) == provenance_signing_preimage(
        changed_signature
    )
    changed_time = provenance.model_copy(update={"issued_at": "2026-09-04T19:56:00Z"})
    assert provenance_signing_preimage(provenance) != provenance_signing_preimage(
        changed_time
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        "schema_version: rsd.model-artifact-provenance.v1\n"
        "schema_version: rsd.model-artifact-provenance.v1\n",
        "a: &anchor\n  b: 1\nref: *anchor\n",
        "[",
    ],
)
def test_loader_redacts_malformed_duplicate_and_alias_documents(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "provenance.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError) as error:
        load_rsd_model_artifact_provenance(path)

    assert "RSD model artifact provenance is invalid" in str(error.value)
    assert "schema_version" not in str(error.value)


@pytest.mark.unit
def test_loader_rejects_oversized_document_without_echoing_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.yaml"
    path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError) as error:
        load_rsd_model_artifact_provenance(path)

    assert "RSD model artifact provenance is invalid" in str(error.value)


@pytest.mark.unit
def test_loader_uses_one_bounded_descriptor_read_without_stat_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _EXAMPLE_PATH.read_bytes()

    class _TrackingReader:
        def __init__(self, value: bytes) -> None:
            self._stream = io.BytesIO(value)
            self.read_sizes: list[int] = []

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            self._stream.close()

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return self._stream.read(size)

    reader = _TrackingReader(payload)

    original_stat = Path.stat

    def fail_stat(path: Path, *, follow_symlinks: bool = True) -> object:
        if path == _EXAMPLE_PATH:
            raise AssertionError("loader must not stat before opening the descriptor")
        return original_stat(path, follow_symlinks=follow_symlinks)

    def open_once(_path: Path, mode: str) -> _TrackingReader:
        assert mode == "rb"
        return reader

    monkeypatch.setattr(Path, "stat", fail_stat)
    monkeypatch.setattr(Path, "open", open_once)

    assert load_rsd_model_artifact_provenance(_EXAMPLE_PATH).execute_enabled is False
    assert reader.read_sizes == [64 * 1024 + 1]


@pytest.mark.unit
def test_preimage_is_canonical_json() -> None:
    provenance, _ = _signed_example()

    preimage = provenance_signing_preimage(provenance)
    assert preimage.startswith(b"omninode-rsd.model-artifact-provenance.v1\x00{")
    assert json.dumps(json.loads(preimage.split(b"\x00", 1)[1]), separators=(",", ":"))
