# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: service_kernel must bind build identity before attach (OMN-10856).

Mirrors the source-level analysis pattern used by
``test_service_kernel_introspection_wiring.py`` (OMN-6405): driving the full
``bootstrap()`` coroutine end-to-end requires a real Kafka/DB/contract
environment, so the wiring guarantee is checked via AST/source analysis
instead. The behavioral half (identity actually gets bound correctly) is
covered by ``test_runtime_topology_manifest_identity_omn10856.py``, which
unit-tests ``bind_introspection_manifest_identity`` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

SERVICE_KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)


def _read_source() -> str:
    assert SERVICE_KERNEL_PATH.exists(), (
        f"service_kernel.py not found at {SERVICE_KERNEL_PATH}"
    )
    return SERVICE_KERNEL_PATH.read_text(encoding="utf-8")


class TestIntrospectionManifestIdentityWiring:
    """attach_manifest() must be called with an identity-bound manifest."""

    def test_bind_introspection_manifest_identity_is_imported(self) -> None:
        source = _read_source()
        assert "bind_introspection_manifest_identity" in source, (
            "service_kernel.py must call bind_introspection_manifest_identity "
            "before health_server.attach_manifest(...) so the served "
            "/v1/introspection/manifest carries runtime_profile/image_sha/"
            "deployment_sha (OMN-10856)."
        )

    def test_binding_happens_before_attach_manifest(self) -> None:
        """The bind call must precede attach_manifest in source order within
        the same block, not just appear anywhere in the file."""
        source = _read_source()
        bind_idx = source.index("bind_introspection_manifest_identity(")
        attach_idx = source.index(
            "health_server.attach_manifest(_introspection_manifest)"
        )
        assert bind_idx < attach_idx, (
            "bind_introspection_manifest_identity(...) must run before "
            "health_server.attach_manifest(_introspection_manifest) so the "
            "attached manifest carries build identity, not the bare "
            "discovery result (OMN-10856)."
        )

    def test_binding_passes_kernel_profile_name(self) -> None:
        source = _read_source()
        assert "runtime_profile=kernel_profile.name" in source, (
            "bind_introspection_manifest_identity must be called with "
            "runtime_profile=kernel_profile.name — the same resolved "
            "profile identity used elsewhere in the kernel — not a fresh "
            "os.getenv('RUNTIME_PROFILE') re-read (single source, OMN-10856)."
        )

    def test_image_sha_reuses_onex_image_digest_env_var(self) -> None:
        """ONEX_IMAGE_DIGEST is already read by publish_runtime_manifest's
        image_digest= kwarg (OMN-11196/97, runtime_manifests projection
        pathway) — the introspection binder must reuse that exact name, not
        introduce a second env var for the same concept."""
        source = _read_source()
        assert 'ENV_VAR_IMAGE_SHA = "ONEX_IMAGE_DIGEST"' in source or (
            "ENV_VAR_IMAGE_SHA" in source
        ), (
            "service_kernel.py must resolve image_sha via "
            "ENV_VAR_IMAGE_SHA (ONEX_IMAGE_DIGEST), reusing the same env var "
            "already read by publish_runtime_manifest(image_digest=...) "
            "(OMN-10856)."
        )
        assert 'os.getenv("ONEX_IMAGE_DIGEST")' in source, (
            "publish_runtime_manifest's pre-existing ONEX_IMAGE_DIGEST read "
            "must remain intact (OMN-10856 must not orphan it)."
        )

    def test_identity_resolution_uses_fail_fast_from_raw(self) -> None:
        """The env values feeding image_sha/deployment_sha must route through
        ModelRuntimeBuildSha.from_raw (explicit absent-with-reason), never a
        bare os.getenv(..., "unknown")-style silent default."""
        source = _read_source()
        assert "ModelRuntimeBuildSha.from_raw(" in source
        assert "ENV_VAR_DEPLOYMENT_SHA" in source
