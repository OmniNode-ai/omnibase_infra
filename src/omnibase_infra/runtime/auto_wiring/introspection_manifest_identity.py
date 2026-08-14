# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind runtime build identity onto the introspection manifest (OMN-10856).

Extracted as a standalone pure function — analogous to
``manifest_builder.build_runtime_manifest`` — so the identity-binding seam
served over ``/v1/introspection/manifest`` is unit-testable without driving
the full ``service_kernel.bootstrap()`` sequence. The kernel calls exactly
this function at its ``health_server.attach_manifest(...)`` call site
(``service_kernel.py`` ~L3199-3208), so a test that asserts on this
function's output is asserting on the artifact that runs.

This module does NOT read process environment variables itself.
``scripts/check-env-reads.sh`` restricts new env-var reads to an approved
boundary set that ``service_kernel.py`` is on and this module is not; the
kernel resolves ``ONEX_IMAGE_DIGEST`` / ``ONEX_DEPLOYMENT_SHA`` there and
passes the already-classified ``ModelRuntimeBuildSha`` values in.
``ONEX_IMAGE_DIGEST`` is chosen to reuse the existing single source of
truth: it is already read by ``manifest_builder.build_runtime_manifest``
(the separate ``runtime_manifests`` projection pathway,
OMN-11196/OMN-11197) via ``service_kernel.py``'s
``publish_runtime_manifest(image_digest=...)`` call — reusing the same name
means one env var injection serves both the projection pathway and the
introspection HTTP surface, not two divergently named ones.
``ONEX_DEPLOYMENT_SHA`` has no prior reader in this repo; as of
OMN-10856 neither var is injected anywhere in ``omninode_infra``'s
``k8s/onex-dev/runtime/`` Deployments (verified by direct grep) — both
surface as absent-with-reason until the companion env-injection change
lands there.
"""

from __future__ import annotations

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_runtime_build_sha import (
    ModelRuntimeBuildSha,
)

ENV_VAR_IMAGE_SHA = "ONEX_IMAGE_DIGEST"
ENV_VAR_DEPLOYMENT_SHA = "ONEX_DEPLOYMENT_SHA"


def bind_introspection_manifest_identity(
    manifest: ModelAutoWiringManifest,
    *,
    runtime_profile: str,
    image_sha: ModelRuntimeBuildSha,
    deployment_sha: ModelRuntimeBuildSha,
) -> ModelAutoWiringManifest:
    """Return a copy of ``manifest`` with build identity bound (OMN-10856).

    Reuses ``manifest.contracts`` / ``manifest.errors`` verbatim — topology
    is never re-derived here, only the identity fields are added — so this
    stays the single source for what auto-discovery found.

    Args:
        manifest: The (filtered or discovery-only) auto-wiring manifest to
            enrich. Its ``contracts``/``errors`` are carried through
            unchanged.
        runtime_profile: The resolved ``RUNTIME_PROFILE`` identity (e.g.
            from ``load_runtime_profile().name``).
        image_sha: Already-classified build SHA (present or
            absent-with-reason) — the caller resolves this from
            ``ENV_VAR_IMAGE_SHA``.
        deployment_sha: Already-classified deployment SHA — the caller
            resolves this from ``ENV_VAR_DEPLOYMENT_SHA``.

    Returns:
        A new ``ModelAutoWiringManifest`` with all three identity fields set.
    """
    return ModelAutoWiringManifest(
        contracts=manifest.contracts,
        errors=manifest.errors,
        runtime_profile=runtime_profile,
        image_sha=image_sha,
        deployment_sha=deployment_sha,
    )


__all__: list[str] = [
    "ENV_VAR_DEPLOYMENT_SHA",
    "ENV_VAR_IMAGE_SHA",
    "bind_introspection_manifest_identity",
]
