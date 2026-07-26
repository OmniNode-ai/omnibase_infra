# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Pure COMPUTE handler for the release-identity fresh-deploy fitness gate.

Canonical replacement for the freestanding imperative script
``scripts/check_release_identity.py`` (legacy, OMN-13412). The gate enforces:
when a diff touches packaged source (``src/**``) AND any published tag exists,
``pyproject.toml``'s ``project.version`` MUST be strictly greater than the highest
published version — otherwise two distinct builds ship the same version string and
every downstream proof packet that cites the runtime version can no longer
distinguish them.

Canonical shape (definition B, OMN-14355/OMN-14407): the handler core is
``handle(request: ModelReleaseIdentityRequest) -> ModelReleaseIdentityDecision``,
a single BaseModel-typed positional the shared runtime adapter can bind. The core
does NOT reference the runtime event-envelope type (that boundary lives in the
shared runtime adapter, never a per-node wrapper).

Pure vs I/O split (OMN-14471): version parsing, latest-tag selection, ``src/**``
detection and the bump invariant are pure and live here. All I/O — reading
``pyproject.toml``, running ``git tag --list`` / ``git diff`` — is done by the thin
CLI collector/shim and handed in as ``ModelReleaseIdentityRequest``, so the handler
is deterministic and unit-testable with no subprocess or filesystem access.

Ticket: OMN-14471
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_release_identity_compute.models.model_release_identity_decision import (
    ModelReleaseIdentityDecision,
)
from omnibase_infra.nodes.node_release_identity_compute.models.model_release_identity_request import (
    ModelReleaseIdentityRequest,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

# Packaged-source prefixes whose change requires a version bump. Kept identical to
# the legacy gate's ``_PACKAGED_PREFIXES`` so exemption behavior is unchanged.
_PACKAGED_PREFIXES: tuple[str, ...] = ("src/",)


class HandlerReleaseIdentity:
    """Pure COMPUTE handler for the release-identity fitness decision.

    ``handle`` is a pure function of its ``ModelReleaseIdentityRequest``: same
    request -> same decision, no I/O. The messages and exit codes are preserved
    byte-for-byte from the legacy ``scripts/check_release_identity.py`` so the two
    wiring points (fresh-deploy-fitness CI + the pre-commit hook) behave identically.

    Attributes:
        handler_type: EnumHandlerType.COMPUTE_HANDLER
        handler_category: EnumHandlerTypeCategory.COMPUTE
    """

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        """Initialize the handler.

        Args:
            container: ONEX DI container. Optional because the gate is a pure
                function with no injected dependencies; the CLI shim constructs
                the handler directly without a container.
        """
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelReleaseIdentityRequest
    ) -> ModelReleaseIdentityDecision:
        """Evaluate the release-identity invariant for pre-collected gate inputs.

        Decision order (first match wins), mirroring the legacy gate exactly:
            1. pyproject version absent/empty  -> exit 2 (config error)
            2. pyproject version malformed     -> exit 2 (config error)
            3. no published tag yet            -> exit 0 (bump not required)
            4. no packaged src/** change       -> exit 0 (bump not required)
            5. version ahead of latest tag     -> exit 0 (pass)
            6. otherwise                       -> exit 1 (not ahead; must bump)

        Args:
            request: Pre-collected gate inputs (see ModelReleaseIdentityRequest).

        Returns:
            The gate decision with exit code, stream, message, and reason code.
        """
        # Step 1 & 2: resolve the declared version (config-error paths).
        raw = request.pyproject_version_raw
        if not raw:
            return ModelReleaseIdentityDecision(
                exit_code=2,
                stream="stderr",
                message=f"ERROR: no project.version in {request.pyproject_path}",
                reason_code="no_pyproject_version",
            )
        try:
            pyproject_version = Version(str(raw))
        except InvalidVersion as exc:
            return ModelReleaseIdentityDecision(
                exit_code=2,
                stream="stderr",
                message=f"ERROR: malformed project.version {raw!r}: {exc}",
                reason_code="malformed_pyproject_version",
            )

        # Step 3: no published tag -> the bump invariant does not apply yet.
        latest = self._latest_published_version(request.published_tags)
        if latest is None:
            return ModelReleaseIdentityDecision(
                exit_code=0,
                stream="stdout",
                message="OK: no published tag yet — release-identity bump not required.",
                reason_code="no_published_tag",
            )

        # Step 4: no packaged src/** change -> the published image is unaffected.
        if not self._packaged_source_changed(request.changed_files):
            return ModelReleaseIdentityDecision(
                exit_code=0,
                stream="stdout",
                message=(
                    "OK: no packaged src/** change in this diff — version bump not "
                    f"required (pyproject {pyproject_version}, latest published {latest})."
                ),
                reason_code="no_packaged_change",
            )

        # Step 5: version strictly ahead of the latest published tag -> pass.
        if pyproject_version > latest:
            return ModelReleaseIdentityDecision(
                exit_code=0,
                stream="stdout",
                message=(
                    f"OK: version {pyproject_version} is ahead of latest published {latest}."
                ),
                reason_code="version_ahead",
            )

        # Step 6: packaged source changed but the version is not ahead -> FAIL.
        suggested_bump = Version(f"{latest.major}.{latest.minor}.{latest.micro + 1}")
        fail_line = (
            "FAIL: packaged source changed but pyproject version "
            f"{pyproject_version} is NOT ahead of the latest published version "
            f"{latest} (OMN-13412 release-identity gate)."
        )
        guidance_line = (
            "Merging code onto an already-published version aliases two code states "
            "under one image version. Bump project.version in pyproject.toml past "
            f"{latest} (e.g. {suggested_bump})."
        )
        return ModelReleaseIdentityDecision(
            exit_code=1,
            stream="stderr",
            message=f"{fail_line}\n{guidance_line}",
            reason_code="version_not_ahead",
        )

    @staticmethod
    def _latest_published_version(published_tags: tuple[str, ...]) -> Version | None:
        """Return the highest published semver tag, or None if there are none.

        A leading ``v`` is stripped; unparseable tags are skipped. Matches the
        legacy ``_latest_published_version`` selection exactly.

        Args:
            published_tags: Raw `git tag --list` lines.

        Returns:
            The highest ``Version`` across the tags, or None when there are no
            parseable tags.
        """
        best: Version | None = None
        for line in published_tags:
            tag = line.strip()
            if not tag:
                continue
            candidate = tag[1:] if tag.startswith("v") else tag
            try:
                ver = Version(candidate)
            except InvalidVersion:
                continue
            if best is None or ver > best:
                best = ver
        return best

    @staticmethod
    def _packaged_source_changed(changed_files: tuple[str, ...] | None) -> bool:
        """Decide whether packaged source changed.

        ``None`` means the change set was undeterminable (no base + no explicit
        list) and the invariant is enforced (fail-safe) — identical to the legacy
        gate's ``_packaged_source_changed`` "cannot prove exempt" branch.

        Args:
            changed_files: Changed files vs the base, or None if undeterminable.

        Returns:
            True if packaged source changed (or the set is undeterminable).
        """
        if changed_files is None:
            return True
        return any(
            f.startswith(prefix) for f in changed_files for prefix in _PACKAGED_PREFIXES
        )


__all__: list[str] = ["HandlerReleaseIdentity"]
