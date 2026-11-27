# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse runtime contracts that are not the revision the image was built from (OMN-25856).

The dev services bind-mount the host clone's ``contracts/`` over the image copy.
A queued build of an older ref booted with the clone's newer
``runtime_config.yaml`` and died on a policy its code did not know. The image
stamps a ``sha256sum`` manifest of its own ``contracts/runtime`` files at build
into ``/app/config`` (never mounted over; ``ONEX_CONTRACTS_STAMP_PATH`` names it, set by the
Dockerfile beside ``ONEX_RUNTIME_CONTRACTS_DIR``); this guard compares the files the
container actually sees against it, before the kernel starts.

A file under the mounted runtime contracts directory that differs from the
manifest, or that the manifest does not list, is a mismatch. Files the mount
does not carry (core's own runtime contracts) are not judged.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.runtime.model_contracts_revision_request import (
    ModelContractsRevisionRequest,
)
from omnibase_infra.models.runtime.model_contracts_revision_result import (
    ModelContractsRevisionResult,
)

ENV_CONTRACTS_STAMP_PATH = "ONEX_CONTRACTS_STAMP_PATH"
ENV_RUNTIME_CONTRACTS_DIR = "ONEX_RUNTIME_CONTRACTS_DIR"


class ContractsRevisionMismatchError(RuntimeHostError):
    """Raised at boot when the mounted contracts are not the image's revision."""

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        super().__init__(
            message=f"CONTRACTS_REVISION_MISMATCH: {message}",
            error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
            context=context,
            **extra_context,
        )


class HandlerContractsRevisionGuard:
    """Raises :class:`ContractsRevisionMismatchError` for contracts of another revision."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelContractsRevisionRequest
    ) -> ModelContractsRevisionResult:
        stamp_path = Path(request.stamp_path)
        if not request.stamp_path or not stamp_path.is_file():
            raise ContractsRevisionMismatchError(
                f"the image carries no contracts stamp (path {request.stamp_path!r})",
                stamp_path=str(stamp_path),
            )
        stamped: dict[str, set[str]] = {}
        for line in stamp_path.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.partition("  ")
            if digest and name:
                stamped.setdefault(name.removeprefix("./"), set()).add(digest)
        runtime_dir = Path(request.runtime_contracts_dir)
        mounted = sorted(p for p in runtime_dir.rglob("*") if p.is_file())
        mismatched = [
            name
            for name, path in (
                (p.relative_to(runtime_dir).as_posix(), p) for p in mounted
            )
            if hashlib.sha256(path.read_bytes()).hexdigest()
            not in stamped.get(name, ())
        ]
        if mismatched:
            raise ContractsRevisionMismatchError(
                f"{len(mismatched)} file(s) under {runtime_dir} differ from the "
                "revision this image was built from (a bind mount from another "
                f"revision?): {', '.join(mismatched[:10])}",
                files=tuple(mismatched),
            )
        return ModelContractsRevisionResult(files_checked=len(mounted))


_GUARD = HandlerContractsRevisionGuard()


def refuse_mismatched_contracts(env: Mapping[str, str]) -> int:
    """Entrypoint step: 0 to boot, 1 (with the named error on stderr) to abort."""
    try:
        _GUARD.handle(
            ModelContractsRevisionRequest(
                stamp_path=env.get(ENV_CONTRACTS_STAMP_PATH, ""),
                runtime_contracts_dir=env.get(ENV_RUNTIME_CONTRACTS_DIR, ""),
            )
        )
    except ContractsRevisionMismatchError as exc:
        sys.stderr.write(f"[entrypoint] ERROR: {exc} -- aborting boot\n")
        return 1
    return 0
