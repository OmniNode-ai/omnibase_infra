# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed operation over the atomic in-place install (OMN-22712).

The engine is ``atomic_venv_install`` and ``skill_install_verify``, which are
stdlib-only because they run under the interpreter of the venv they repair. The
host reconcile reaches them through the node runtime's command line; this handler
is the same operation as a typed request and result.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_venv_atomic_install_request import (
    ModelVenvAtomicInstallRequest,
)
from omnibase_infra.models.model_venv_atomic_install_result import (
    ModelVenvAtomicInstallResult,
)

_HERE = Path(__file__).resolve().parent
_REFUSED = 3


def _load(name: str) -> ModuleType:
    """The engine modules by path: they hold no package imports on purpose."""
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


class HandlerVenvAtomicInstall:
    """Stage the moved packages, exchange them in, restore them on a failure."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Hold the container the node was composed with."""
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify the package-tool adapter."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """File system and package operations are effects."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelVenvAtomicInstallRequest
    ) -> ModelVenvAtomicInstallResult:
        """Run one step and return its verdict with the output it wrote."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self._run(request)
        status = (
            "ok"
            if code == 0
            else "refused"
            if code == _REFUSED
            else "declined"
            if code == 4
            else "failed"
        )
        return ModelVenvAtomicInstallResult(
            status=status, exit_code=code, output=out.getvalue() + err.getvalue()
        )

    @staticmethod
    def _run(request: ModelVenvAtomicInstallRequest) -> int:
        if request.command == "verify":
            verify = _load("skill_install_verify")
            return int(verify.verify_nodes() or verify.verify_cli())
        if request.command == "installed_commit":
            return int(_load("skill_install_verify").installed_market_commit())
        if request.command in (
            "staging_lock_acquire",
            "staging_lock_release",
            "staging_lock_held",
            "promote_floor",
        ):
            return int(
                _load("dispatch_venv_promotion").run_promotion_command(
                    request.command,
                    path=request.lock_path,
                    pid=request.holder_pid,
                    venv=request.venv,
                    uv_lock=request.uv_lock,
                    workspace_root=request.workspace_root,
                    floor=request.floor_path,
                    changed=request.changed,
                )
            )
        engine = _load("atomic_venv_install")
        return int(
            engine.run_command(
                request.command,
                python_bin=str(request.python_bin) if request.python_bin else None,
                stage_dir=request.stage_dir,
                requirements=list(request.requirements),
                no_deps=request.no_deps,
                label=request.label,
                paths=[str(p) for p in request.paths],
            )
        )
