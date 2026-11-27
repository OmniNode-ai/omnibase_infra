# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's way in to the build repository ref handler (OMN-26724).

The agent runs from its own checkout beside a published ``omnibase_infra`` that does not carry
this handler, so it loads the handler and its models by path. ``BuildRepositoryRefCheck`` reads
the clone through the runner the agent hands it (cat-file, one fetch on a miss) and returns the handler's decision.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol

logger = logging.getLogger(__name__)


class _Completed(Protocol):
    returncode: int


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_MODULES = (
    "enums/enum_build_repository_ref_verdict.py",
    "models/model_build_repository_ref_request.py",
    "models/model_build_repository_ref_decision.py",
    "handlers/build_repository_ref/handler_build_repository_ref.py",
)
_GIT_SECONDS = 30
_FETCH_SECONDS = 120


def _load() -> tuple[ModuleType, ModuleType]:
    for relative in _MODULES:
        name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, _PACKAGE_ROOT / relative
            )
            assert spec is not None and spec.loader is not None
            sys.modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules[name])
    return (
        sys.modules["omnibase_infra.models.model_build_repository_ref_request"],
        sys.modules[
            "omnibase_infra.handlers.build_repository_ref.handler_build_repository_ref"
        ],
    )


class BuildRepositoryRefCheck:
    """Whether the deploy clone holds a commit; one fetch on a miss, then the handler decides."""

    def __init__(
        self,
        repo_dir: str,
        run: Callable[[list[str], int], _Completed],
    ) -> None:
        self.repo_dir = repo_dir
        self._run = run

    def _holds(self, git_ref: str) -> bool:
        return (
            self._run(
                ["git", "-C", self.repo_dir, "cat-file", "-e", f"{git_ref}^{{commit}}"],
                _GIT_SECONDS,
            ).returncode
            == 0
        )

    def __call__(self, git_ref: str) -> object:
        request_module, handler_module = _load()
        handler = handler_module.HandlerBuildRepositoryRef()
        if self._holds(git_ref):
            return handler.handle(
                request_module.ModelBuildRepositoryRefRequest(
                    git_ref=git_ref, held=True
                )
            )
        try:
            fetch = self._run(
                ["git", "-C", self.repo_dir, "fetch", "--quiet", "--no-tags", "origin"],
                _FETCH_SECONDS,
            )
            fetched = fetch.returncode
        except Exception:
            logger.exception("fetch for %s did not answer", git_ref)
            fetched = None
        return handler.handle(
            request_module.ModelBuildRepositoryRefRequest(
                git_ref=git_ref,
                held=False,
                fetch_exit_code=fetched,
                held_after_fetch=fetched == 0 and self._holds(git_ref),
            )
        )
