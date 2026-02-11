# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler that collects build-toolchain versions for RRH validation.

Detects installed versions of pre-commit, ruff, pytest, and mypy by
running ``<tool> --version`` via ``create_subprocess_exec`` (no shell).
"""

from __future__ import annotations

import asyncio
import logging
import re

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.rrh.model_rrh_toolchain_versions import (
    ModelRRHToolchainVersions,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


class HandlerToolchainCollect:
    """Collect build-tool versions.

    Gathers: pre_commit, ruff, pytest, mypy version strings.
    Returns empty string for any tool that is not installed or errors.

    Attributes:
        handler_type: ``INFRA_HANDLER``
        handler_category: ``EFFECT``
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self) -> ModelRRHToolchainVersions:
        """Collect toolchain versions in parallel.

        Returns:
            Populated ``ModelRRHToolchainVersions``.
        """
        pre_commit, ruff, pytest_ver, mypy = await asyncio.gather(
            self._version("pre-commit"),
            self._version("ruff"),
            self._version("pytest"),
            self._version("mypy"),
        )
        return ModelRRHToolchainVersions(
            pre_commit=pre_commit,
            ruff=ruff,
            pytest=pytest_ver,
            mypy=mypy,
        )

    @staticmethod
    async def _version(tool: str) -> str:
        """Run ``<tool> --version`` via create_subprocess_exec and extract version.

        Uses create_subprocess_exec (not shell) for safety — the tool
        name is from a fixed set, not from user input.

        Returns empty string on failure.
        """
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                tool,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            if proc.returncode != 0:
                return ""
            output = stdout.decode(errors="replace").strip()
            match = _VERSION_RE.search(output)
            return match.group(1) if match else ""
        except TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.debug("%s --version error: timed out after 10s", tool)
            return ""
        except (FileNotFoundError, OSError) as exc:
            logger.debug(
                "%s --version error: %s", tool, sanitize_error_string(str(exc))
            )
            return ""


__all__: list[str] = ["HandlerToolchainCollect"]
