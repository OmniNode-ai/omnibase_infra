# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent subprocess invocation seam model.

Bundles the inputs of one subprocess run (argv, cwd, timeout, network posture,
optional piped stdin, and the EXPLICIT child env) into a single seam object so the
``SubprocessRunner`` seam takes one parameter. The explicit ``env`` carries
``HOME`` overridden to the contract-resolved credential home so the spawned
claude/codex subprocess finds its ambient OAuth creds even when the runtime runs
as root (OMN-13247 Phase B dev-verify fix).
"""

from __future__ import annotations

from collections.abc import Mapping


class ModelSubprocessInvocation:
    """Plain inputs for one subprocess run, accepted by the subprocess seam."""

    __slots__ = ("argv", "cwd", "env", "network", "stdin", "timeout_s")

    def __init__(
        self,
        *,
        argv: list[str],
        cwd: str,
        timeout_s: int,
        network: bool,
        stdin: str | None,
        env: Mapping[str, str],
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.network = network
        self.stdin = stdin
        self.env = env


__all__ = ["ModelSubprocessInvocation"]
