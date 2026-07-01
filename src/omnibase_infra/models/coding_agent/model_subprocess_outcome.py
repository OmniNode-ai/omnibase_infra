# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent subprocess outcome seam model."""

from __future__ import annotations


class ModelSubprocessOutcome:
    """Plain result of one subprocess run, returned by the subprocess seam."""

    __slots__ = ("returncode", "stderr", "stdout", "timed_out")

    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


__all__ = ["ModelSubprocessOutcome"]
