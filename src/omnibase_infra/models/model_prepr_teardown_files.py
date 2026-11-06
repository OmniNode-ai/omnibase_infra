# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Filesystem callables injected by the host teardown composition root."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPreprTeardownFiles:
    """Concrete dependencies for the existing host-local teardown handler."""

    output: Callable[[str], object]
    read: Callable[[Path], str] = Path.read_text
    write: Callable[[Path, str], object] = Path.write_text
    present: Callable[[Path], bool] = Path.exists
