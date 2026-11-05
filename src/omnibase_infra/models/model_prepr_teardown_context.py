# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected host command adapter and private environment for cleanup."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from subprocess import CompletedProcess

from omnibase_infra.models.model_prepr_teardown_files import (
    ModelPreprTeardownFiles,
)

PreprTeardownRunner = Callable[
    [list[str], dict[str, str] | None], CompletedProcess[str]
]


@dataclass(frozen=True)
class ModelPreprTeardownContext:
    """Host dependencies registered by the existing CLI's composition root."""

    runner: PreprTeardownRunner
    environment: Mapping[str, str] = field(repr=False)
    files: ModelPreprTeardownFiles
