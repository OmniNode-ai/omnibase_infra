# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host I/O injected into estimator build policy by the provisioning effect."""

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolCostEstimatorBuildIO(Protocol):
    def execute_build_command(
        self, argv: list[str], timeout: int, *, allow_failure: bool = False
    ) -> str | None: ...

    def workspace(self) -> AbstractContextManager[str]: ...

    def build_image(self, argv: list[str], timeout: int, identity_path: str) -> str: ...
