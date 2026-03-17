# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocol for verification specifications."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VerificationSpec(Protocol):
    """Protocol for verification specifications.

    Compatible with ModelStepVerification from omnibase_core and
    ModelVerificationSpec from this package.
    """

    @property
    def check_type(self) -> str: ...

    @property
    def target(self) -> str: ...

    @property
    def timeout_seconds(self) -> int: ...


__all__ = ["VerificationSpec"]
