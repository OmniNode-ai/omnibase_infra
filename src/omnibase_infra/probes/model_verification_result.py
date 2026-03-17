# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for verification execution."""

from pydantic import BaseModel, Field


class ModelVerificationResult(BaseModel):
    """Result of executing a single verification check.

    Attributes:
        passed: Whether the verification succeeded.
        check_type: The check type that was executed.
        target: The target that was checked.
        message: Human-readable result description.
        elapsed_ms: Time taken for the check in milliseconds.
    """

    passed: bool = Field(description="Whether the verification succeeded")
    check_type: str = Field(description="The check type that was executed")
    target: str = Field(description="The target that was checked")
    message: str = Field(description="Human-readable result description")
    elapsed_ms: int = Field(description="Time taken in milliseconds")


__all__ = ["ModelVerificationResult"]
