# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One broker resource permission."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

WILDCARD_RESOURCE = "*"


class ModelBrokerGrant(BaseModel):
    """A resource grant, exact by default.

    ``pattern`` is ``LITERAL`` for one named resource and ``PREFIXED`` for every
    resource whose name starts with ``resource``. The only wildcard is the literal
    ``*`` group carrying DESCRIBE; it is stated by a described group and never
    inferred from a binding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str = Field(min_length=1)
    resource_type: Literal["TOPIC", "GROUP"]
    resource: str = Field(min_length=1)
    operation: Literal["READ", "WRITE", "DESCRIBE"]
    pattern: Literal["LITERAL", "PREFIXED"] = "LITERAL"

    @model_validator(mode="after")
    def validate_wildcard(self) -> Self:
        """Allow ``*`` only as a literal group with DESCRIBE."""
        if self.resource == WILDCARD_RESOURCE and (
            self.pattern != "LITERAL"
            or self.resource_type != "GROUP"
            or self.operation != "DESCRIBE"
        ):
            raise ValueError("'*' is a literal group grant carrying DESCRIBE only")
        return self
