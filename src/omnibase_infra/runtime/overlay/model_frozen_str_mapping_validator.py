# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic annotation for coercing Mapping[str, str] to an immutable MappingProxyType."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ModelFrozenStrMappingValidator:
    """Pydantic annotation that coerces any Mapping[str, str] to MappingProxyType.

    Use as an Annotated metadata argument to enforce deep immutability on
    dict-valued fields in frozen Pydantic models, where ConfigDict(frozen=True)
    only prevents attribute reassignment on the model instance itself.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, v: object) -> MappingProxyType[str, str]:
        if isinstance(v, MappingProxyType):
            return v
        if isinstance(v, Mapping):
            return MappingProxyType({str(k): str(val) for k, val in v.items()})
        msg = f"resolved must be a Mapping, got {type(v).__name__}"
        raise TypeError(msg)


__all__ = ["ModelFrozenStrMappingValidator"]
