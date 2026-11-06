# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One bounded sampling invocation of the existing collector."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_infra.models.model_census_oneshot_policy import ModelCensusOneshotPolicy

# The collector's own command line (scripts/lane-census-check.sh). A request arrives on the
# command topic, so nothing outside this grammar reaches the child process.
_FLAGS = frozenset({"--dry-run", "--json", "--memory", "--help", "-h"})
_LANE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PATH_OPTIONS = frozenset({"--snapshot", "--observed-out", "--memory-event-out"})
_UNSAFE_PATH = re.compile(r"[\x00-\x1f\x7f]|(?:^|/)\.\.(?:/|$)")


class ModelCensusOneshotRequest(BaseModel):
    """One bounded sampling invocation of the existing collector.

    The collector directory is not a field: it is bound by the adapter the collector started,
    never something a command payload names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    operation: Literal["census_oneshot.sample"] = "census_oneshot.sample"
    policy: ModelCensusOneshotPolicy
    arguments: tuple[str, ...] = ()

    @field_validator("arguments")
    @classmethod
    def _only_the_collectors_options(
        cls, arguments: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Refuse any argument the collector's own option grammar does not define."""
        position = 0
        while position < len(arguments):
            option = arguments[position]
            position += 1
            if option in _FLAGS:
                continue
            if option != "--lane" and option not in _PATH_OPTIONS:
                raise ValueError(
                    f"census argument {option!r} is not a collector option"
                )
            if position >= len(arguments):
                raise ValueError(f"census option {option} requires a value")
            value = arguments[position]
            position += 1
            if option == "--lane":
                ok = _LANE_VALUE.fullmatch(value) is not None
            else:
                ok = value == "-" or (
                    bool(value)
                    and not value.startswith("-")
                    and _UNSAFE_PATH.search(value) is None
                )
            if not ok:
                raise ValueError(f"census option {option} has an unacceptable value")
        return arguments
