# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Example config overlay documents, one per key (OMN-19749).

``runtime.lane.example.json`` is the ``runtime.lane`` document of a
single-machine install: lane ``local``, no roles. A runtime never reads it.
``onex local init`` copies it into the local-home overlay directory
(``~/.omninode/config/<environment>/<lane>/runtime.lane.json``), and an operator
running their own deployment copies it into their overlay source under the lane
id they choose. Its schema is core's ``runtime_lane`` (see
``EnumConfigOverlayKey.RUNTIME_LANE.schema_ref``).
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final

__all__ = ["RUNTIME_LANE_EXAMPLE_NAME", "read_runtime_lane_example"]

#: File name of the example ``runtime.lane`` document inside this package.
RUNTIME_LANE_EXAMPLE_NAME: Final[str] = "runtime.lane.example.json"


def read_runtime_lane_example() -> bytes:
    """Return the example ``runtime.lane`` document's bytes, exactly as shipped."""
    return files(__name__).joinpath(RUNTIME_LANE_EXAMPLE_NAME).read_bytes()
