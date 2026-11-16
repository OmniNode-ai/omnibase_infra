# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One Docker healthcheck probe entry."""

from typing_extensions import TypedDict


class ModelDevBootHealthProbe(TypedDict, total=False):
    Start: str
    End: str
    ExitCode: int
    Output: str
