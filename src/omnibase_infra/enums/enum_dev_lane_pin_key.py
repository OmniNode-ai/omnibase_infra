# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane image pins the dev lane pin handler may read or set (OMN-18475)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLanePinKey(StrEnum):
    """An image key the compose dev lane renders from the operator env store.

    Adding a member is a deliberate act: ``test_handler_dev_lane_pin_omn18475``
    fails when a member is rendered by a compose file of a governed lane, so a
    dev pin can never reach a governed container through the shared store.
    """

    ONEX_API_IMAGE = "ONEX_API_IMAGE"
    """The ``onex-api`` image of the compose dev lane."""

    ONEX_CLOUD_MIGRATE_IMAGE = "ONEX_CLOUD_MIGRATE_IMAGE"
    """The cloud corpus migration image of the compose dev lane."""
