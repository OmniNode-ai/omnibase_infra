# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent target base URIs declared in ``intent_target_uris.yaml`` (OMN-21792)."""

from omnibase_infra.models.intent_target_uris.model_intent_target_uris import (
    ModelIntentTargetUris,
    load_intent_target_uris,
)

__all__ = ["ModelIntentTargetUris", "load_intent_target_uris"]
