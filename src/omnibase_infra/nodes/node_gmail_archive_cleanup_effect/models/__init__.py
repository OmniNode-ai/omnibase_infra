# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for the Gmail Archive Cleanup Effect node."""

from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_config import (
    ModelGmailCleanupConfig,
)
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_result import (
    ModelGmailCleanupResult,
)

__all__ = ["ModelGmailCleanupConfig", "ModelGmailCleanupResult"]
