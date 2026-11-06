# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The import path omniintelligence's review pairing still loads (OMN-22478).

omnibase_infra#5683 moved this model to ``omnibase_infra.models.resilience``. The
move is right and stays. Delete this module once omniintelligence imports the new
path.
"""

from omnibase_infra.models.resilience.model_retry_state import ModelRetryState

__all__: list[str] = ["ModelRetryState"]
