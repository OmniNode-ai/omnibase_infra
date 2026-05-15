# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Conftest for overlay unit tests.

Inserts the omnibase_core overlay-config worktree into sys.path so that
ModelOverlayFile and ModelOverlayResolutionManifest are importable before their
PR merges to main and gets a new package release.
"""

from __future__ import annotations

import sys
from pathlib import Path

# omnibase_core overlay models are built in the Wave-1 worktree.
# Once PR#1082 merges and a new release is cut, this path insertion can be removed.
_CORE_WORKTREE_SRC = Path(
    "/Users/jonah/Code/omni_home/omni_worktrees/overlay-config/omnibase_core/src"  # local-path-ok: Wave-1 worktree dependency during pre-release
)
if _CORE_WORKTREE_SRC.exists() and str(_CORE_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_CORE_WORKTREE_SRC))
