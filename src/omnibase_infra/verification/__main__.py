# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Allow running verification as: python -m omnibase_infra.verification."""

import sys

from omnibase_infra.verification.cli import main

if __name__ == "__main__":
    sys.exit(main())
