# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Allow running as python -m omnibase_infra.docker.catalog."""

if __name__ == "__main__":
    from omnibase_infra.docker.catalog.cli import main

    main()
