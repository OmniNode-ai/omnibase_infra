# SPDX-License-Identifier: Apache-2.0
"""Shared testing utilities for omnibase_infra.

This module provides utility functions commonly needed across test files,
such as environment detection and test configuration helpers.
"""

import os


def is_ci_environment() -> bool:
    """Detect if running in CI environment.

    Checks common CI environment variables:
    - CI: Generic CI flag used by most CI systems
    - GITHUB_ACTIONS: GitHub Actions specific

    Returns:
        True if running in a CI environment, False otherwise.
    """
    ci_value = os.getenv("CI", "").lower()
    github_actions = os.getenv("GITHUB_ACTIONS", "").lower()
    return ci_value in ("true", "1", "yes") or github_actions in ("true", "1", "yes")
