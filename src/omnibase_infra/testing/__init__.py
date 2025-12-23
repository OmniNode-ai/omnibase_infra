# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for omnibase_infra.

This module provides shared testing utilities for the infrastructure layer,
including CI environment detection and other common test helpers.
"""

from omnibase_infra.testing.utils import is_ci_environment

__all__ = ["is_ci_environment"]
