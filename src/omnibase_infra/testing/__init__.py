# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for omnibase_infra.

Shared testing utilities for the infrastructure layer,
including CI environment detection, effect mock registry,
and thread-local registry helpers.
"""

from omnibase_infra.testing.service_effect_mock_registry import (
    EffectMockRegistry,
)
from omnibase_infra.testing.service_effect_mock_registry_thread_local import (
    clear_thread_local_registry,
    get_thread_local_registry,
    scoped_effect_mock_registry,
)
from omnibase_infra.testing.utils import is_ci_environment

__all__ = [
    "EffectMockRegistry",
    "clear_thread_local_registry",
    "get_thread_local_registry",
    "is_ci_environment",
    "scoped_effect_mock_registry",
]
