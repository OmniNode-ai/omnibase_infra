# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-contract parity between the LibYAML and pure-Python loaders [OMN-18843].

Cold contract discovery parses every installed contract with ``CSafeLoader``
when LibYAML is present and falls back to ``SafeLoader`` otherwise. The speed
gain is only safe if both loaders yield the same manifest for the real
contracts on disk, so this runs discovery against the installed entry points
twice and compares the results.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)


@pytest.mark.integration
def test_real_manifest_identical_under_both_yaml_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("LibYAML not installed; only the SafeLoader path exists")

    discover_contracts_cache_clear()
    fast = discover_contracts()

    monkeypatch.delattr(yaml, "CSafeLoader")
    discover_contracts_cache_clear()
    portable = discover_contracts()
    discover_contracts_cache_clear()

    assert fast.contracts, "real discovery found no contracts"
    assert fast.contracts == portable.contracts
    assert fast.errors == portable.errors
