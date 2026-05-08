# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.service_kernel import load_runtime_config

pytestmark = pytest.mark.integration


def test_runtime_config_enables_main_profile_pattern_b_ingress() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    config = load_runtime_config(repo_root / "contracts")

    assert config.local_ingress.enabled is True
    assert config.local_ingress.socket_path == "/run/onex-runtime/onex-runtime.sock"
    assert config.local_ingress.package_names == ("omnibase_infra", "omnimarket")
    assert config.local_ingress.enabled_profiles == ("main",)
    assert config.pattern_b_broker.enabled is True
    assert (
        config.pattern_b_broker.command_topic
        == "onex.cmd.omnimarket.pattern-b-dispatch.v1"
    )
    assert config.pattern_b_broker.package_names == ("omnibase_infra", "omnimarket")
    assert config.pattern_b_broker.enabled_profiles == ("main",)
