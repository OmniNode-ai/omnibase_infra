# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: kernel hard-fails at startup when deprecated topic env vars are set.

OMN-8784: ONEX_INPUT_TOPIC and ONEX_OUTPUT_TOPIC were removed from the kernel.
Topics must be declared in node contract event_bus.subscribe_topics /
event_bus.publish_topics. Setting the deprecated vars now raises
ProtocolConfigurationError at startup — both with a config file and without.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.service_kernel import load_runtime_config

pytestmark = pytest.mark.integration


class TestKernelDeprecatedTopicEnvVars:
    """OMN-8784: Verify kernel rejects deprecated topic env vars at startup."""

    @pytest.mark.parametrize(
        "deprecated_var", ["ONEX_INPUT_TOPIC", "ONEX_OUTPUT_TOPIC"]
    )
    def test_hard_fail_no_config_path(
        self,
        deprecated_var: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Kernel raises ProtocolConfigurationError when deprecated topic env var is set
        and no config file exists (no-config path still enforces the guard).
        """
        monkeypatch.setenv(deprecated_var, "some-topic-value")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path / "contracts")

        assert deprecated_var in str(exc_info.value)
        assert "OMN-8784" in str(exc_info.value)

    @pytest.mark.parametrize(
        "deprecated_var", ["ONEX_INPUT_TOPIC", "ONEX_OUTPUT_TOPIC"]
    )
    def test_succeeds_when_env_var_absent(
        self,
        deprecated_var: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Kernel loads successfully with defaults when deprecated topic env vars are unset."""
        monkeypatch.delenv(deprecated_var, raising=False)

        config = load_runtime_config(tmp_path / "contracts")
        assert config is not None
