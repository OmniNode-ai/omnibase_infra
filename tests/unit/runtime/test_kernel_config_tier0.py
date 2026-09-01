# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tier-0 shipped default runtime config pins (OMN-17304).

The operator ruling makes an unconfigured install CONFIG-RESOLVED: when a
contracts directory has no ``runtime/runtime_config.yaml``, the kernel loads
the SHIPPED tier-0 default (``runtime/tier0_runtime_config.yaml``, packaged in
the wheel) — in-memory bus, local profile — instead of hand-constructed code
defaults. These tests pin:

* the tier-0 file itself (transport pair, topic parity with the kernel
  constants, NO ``name`` — OMN-17287 forbids a fabricated service identity);
* ``load_runtime_config``'s absent-file branch resolving it, with the
  ``ONEX_GROUP_ID`` override still honoured;
* ``resolve_embedded_runtime_config`` — the per-runtime config discovery for
  a CLI-hosted (embedded) runtime: ``ONEX_CONTRACTS_DIR`` bootstrap pointer
  first, shipped tier-0 default otherwise, and DELIBERATELY no cwd tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.cli.cli_delegate import DEFAULT_BUS
from omnibase_infra.runtime.models.enum_event_bus_profile import EnumEventBusProfile
from omnibase_infra.runtime.service_kernel import (
    DEFAULT_GROUP_ID,
    DEFAULT_INPUT_TOPIC,
    DEFAULT_OUTPUT_TOPIC,
    TIER0_RUNTIME_CONFIG_RESOURCE,
    load_runtime_config,
    resolve_embedded_runtime_config,
)

pytestmark = pytest.mark.unit

_TIER0_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / TIER0_RUNTIME_CONFIG_RESOURCE
)


@pytest.fixture(autouse=True)
def _clear_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient pointer/override may leak into tier-0 resolution tests."""
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.delenv("ONEX_EVENT_BUS_TYPE", raising=False)
    monkeypatch.delenv("ONEX_GROUP_ID", raising=False)


class TestTier0ShippedFile:
    """The shipped overlay is the absent-authority default — pin its content."""

    def test_tier0_file_is_packaged_next_to_the_kernel(self) -> None:
        assert _TIER0_PATH.exists(), (
            f"the shipped tier-0 default runtime config is missing at "
            f"{_TIER0_PATH} — an unconfigured install would stop being "
            f"config-resolved (OMN-17304)"
        )

    def test_tier0_declares_inmemory_under_local_profile(self) -> None:
        data = yaml.safe_load(_TIER0_PATH.read_text(encoding="utf-8"))
        assert data["event_bus"]["type"] == "inmemory"
        assert data["event_bus"]["profile"] == "local"

    def test_tier0_topics_mirror_kernel_constants(self) -> None:
        # The absent-file behaviour must be unchanged except for the ruled
        # transport flip — topics and group stay what the code defaults were.
        data = yaml.safe_load(_TIER0_PATH.read_text(encoding="utf-8"))
        assert data["input_topic"] == DEFAULT_INPUT_TOPIC
        assert data["output_topic"] == DEFAULT_OUTPUT_TOPIC
        assert data["group_id"] == DEFAULT_GROUP_ID

    def test_tier0_declares_no_service_identity(self) -> None:
        # OMN-17287: service_name/node_name derive from `name`; a fabricated
        # default identity would let an empty contracts bind-mount boot a lane
        # under an invented service name instead of failing loudly.
        data = yaml.safe_load(_TIER0_PATH.read_text(encoding="utf-8"))
        assert "name" not in data

    def test_cli_default_bus_mirrors_tier0(self) -> None:
        data = yaml.safe_load(_TIER0_PATH.read_text(encoding="utf-8"))
        assert data["event_bus"]["type"] == DEFAULT_BUS


class TestAbsentFileResolvesTier0:
    """load_runtime_config falls back to the shipped overlay, not code defaults."""

    def test_absent_file_resolves_inmemory_local(self, tmp_path: Path) -> None:
        config = load_runtime_config(tmp_path)

        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert config.event_bus.profile is EnumEventBusProfile.LOCAL
        assert config.name is None  # OMN-17287 — identity never fabricated
        assert config.input_topic == DEFAULT_INPUT_TOPIC
        assert config.output_topic == DEFAULT_OUTPUT_TOPIC
        assert config.consumer_group == DEFAULT_GROUP_ID

    def test_absent_file_still_honours_group_id_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_GROUP_ID", "override-group")

        config = load_runtime_config(tmp_path)

        assert config.consumer_group == "override-group"
        assert config.event_bus.type is EnumEventBusType.INMEMORY


class TestResolveEmbeddedRuntimeConfig:
    """Per-runtime config discovery for the CLI's embedded runtime."""

    def test_no_pointer_resolves_tier0_with_provenance(self) -> None:
        config, source = resolve_embedded_runtime_config()

        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert "tier-0" in source
        assert "ONEX_CONTRACTS_DIR" in source

    def test_pointer_with_config_wins_and_names_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contracts_dir = tmp_path / "contracts"
        (contracts_dir / "runtime").mkdir(parents=True)
        config_path = contracts_dir / "runtime" / "runtime_config.yaml"
        config_path.write_text('event_bus:\n  type: "kafka"\n', encoding="utf-8")
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))

        config, source = resolve_embedded_runtime_config()

        assert config.event_bus.type is EnumEventBusType.KAFKA
        assert str(config_path) in source

    def test_pointer_without_config_falls_back_to_tier0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(empty))

        config, source = resolve_embedded_runtime_config()

        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert "tier-0" in source

    def test_cwd_contracts_dir_is_not_a_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transport that flips with `cd` is the environmental accident the
        # ruling removes: a ./contracts directory in the working directory
        # must NOT become the embedded runtime's authority.
        cwd_contracts = tmp_path / "contracts" / "runtime"
        cwd_contracts.mkdir(parents=True)
        (cwd_contracts / "runtime_config.yaml").write_text(
            'event_bus:\n  type: "kafka"\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        config, source = resolve_embedded_runtime_config()

        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert "tier-0" in source
