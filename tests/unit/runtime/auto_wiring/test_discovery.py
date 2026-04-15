# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for contract auto-discovery engine."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import (
    _parse_contract,
    _resolve_contract_path,
    discover_contracts,
    discover_contracts_from_paths,
)

_EP_MODULE = "omnibase_infra.runtime.auto_wiring.discovery.entry_points"


def _make_contract_yaml(
    tmp_path: Path,
    *,
    name: str = "node_test_effect",
    node_type: str = "EFFECT_GENERIC",
    with_event_bus: bool = False,
) -> Path:
    """Write a minimal contract.yaml and return its path."""
    content = dedent(f"""\
        name: "{name}"
        node_type: "{node_type}"
        contract_version:
          major: 1
          minor: 2
          patch: 3
        node_version: "2.0.0"
        description: "A test node"
    """)
    if with_event_bus:
        content += dedent("""\
            event_bus:
              subscribe_topics:
                - "onex.evt.platform.test-input.v1"
              publish_topics:
                - "onex.evt.platform.test-output.v1"
        """)

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(content)
    return contract_path


def _make_entry_point(
    name: str,
    *,
    node_cls: type | None = None,
    dist_name: str = "my-plugin",
    dist_version: str = "1.0.0",
    load_raises: Exception | None = None,
) -> MagicMock:
    """Create a mock entry point."""
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    ep.dist.version = dist_version
    if load_raises is not None:
        ep.load.side_effect = load_raises
    elif node_cls is not None:
        ep.load.return_value = node_cls
    else:
        cls = type("FakeNode", (), {"process": lambda self: None})
        ep.load.return_value = cls
    return ep


class TestParseContract:
    """Tests for _parse_contract."""

    @pytest.mark.unit
    def test_parses_minimal_contract(self, tmp_path: Path) -> None:
        contract_path = _make_contract_yaml(tmp_path)
        result = _parse_contract(
            contract_path=contract_path,
            entry_point_name="test_node",
            package_name="test-pkg",
            package_version="1.0.0",
        )
        assert result.name == "node_test_effect"
        assert result.node_type == "EFFECT_GENERIC"
        assert str(result.contract_version) == "1.2.3"
        assert result.node_version == "2.0.0"
        assert result.entry_point_name == "test_node"
        assert result.package_name == "test-pkg"
        assert result.event_bus is None

    @pytest.mark.unit
    def test_parses_event_bus_wiring(self, tmp_path: Path) -> None:
        contract_path = _make_contract_yaml(tmp_path, with_event_bus=True)
        result = _parse_contract(
            contract_path=contract_path,
            entry_point_name="test_node",
            package_name="test-pkg",
            package_version="1.0.0",
        )
        assert result.event_bus is not None
        assert result.event_bus.subscribe_topics == ("onex.evt.platform.test-input.v1",)
        assert result.event_bus.publish_topics == ("onex.evt.platform.test-output.v1",)

    @pytest.mark.unit
    def test_raises_on_non_dict_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "contract.yaml"
        path.write_text("- just a list")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            _parse_contract(
                contract_path=path,
                entry_point_name="bad",
                package_name="pkg",
                package_version="1.0.0",
            )


class TestResolveContractPath:
    """Tests for _resolve_contract_path."""

    @pytest.mark.unit
    def test_resolves_explicit_contract_path(self, tmp_path: Path) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("name: test")
        cls = type("MyNode", (), {"contract_path": str(contract_path)})
        result = _resolve_contract_path(cls)
        assert result == contract_path

    @pytest.mark.unit
    def test_resolves_sibling_contract(self, tmp_path: Path) -> None:
        # Create a fake module file and sibling contract.yaml
        module_file = tmp_path / "node.py"
        module_file.write_text("")
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("name: test")

        cls = type("MyNode", (), {})
        with patch("inspect.getfile", return_value=str(module_file)):
            result = _resolve_contract_path(cls)
        assert result == contract_file

    @pytest.mark.unit
    def test_raises_when_no_contract_found(self, tmp_path: Path) -> None:
        module_file = tmp_path / "subdir" / "node.py"
        module_file.parent.mkdir()
        module_file.write_text("")

        cls = type("MyNode", (), {})
        with patch("inspect.getfile", return_value=str(module_file)):
            with pytest.raises(FileNotFoundError, match=r"No contract\.yaml found"):
                _resolve_contract_path(cls)

    @pytest.mark.unit
    def test_resolves_namespace_package_contract(self, tmp_path: Path) -> None:
        # Namespace packages have __path__ but no __file__ — inspect.getfile
        # raises TypeError for them. _resolve_contract_path must fall back to
        # searching __path__ entries.
        pkg_dir = tmp_path / "node_namespace_pkg"
        pkg_dir.mkdir()
        contract_file = pkg_dir / "contract.yaml"
        contract_file.write_text("name: test")

        # Simulate a namespace module: has __path__ but raises on getfile
        import types

        ns_mod = types.ModuleType("node_namespace_pkg")
        ns_mod.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        # Do NOT set __file__ — simulates namespace package

        result = _resolve_contract_path(ns_mod)  # type: ignore[arg-type]
        assert result == contract_file

    @pytest.mark.unit
    def test_namespace_package_without_contract_raises(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "node_empty_ns"
        pkg_dir.mkdir()
        # No contract.yaml in pkg_dir

        import types

        ns_mod = types.ModuleType("node_empty_ns")
        ns_mod.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]

        with pytest.raises(FileNotFoundError, match=r"No contract\.yaml found"):
            _resolve_contract_path(ns_mod)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_discover_contracts_tolerates_namespace_package_entry_points(
        self, tmp_path: Path
    ) -> None:
        # An entry point that loads a namespace module (no __file__) must be
        # captured as an error rather than aborting the entire discovery scan.
        import types

        ns_mod = types.ModuleType("node_ns_no_contract")
        ns_mod.__path__ = [str(tmp_path / "nonexistent")]  # type: ignore[attr-defined]

        ep = _make_entry_point("ns_node", node_cls=None)
        ep.load.return_value = ns_mod

        with patch(_EP_MODULE, return_value=[ep]):
            manifest = discover_contracts()

        assert manifest.total_discovered == 0
        assert manifest.total_errors == 1


class TestDiscoverContracts:
    """Tests for discover_contracts (entry-point based)."""

    @pytest.mark.unit
    def test_returns_empty_manifest_when_no_entry_points(self) -> None:
        with patch(_EP_MODULE, return_value=[]):
            manifest = discover_contracts()
        assert manifest.total_discovered == 0
        assert manifest.total_errors == 0

    @pytest.mark.unit
    def test_captures_load_failure_as_error(self) -> None:
        ep = _make_entry_point("bad", load_raises=ImportError("no module"))
        with patch(_EP_MODULE, return_value=[ep]):
            manifest = discover_contracts()
        assert manifest.total_discovered == 0
        assert manifest.total_errors == 1
        assert "Failed to load entry point" in manifest.errors[0].error

    @pytest.mark.unit
    def test_discovers_contract_from_entry_point(self, tmp_path: Path) -> None:
        _make_contract_yaml(tmp_path, name="my_effect", node_type="EFFECT_GENERIC")
        module_file = tmp_path / "node.py"
        module_file.write_text("")

        cls = type("MyEffect", (), {"process": lambda self: None})
        ep = _make_entry_point("my_effect", node_cls=cls)

        with (
            patch(_EP_MODULE, return_value=[ep]),
            patch("inspect.getfile", return_value=str(module_file)),
        ):
            manifest = discover_contracts()

        assert manifest.total_discovered == 1
        assert manifest.total_errors == 0
        assert manifest.contracts[0].name == "my_effect"
        assert manifest.contracts[0].node_type == "EFFECT_GENERIC"

    @pytest.mark.unit
    def test_captures_missing_contract_as_error(self, tmp_path: Path) -> None:
        # Module dir exists but no contract.yaml
        module_file = tmp_path / "subdir" / "node.py"
        module_file.parent.mkdir()
        module_file.write_text("")

        cls = type("OrphanNode", (), {"process": lambda self: None})
        ep = _make_entry_point("orphan", node_cls=cls)

        with (
            patch(_EP_MODULE, return_value=[ep]),
            patch("inspect.getfile", return_value=str(module_file)),
        ):
            manifest = discover_contracts()

        assert manifest.total_discovered == 0
        assert manifest.total_errors == 1
        assert "No contract.yaml found" in manifest.errors[0].error


class TestDiscoverContractsFromPaths:
    """Tests for discover_contracts_from_paths."""

    @pytest.mark.unit
    def test_discovers_from_explicit_paths(self, tmp_path: Path) -> None:
        # Create two contract files in separate dirs
        dir_a = tmp_path / "node_a"
        dir_a.mkdir()
        path_a = _make_contract_yaml(dir_a, name="node_a", node_type="EFFECT_GENERIC")

        dir_b = tmp_path / "node_b"
        dir_b.mkdir()
        path_b = _make_contract_yaml(
            dir_b, name="node_b", node_type="REDUCER_GENERIC", with_event_bus=True
        )

        manifest = discover_contracts_from_paths([path_a, path_b])

        assert manifest.total_discovered == 2
        assert manifest.total_errors == 0
        names = {c.name for c in manifest.contracts}
        assert names == {"node_a", "node_b"}

    @pytest.mark.unit
    def test_captures_invalid_path_as_error(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "nonexistent" / "contract.yaml"
        manifest = discover_contracts_from_paths([bad_path])
        assert manifest.total_discovered == 0
        assert manifest.total_errors == 1


class TestModelAutoWiringManifest:
    """Tests for manifest query methods."""

    @pytest.mark.unit
    def test_get_by_node_type(self, tmp_path: Path) -> None:
        dir_e = tmp_path / "effect"
        dir_e.mkdir()
        path_e = _make_contract_yaml(dir_e, name="eff", node_type="EFFECT_GENERIC")

        dir_r = tmp_path / "reducer"
        dir_r.mkdir()
        path_r = _make_contract_yaml(dir_r, name="red", node_type="REDUCER_GENERIC")

        manifest = discover_contracts_from_paths([path_e, path_r])
        effects = manifest.get_by_node_type("EFFECT_GENERIC")
        assert len(effects) == 1
        assert effects[0].name == "eff"

    @pytest.mark.unit
    def test_get_all_topics(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "node_a"
        dir_a.mkdir()
        path_a = _make_contract_yaml(dir_a, name="node_a", with_event_bus=True)

        manifest = discover_contracts_from_paths([path_a])
        assert "onex.evt.platform.test-input.v1" in manifest.get_all_subscribe_topics()
        assert "onex.evt.platform.test-output.v1" in manifest.get_all_publish_topics()

    @pytest.mark.unit
    def test_protocol_method_all_subscribe_topics(self, tmp_path: Path) -> None:
        """OMN-8854: ModelAutoWiringManifest must expose all_subscribe_topics()
        to satisfy ProtocolAutoWiringManifestLike — the health monitor calls
        this method directly and raises AttributeError when it is absent."""
        dir_a = tmp_path / "node_a"
        dir_a.mkdir()
        path_a = _make_contract_yaml(dir_a, name="node_a", with_event_bus=True)

        manifest = discover_contracts_from_paths([path_a])
        # Must not raise AttributeError
        topics = manifest.all_subscribe_topics()
        assert "onex.evt.platform.test-input.v1" in topics
