# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for BindingConfigResolver security controls.

This module tests system-level security controls for the BindingConfigResolver,
focusing on path traversal protection with real filesystem operations.

Test Coverage:
- TestPathTraversalIntegration: Real filesystem path traversal attacks
- TestSymlinkSecurityIntegration: Symlink-based path traversal attempts
- TestPathTraversalEdgeCases: Edge cases and boundary conditions

Unlike unit tests that may mock filesystem operations, these integration tests:
1. Use real temporary directories and files
2. Create actual symlinks (where supported by the OS)
3. Test interactions between path validation layers
4. Verify error messages don't leak sensitive path information

Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168: Security enhancements
    - docs/patterns/binding_config_resolver.md#path-traversal-protection
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

if TYPE_CHECKING:
    from collections.abc import Generator


def create_mock_container(config: ModelBindingConfigResolverConfig) -> MagicMock:
    """Create a mock container with the given config registered.

    This helper creates a mock ModelONEXContainer with the required
    service_registry.resolve_service() behavior for BindingConfigResolver.
    """
    container = MagicMock()

    service_map: dict[type, object] = {
        ModelBindingConfigResolverConfig: config,
    }

    def resolve_service_side_effect(service_type: type) -> object:
        if service_type in service_map:
            return service_map[service_type]
        raise KeyError(f"Service {service_type} not registered")

    container.service_registry.resolve_service.side_effect = resolve_service_side_effect
    return container


class TestPathTraversalIntegration:
    """Integration tests for path traversal protection with real filesystem.

    These tests verify that path traversal attacks are blocked when using
    actual filesystem operations, not mocked file I/O.
    """

    def test_parent_directory_traversal_blocked(self, tmp_path: Path) -> None:
        """Verify ../ traversal is blocked with real filesystem.

        Creates a real directory structure and verifies that attempts to
        escape the config_dir using ../ are blocked.
        """
        # Create directory structure
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        # Create a "secret" file outside config_dir
        secret_file = secret_dir / "credentials.yaml"
        secret_file.write_text("handler_type: secret\npassword: super_secret\n")

        # Create a legitimate config inside config_dir
        legit_config = config_dir / "handler.yaml"
        legit_config.write_text("handler_type: db\ntimeout_ms: 5000\n")

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Attempt to access secret file via path traversal
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../secrets/credentials.yaml",
            )

        # Verify error message indicates path traversal was blocked
        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "not allowed" in error_msg

        # Verify the actual path is NOT leaked in the error message
        assert "super_secret" not in str(exc_info.value)
        assert str(secret_dir) not in str(exc_info.value)

    def test_multiple_parent_directory_traversal_blocked(self, tmp_path: Path) -> None:
        """Verify multiple ../ sequences are blocked.

        Tests that deeply nested traversal attempts like ../../../ are blocked.
        Multiple consecutive parent directory sequences are blocked at the
        parsing layer (ModelConfigRef.parse) before resolution, resulting in
        an "invalid config reference" error rather than a "path traversal" error.
        """
        # Create deeply nested config directory
        config_dir = tmp_path / "app" / "configs" / "handlers"
        config_dir.mkdir(parents=True)

        # Create sensitive file at root
        sensitive_file = tmp_path / "etc" / "passwd"
        sensitive_file.parent.mkdir(parents=True, exist_ok=True)
        sensitive_file.write_text("handler_type: passwd\nroot:x:0:0\n")

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Attempt deep traversal - blocked at parsing layer
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../../../etc/passwd",
            )

        # Multiple traversal sequences blocked at parsing or resolution layer
        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "invalid" in error_msg

    def test_traversal_with_absolute_path_still_requires_config_dir(
        self, tmp_path: Path
    ) -> None:
        """Verify absolute paths are validated against config_dir.

        Even with absolute paths, the resolved path must be within config_dir.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()
        secret_file = secret_dir / "creds.yaml"
        secret_file.write_text("handler_type: secret\n")

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Attempt to use absolute path outside config_dir
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref=f"file:///{secret_file}",
            )

        # Should be blocked
        error_msg = str(exc_info.value).lower()
        assert (
            "traversal" in error_msg
            or "not allowed" in error_msg
            or "not found" in error_msg
        )

    def test_legitimate_relative_path_succeeds(self, tmp_path: Path) -> None:
        """Verify legitimate relative paths work correctly.

        Ensures security measures don't break normal operation.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        subdir = config_dir / "handlers"
        subdir.mkdir()

        config_file = subdir / "db.yaml"
        config_file.write_text("handler_type: db\ntimeout_ms: 5000\n")

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Legitimate relative path should work
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:handlers/db.yaml",
        )

        assert result.handler_type == "db"
        assert result.timeout_ms == 5000


class TestSymlinkSecurityIntegration:
    """Integration tests for symlink-based path traversal protection.

    These tests verify that symlinks cannot be used to escape the config_dir
    boundary, even when symlinks are allowed.
    """

    @pytest.fixture
    def symlink_capable(self) -> Generator[bool, None, None]:
        """Check if the current system supports symlinks.

        On Windows, symlinks may require admin privileges or developer mode.
        """
        can_symlink = True
        if platform.system() == "Windows":
            # Check if we can create symlinks on Windows
            try:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    test_target = Path(tmpdir) / "target"
                    test_target.write_text("test")
                    test_link = Path(tmpdir) / "link"
                    test_link.symlink_to(test_target)
            except OSError:
                can_symlink = False

        return can_symlink

    def test_symlink_outside_config_dir_blocked(
        self, tmp_path: Path, symlink_capable: bool
    ) -> None:
        """Verify symlinks pointing outside config_dir are blocked.

        Even when allow_symlinks=True, symlinks that resolve to paths
        outside config_dir should be blocked.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        # Create secret file
        secret_file = secret_dir / "credentials.yaml"
        secret_file.write_text("handler_type: secret\npassword: leaked\n")

        # Create symlink inside config_dir pointing outside
        evil_link = config_dir / "evil.yaml"
        evil_link.symlink_to(secret_file)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,  # Symlinks allowed but must stay in config_dir
        )
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Should be blocked by path traversal protection
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:evil.yaml",
            )

        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "symlink" in error_msg

        # Password should not be in error message
        assert "leaked" not in str(exc_info.value)

    def test_symlink_inside_config_dir_allowed(
        self, tmp_path: Path, symlink_capable: bool
    ) -> None:
        """Verify symlinks within config_dir are allowed when enabled.

        Symlinks that point to other files within config_dir should work
        when allow_symlinks=True.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create real config
        real_config = config_dir / "real.yaml"
        real_config.write_text("handler_type: db\ntimeout_ms: 3000\n")

        # Create symlink to it (both inside config_dir)
        link_config = config_dir / "link.yaml"
        link_config.symlink_to(real_config)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,
        )
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Should succeed
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:link.yaml",
        )

        assert result.handler_type == "db"
        assert result.timeout_ms == 3000

    def test_symlink_blocked_when_disabled(
        self, tmp_path: Path, symlink_capable: bool
    ) -> None:
        """Verify symlinks are blocked when allow_symlinks=False.

        Even symlinks within config_dir should be blocked when symlinks
        are disabled for security.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        real_config = config_dir / "real.yaml"
        real_config.write_text("handler_type: db\n")

        link_config = config_dir / "link.yaml"
        link_config.symlink_to(real_config)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=False,  # Symlinks disabled
        )
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Should be blocked
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="db",
                config_ref="file:link.yaml",
            )

        assert "symlink" in str(exc_info.value).lower()

    def test_nested_symlink_chain_resolved_correctly(
        self, tmp_path: Path, symlink_capable: bool
    ) -> None:
        """Verify nested symlink chains are resolved and validated.

        Tests that chains of symlinks (link -> link -> file) are fully
        resolved and the final path is validated against config_dir.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create real file
        real_config = config_dir / "real.yaml"
        real_config.write_text("handler_type: db\n")

        # Create chain: link2 -> link1 -> real.yaml
        link1 = config_dir / "link1.yaml"
        link1.symlink_to(real_config)

        link2 = config_dir / "link2.yaml"
        link2.symlink_to(link1)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,
        )
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Should succeed since chain stays within config_dir
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:link2.yaml",
        )

        assert result.handler_type == "db"


class TestPathTraversalEdgeCases:
    """Edge cases and boundary conditions for path traversal protection."""

    def test_url_encoded_traversal_blocked(self, tmp_path: Path) -> None:
        """Verify URL-encoded traversal sequences are blocked.

        Tests that %2e%2e%2f (URL-encoded ../) doesn't bypass protection.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # URL-encoded traversal attempt
        with pytest.raises(ProtocolConfigurationError):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:%2e%2e%2fsecrets%2fcreds.yaml",
            )

    def test_unicode_normalization_traversal_blocked(self, tmp_path: Path) -> None:
        """Verify Unicode normalization doesn't bypass traversal protection.

        Some Unicode sequences can normalize to '..' - these should be blocked.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Various Unicode tricks that might normalize to traversal
        unicode_tricks = [
            "file:．．/secrets/creds.yaml",  # noqa: RUF001 - Fullwidth full stop (intentional)
            "file:‥/secrets/creds.yaml",  # Two dot leader
        ]

        for trick in unicode_tricks:
            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="evil",
                    config_ref=trick,
                )

    def test_null_byte_injection_blocked(self, tmp_path: Path) -> None:
        """Verify null byte injection is blocked.

        Null bytes can sometimes truncate paths in C libraries. Python's
        pathlib raises ValueError when encountering null bytes, which
        effectively blocks this attack vector.

        Note: This raises ValueError from pathlib rather than a custom
        ProtocolConfigurationError. This is acceptable since the attack
        is still blocked, but future versions might want to catch and
        convert this to a more informative error.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Null byte injection attempt - Python's pathlib blocks this with ValueError
        with pytest.raises((ProtocolConfigurationError, ValueError)):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:legit.yaml\x00../secrets/creds.yaml",
            )

    def test_backslash_traversal_on_windows_style_paths(self, tmp_path: Path) -> None:
        """Verify backslash traversal is handled correctly.

        On Windows, both / and \\ are path separators.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Backslash traversal attempt
        with pytest.raises(ProtocolConfigurationError):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:..\\secrets\\creds.yaml",
            )

    def test_correlation_id_in_traversal_error(self, tmp_path: Path) -> None:
        """Verify correlation_id is included in path traversal errors.

        For observability, all errors should include correlation IDs.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        test_correlation_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../secret.yaml",
                correlation_id=test_correlation_id,
            )

        # Verify correlation_id is in error context
        assert exc_info.value.model.correlation_id == test_correlation_id

    def test_case_sensitive_traversal_handling(self, tmp_path: Path) -> None:
        """Verify case variations of traversal sequences are handled.

        Tests that case-insensitive filesystems don't create bypass opportunities.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create a file that might be accessible via case tricks
        weird_dir = tmp_path / "CONFIGS"
        if not weird_dir.exists():  # May already exist on case-insensitive FS
            weird_dir.mkdir()
        weird_file = weird_dir / "secret.yaml"
        weird_file.write_text("handler_type: secret\n")

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = create_mock_container(config)
        resolver = BindingConfigResolver(container)

        # Attempt to access via case variation
        with pytest.raises(ProtocolConfigurationError):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../CONFIGS/secret.yaml",
            )


__all__: list[str] = [
    "TestPathTraversalIntegration",
    "TestSymlinkSecurityIntegration",
    "TestPathTraversalEdgeCases",
]
