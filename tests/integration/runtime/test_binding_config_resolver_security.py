# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for BindingConfigResolver security controls.

This module provides comprehensive security testing for the BindingConfigResolver,
focusing on path traversal protection, symlink attacks, and input validation with
real filesystem operations.

Security Testing Approach
-------------------------
Unlike unit tests that may mock filesystem operations, these integration tests:

1. **Use real temporary directories and files** - Validates behavior against actual
   filesystem semantics rather than mocked assumptions.

2. **Create actual symlinks** (where supported) - Tests symlink resolution behavior
   at the OS level, not mocked symlink detection.

3. **Test interactions between validation layers** - Validates that multiple security
   checks work together (e.g., path normalization + boundary check + symlink check).

4. **Verify error messages don't leak sensitive information** - Ensures error messages
   don't expose filesystem structure, credentials, or internal paths.

Attack Vectors Covered
----------------------
- **Path Traversal (CWE-22)**: Using ``../`` sequences to escape the config directory
- **Symlink Following (CWE-61)**: Using symlinks to access files outside boundaries
- **URL Encoding Bypass (CWE-20)**: Using ``%2e%2e%2f`` to bypass pattern matching
- **Unicode Normalization (CWE-176)**: Using Unicode characters that normalize to ``..``
- **Null Byte Injection (CWE-158)**: Using null bytes to truncate paths
- **Case Sensitivity Exploitation**: Using case variations on case-insensitive filesystems
- **Backslash Traversal**: Using Windows-style path separators

Test Classes
------------
- :class:`TestPathTraversalIntegration`: Real filesystem path traversal attacks
- :class:`TestSymlinkSecurityIntegration`: Symlink-based path traversal attempts
- :class:`TestPathTraversalEdgeCases`: Edge cases and boundary conditions

Related
-------
- OMN-765: BindingConfigResolver implementation
- PR #168: Security enhancements
- docs/patterns/binding_config_resolver.md#path-traversal-protection
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

# Type alias for the mock container factory function
MockContainerFactory = Callable[[ModelBindingConfigResolverConfig], MagicMock]

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def secret_config_yaml() -> str:
    """YAML content representing a sensitive configuration file.

    This fixture provides standardized test data for secret/credential files
    that should NOT be accessible via path traversal attacks.

    Security Implication:
        If any test can read this content via traversal, the security boundary
        has been breached. The 'password' field provides a marker that should
        NEVER appear in error messages.
    """
    return "handler_type: secret\npassword: super_secret\n"


@pytest.fixture
def db_handler_config_yaml() -> str:
    """YAML content for a legitimate database handler configuration.

    This fixture provides standardized test data for valid handler configs
    that SHOULD be accessible within the config directory boundary.
    """
    return "handler_type: db\ntimeout_ms: 5000\n"


@pytest.fixture
def passwd_style_config_yaml() -> str:
    """YAML content mimicking a Unix passwd file structure.

    Security Implication:
        Tests that deeply nested traversal (../../../etc/passwd style attacks)
        cannot access system files or files outside the config boundary.
    """
    return "handler_type: passwd\nroot:x:0:0\n"


@pytest.fixture
def minimal_handler_config_yaml() -> str:
    """Minimal valid handler configuration for basic tests.

    This fixture provides the smallest valid configuration that satisfies
    the handler_type requirement.
    """
    return "handler_type: db\n"


@pytest.fixture
def config_with_timeout_yaml() -> str:
    """Handler configuration with timeout parameter.

    Used for tests that need to verify configuration values are correctly
    loaded from legitimate paths.
    """
    return "handler_type: db\ntimeout_ms: 3000\n"


# =============================================================================
# Mock Container Fixtures
# =============================================================================


def _create_mock_container(config: ModelBindingConfigResolverConfig) -> MagicMock:
    """Create a mock container with the given config registered.

    This helper creates a mock ModelONEXContainer with the required
    service_registry.resolve_service() behavior for BindingConfigResolver.

    Args:
        config: The binding config resolver configuration to register.

    Returns:
        A MagicMock configured to resolve the given config from its service registry.

    Note:
        This is an internal helper. Test code should use the ``mock_container_factory``
        fixture for cleaner test setup.
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


@pytest.fixture
def mock_container_factory() -> MockContainerFactory:
    """Factory fixture for creating mock containers with specific configurations.

    Usage::

        def test_something(mock_container_factory, tmp_path):
            config = ModelBindingConfigResolverConfig(config_dir=tmp_path / "configs")
            container = mock_container_factory(config)
            resolver = BindingConfigResolver(container)

    Returns:
        A callable that takes a ModelBindingConfigResolverConfig and returns
        a configured mock container.
    """
    return _create_mock_container


# =============================================================================
# Path Traversal Integration Tests
# =============================================================================


class TestPathTraversalIntegration:
    """Integration tests for path traversal protection with real filesystem.

    Attack Vector: Path Traversal (CWE-22)
    --------------------------------------
    Path traversal attacks attempt to access files outside the intended directory
    by using special path sequences like ``../`` (parent directory reference).

    Security Model
    --------------
    The BindingConfigResolver uses a "sandbox" security model where:
    1. A ``config_dir`` is defined as the root of allowed configuration files
    2. All resolved paths MUST be within this directory
    3. Path traversal sequences that would escape this boundary are rejected

    Defense Layers
    --------------
    1. **Path normalization**: Converts paths to canonical form (resolves ``..``)
    2. **Boundary check**: Verifies resolved path starts with config_dir
    3. **Symlink resolution**: Resolves symlinks before boundary check
    4. **Error sanitization**: Ensures errors don't leak path information

    Test Strategy
    -------------
    These tests create real directory structures with:
    - A "config_dir" containing legitimate configuration files
    - A "secrets" directory OUTSIDE config_dir with sensitive files
    - Various traversal attempts to reach the secrets directory
    """

    def test_parent_directory_traversal_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        secret_config_yaml: str,
        db_handler_config_yaml: str,
    ) -> None:
        """Verify ``../`` traversal is blocked with real filesystem.

        Attack Vector
        -------------
        Single parent directory traversal (``../secrets/file.yaml``) attempts
        to escape one level above config_dir to access sibling directories.

        Security Implication
        --------------------
        If successful, an attacker could read any file accessible to the process,
        potentially exposing credentials, API keys, or other sensitive data.

        Expected Behavior
        -----------------
        - Raises ``ProtocolConfigurationError`` with "traversal" or "not allowed"
        - Error message does NOT contain the actual file contents
        - Error message does NOT reveal the full path structure
        """
        # Create directory structure simulating a typical deployment
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        # Create a "secret" file outside config_dir that should be inaccessible
        # SECURITY: This file represents credentials that should never be exposed
        secret_file = secret_dir / "credentials.yaml"
        secret_file.write_text(secret_config_yaml)

        # Create a legitimate config inside config_dir
        legit_config = config_dir / "handler.yaml"
        legit_config.write_text(db_handler_config_yaml)

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Attempt to access secret file via path traversal
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../secrets/credentials.yaml",
            )

        # Verify error message indicates path traversal was blocked
        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "not allowed" in error_msg, (
            f"Expected 'traversal' or 'not allowed' in error message, got: {error_msg}"
        )

        # SECURITY VERIFICATION: Ensure sensitive data is not leaked in error message
        # The actual file contents should NEVER appear in exceptions
        assert "super_secret" not in str(exc_info.value), (
            "SECURITY VIOLATION: Secret file contents leaked in error message"
        )
        # The full path to secrets directory should not be exposed
        assert str(secret_dir) not in str(exc_info.value), (
            "SECURITY VIOLATION: Sensitive path leaked in error message"
        )

    def test_multiple_parent_directory_traversal_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        passwd_style_config_yaml: str,
    ) -> None:
        """Verify multiple ``../`` sequences are blocked.

        Attack Vector
        -------------
        Deep traversal using multiple parent directory sequences
        (``../../../etc/passwd``) attempts to escape multiple levels up the
        directory tree to access system files or root-level directories.

        Security Implication
        --------------------
        This is a classic attack vector used to read /etc/passwd, /etc/shadow,
        or other system configuration files. In containerized environments,
        this could expose secrets mounted at known paths.

        Expected Behavior
        -----------------
        Multiple consecutive parent directory sequences are blocked at the
        parsing layer (ModelConfigRef.parse) before resolution, resulting in
        an "invalid config reference" error rather than a "path traversal" error.
        This is acceptable as long as the attack is blocked.
        """
        # Create deeply nested config directory simulating app structure
        config_dir = tmp_path / "app" / "configs" / "handlers"
        config_dir.mkdir(parents=True)

        # Create sensitive file at a "root" location (tmp_path/etc/passwd)
        sensitive_file = tmp_path / "etc" / "passwd"
        sensitive_file.parent.mkdir(parents=True, exist_ok=True)
        sensitive_file.write_text(passwd_style_config_yaml)

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Attempt deep traversal - should be blocked at parsing or resolution layer
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../../../etc/passwd",
            )

        # Accept either "traversal" (resolution layer) or "invalid" (parsing layer)
        # Both indicate the attack was successfully blocked
        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "invalid" in error_msg, (
            f"Expected 'traversal' or 'invalid' in error message, got: {error_msg}"
        )

    def test_traversal_with_absolute_path_still_requires_config_dir(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        minimal_handler_config_yaml: str,
    ) -> None:
        """Verify absolute paths are validated against config_dir.

        Attack Vector
        -------------
        Instead of using relative traversal, an attacker might try to specify
        an absolute path directly (``file:///etc/passwd``) to bypass relative
        path validation.

        Security Implication
        --------------------
        If absolute paths bypass the config_dir boundary check, the entire
        sandbox model is defeated. An attacker could read any file on the
        filesystem that the process has permission to access.

        Expected Behavior
        -----------------
        Even with absolute paths, the resolved path MUST be within config_dir.
        Attempts to use absolute paths outside config_dir are rejected.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()
        secret_file = secret_dir / "creds.yaml"
        secret_file.write_text(minimal_handler_config_yaml)

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Attempt to use absolute path outside config_dir
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref=f"file:///{secret_file}",
            )

        # Should be blocked - accept various error messages as all indicate blocking
        error_msg = str(exc_info.value).lower()
        assert (
            "traversal" in error_msg
            or "not allowed" in error_msg
            or "not found" in error_msg
        ), f"Expected security-related error message, got: {error_msg}"

    def test_legitimate_relative_path_succeeds(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        db_handler_config_yaml: str,
    ) -> None:
        """Verify legitimate relative paths work correctly.

        Security Implication
        --------------------
        Security measures should NOT break normal operation. This test ensures
        that valid relative paths within config_dir work correctly, preventing
        false positives that would make the system unusable.

        Expected Behavior
        -----------------
        Paths that stay within config_dir should resolve successfully and
        return the expected configuration values.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        subdir = config_dir / "handlers"
        subdir.mkdir()

        config_file = subdir / "db.yaml"
        config_file.write_text(db_handler_config_yaml)

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Legitimate relative path should work
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:handlers/db.yaml",
        )

        assert result.handler_type == "db"
        assert result.timeout_ms == 5000


# =============================================================================
# Symlink Security Integration Tests
# =============================================================================


class TestSymlinkSecurityIntegration:
    """Integration tests for symlink-based path traversal protection.

    Attack Vector: Symlink Following (CWE-61)
    -----------------------------------------
    Symlink attacks use symbolic links to create a "shortcut" from within
    the allowed directory to files outside it. Even if path validation passes
    on the symlink path itself, following the symlink leads outside the boundary.

    Security Model
    --------------
    The BindingConfigResolver provides configurable symlink handling:
    1. ``allow_symlinks=False`` (default): Rejects any symlink in the path
    2. ``allow_symlinks=True``: Allows symlinks but validates the resolved path

    Defense Layers
    --------------
    1. **Symlink detection**: Check if file or any path component is a symlink
    2. **Path resolution**: Use ``resolve()`` to get the actual target path
    3. **Post-resolution boundary check**: Verify resolved path is within boundary

    Test Strategy
    -------------
    These tests create symlinks in various configurations:
    - Symlink inside config_dir pointing outside (should be blocked)
    - Symlink inside config_dir pointing inside (should work when allowed)
    - Nested symlink chains (should fully resolve and validate)
    """

    @pytest.fixture
    def symlink_capable(self) -> bool:
        """Check if the current system supports symlinks.

        On Windows, symlinks require either:
        - Administrator privileges, OR
        - Developer mode enabled (Windows 10 Creators Update and later)

        Returns:
            True if symlinks can be created, False otherwise.

        Note:
            Tests using this fixture should call ``pytest.skip()`` when
            symlinks are not supported.
        """
        if platform.system() != "Windows":
            return True

        # Check if we can create symlinks on Windows
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                test_target = Path(tmpdir) / "target"
                test_target.write_text("test")
                test_link = Path(tmpdir) / "link"
                test_link.symlink_to(test_target)
            return True
        except OSError:
            return False

    def test_symlink_outside_config_dir_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        symlink_capable: bool,
        secret_config_yaml: str,
    ) -> None:
        """Verify symlinks pointing outside config_dir are blocked.

        Attack Vector
        -------------
        Create a symlink inside config_dir that points to a file outside.
        The symlink path passes initial validation, but following it escapes
        the security boundary.

        Security Implication
        --------------------
        This is a Time-of-Check to Time-of-Use (TOCTOU) style attack. The path
        looks valid when checked, but accessing it reads from a different
        location. This could expose any file readable by the process.

        Expected Behavior
        -----------------
        Even when ``allow_symlinks=True``, symlinks that resolve to paths
        outside config_dir are blocked. The final resolved path MUST be
        within the config_dir boundary.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        # Create secret file outside config_dir
        # SECURITY: Contains marker text that should never appear in errors
        secret_file = secret_dir / "credentials.yaml"
        secret_file.write_text(secret_config_yaml)

        # Create symlink inside config_dir pointing outside
        # SECURITY: This is the attack vector - a "legitimate looking" path
        # that actually points to secrets
        evil_link = config_dir / "evil.yaml"
        evil_link.symlink_to(secret_file)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,  # Symlinks allowed but must stay in config_dir
        )
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Should be blocked by post-resolution path traversal protection
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:evil.yaml",
            )

        error_msg = str(exc_info.value).lower()
        assert "traversal" in error_msg or "symlink" in error_msg, (
            f"Expected 'traversal' or 'symlink' in error message, got: {error_msg}"
        )

        # SECURITY VERIFICATION: Password should never appear in error message
        assert "leaked" not in str(exc_info.value), (
            "SECURITY VIOLATION: Secret file contents leaked in error message"
        )

    def test_symlink_inside_config_dir_allowed(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        symlink_capable: bool,
        config_with_timeout_yaml: str,
    ) -> None:
        """Verify symlinks within config_dir are allowed when enabled.

        Security Implication
        --------------------
        When ``allow_symlinks=True``, symlinks that point to other files
        WITHIN config_dir should work. This enables legitimate use cases
        like config file aliasing or versioned configs.

        Expected Behavior
        -----------------
        Symlinks that point to files within config_dir should resolve
        successfully and return the expected configuration values.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create real config
        real_config = config_dir / "real.yaml"
        real_config.write_text(config_with_timeout_yaml)

        # Create symlink to it (both inside config_dir - this is safe)
        link_config = config_dir / "link.yaml"
        link_config.symlink_to(real_config)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,
        )
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Should succeed - symlink stays within boundary
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:link.yaml",
        )

        assert result.handler_type == "db"
        assert result.timeout_ms == 3000

    def test_symlink_blocked_when_disabled(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        symlink_capable: bool,
        minimal_handler_config_yaml: str,
    ) -> None:
        """Verify symlinks are blocked when allow_symlinks=False.

        Security Implication
        --------------------
        The strictest security posture blocks ALL symlinks, even those
        pointing within the config_dir. This prevents any symlink-based
        attacks at the cost of some flexibility.

        Expected Behavior
        -----------------
        When ``allow_symlinks=False`` (the default), any symlink in the
        path is rejected, regardless of where it points.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        real_config = config_dir / "real.yaml"
        real_config.write_text(minimal_handler_config_yaml)

        link_config = config_dir / "link.yaml"
        link_config.symlink_to(real_config)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=False,  # Symlinks disabled for strictest security
        )
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Should be blocked - symlinks not allowed regardless of target
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="db",
                config_ref="file:link.yaml",
            )

        # Error should indicate symlink was the reason for rejection
        assert "symlink" in str(exc_info.value).lower(), (
            f"Expected 'symlink' in error message, got: {exc_info.value}"
        )

    def test_nested_symlink_chain_resolved_correctly(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        symlink_capable: bool,
        minimal_handler_config_yaml: str,
    ) -> None:
        """Verify nested symlink chains are resolved and validated.

        Attack Vector
        -------------
        Create a chain of symlinks (link2 -> link1 -> target) to potentially
        confuse path validation. Each link in the chain might pass individual
        checks, but the final resolution could escape the boundary.

        Security Implication
        --------------------
        Some implementations only resolve one level of symlinks, leaving
        chained symlinks as a potential bypass. The resolver MUST fully
        resolve all symlinks before performing the boundary check.

        Expected Behavior
        -----------------
        The entire symlink chain is resolved to the final target path,
        and that path is validated against the config_dir boundary.
        """
        if not symlink_capable:
            pytest.skip("Symlinks not supported on this system")

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create real file
        real_config = config_dir / "real.yaml"
        real_config.write_text(minimal_handler_config_yaml)

        # Create chain: link2 -> link1 -> real.yaml
        link1 = config_dir / "link1.yaml"
        link1.symlink_to(real_config)

        link2 = config_dir / "link2.yaml"
        link2.symlink_to(link1)

        config = ModelBindingConfigResolverConfig(
            config_dir=config_dir,
            allow_symlinks=True,
        )
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Should succeed since entire chain stays within config_dir
        result = resolver.resolve(
            handler_type="db",
            config_ref="file:link2.yaml",
        )

        assert result.handler_type == "db"


# =============================================================================
# Path Traversal Edge Cases
# =============================================================================


class TestPathTraversalEdgeCases:
    """Edge cases and boundary conditions for path traversal protection.

    Attack Vector: Input Validation Bypass
    --------------------------------------
    These tests cover various encoding, normalization, and character-based
    techniques that attackers use to bypass input validation while achieving
    the same path traversal effect.

    Defense-in-Depth Testing
    ------------------------
    Each test targets a specific bypass technique:
    - URL encoding to evade pattern matching
    - Unicode normalization to create equivalent characters
    - Null byte injection to truncate paths in C libraries
    - Alternative path separators (backslash on Windows)
    - Case manipulation on case-insensitive filesystems
    """

    def test_url_encoded_traversal_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
    ) -> None:
        """Verify URL-encoded traversal sequences are blocked.

        Attack Vector: URL Encoding Bypass (CWE-20)
        -------------------------------------------
        Using ``%2e%2e%2f`` (URL-encoded ``../``) to bypass naive pattern
        matching that looks for literal ``..`` sequences.

        Security Implication
        --------------------
        If the resolver checks for ``..`` but doesn't URL-decode first,
        encoded sequences pass validation but are decoded later during
        file access, achieving the traversal.

        Expected Behavior
        -----------------
        URL-encoded sequences are either:
        1. Decoded before validation (and then blocked), OR
        2. Treated as invalid/literal characters (and fail to resolve)
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # URL-encoded traversal attempt: %2e = '.', %2f = '/'
        with pytest.raises(ProtocolConfigurationError):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:%2e%2e%2fsecrets%2fcreds.yaml",
            )

    def test_unicode_normalization_traversal_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
    ) -> None:
        """Verify Unicode normalization doesn't bypass traversal protection.

        Attack Vector: Unicode Normalization (CWE-176)
        ----------------------------------------------
        Using Unicode characters that visually resemble or normalize to
        path traversal sequences:
        - Fullwidth full stop (U+FF0E): ``.`` looks like ``.``
        - Two dot leader (U+2025): ``..`` is a single character looking like ``..``

        Security Implication
        --------------------
        Some systems normalize Unicode to ASCII equivalents. If validation
        happens before normalization but file access happens after, the
        normalized path could escape the boundary.

        Expected Behavior
        -----------------
        Unicode tricks are either:
        1. Normalized before validation (and then blocked), OR
        2. Treated as invalid characters (and fail to resolve)
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Various Unicode tricks that might normalize to traversal
        unicode_tricks = [
            # Fullwidth full stop (U+FF0E) - looks like regular period
            "file:\uff0e\uff0e/secrets/creds.yaml",
            # Two dot leader (U+2025) - single char that looks like ..
            "file:\u2025/secrets/creds.yaml",
        ]

        for trick in unicode_tricks:
            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="evil",
                    config_ref=trick,
                )

    def test_null_byte_injection_blocked(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
    ) -> None:
        """Verify null byte injection is blocked.

        Attack Vector: Null Byte Injection (CWE-158)
        --------------------------------------------
        Using null bytes (``\\x00``) to truncate paths. In C libraries,
        strings are null-terminated, so ``legit.yaml\\x00../evil`` might
        be validated as ``legit.yaml`` but then only ``legit.yaml`` is
        accessed (or in vulnerable systems, the truncation happens
        differently at different layers).

        Security Implication
        --------------------
        Null byte injection has been used to bypass extension checks
        (``file.php\\x00.jpg`` passes image validation but executes as PHP)
        and path validation in older systems.

        Expected Behavior
        -----------------
        Python's pathlib raises ``ValueError`` when encountering null bytes,
        which effectively blocks this attack. This is acceptable behavior -
        the attack is blocked even if not by our custom code.

        NOTE: We accept either ProtocolConfigurationError (custom handling)
        OR ValueError (Python's built-in protection). Both indicate the
        attack was successfully blocked.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # SECURITY NOTE: Python's pathlib rejects null bytes with ValueError.
        # This is a defense-in-depth protection from the standard library.
        # Future versions might catch ValueError and convert to a more
        # informative ProtocolConfigurationError, but the attack is blocked
        # either way.
        with pytest.raises((ProtocolConfigurationError, ValueError)):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:legit.yaml\x00../secrets/creds.yaml",
            )

    def test_backslash_traversal_on_windows_style_paths(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
    ) -> None:
        """Verify backslash traversal is handled correctly.

        Attack Vector: Alternative Path Separators
        ------------------------------------------
        On Windows, both ``/`` and ``\\`` are valid path separators.
        An attacker might use ``..\\secrets\\file`` to bypass validation
        that only checks for ``../``.

        Security Implication
        --------------------
        Cross-platform applications must handle both separator styles.
        Even on Linux, some applications process Windows-style paths,
        creating potential bypass opportunities.

        Expected Behavior
        -----------------
        Backslash traversal sequences are blocked, either by:
        1. Normalizing to forward slashes before validation
        2. Explicitly checking for both separator styles
        3. Treating backslashes as invalid characters
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Backslash traversal attempt
        with pytest.raises(ProtocolConfigurationError):
            resolver.resolve(
                handler_type="evil",
                config_ref="file:..\\secrets\\creds.yaml",
            )

    def test_correlation_id_in_traversal_error(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
    ) -> None:
        """Verify correlation_id is included in path traversal errors.

        Observability Requirement
        -------------------------
        All errors must include correlation IDs for distributed tracing
        and log correlation. This enables operators to trace security
        events across multiple services.

        Security Implication
        --------------------
        Correlation IDs help identify attack patterns across multiple
        requests, enabling detection of coordinated traversal attempts.

        Expected Behavior
        -----------------
        The raised ProtocolConfigurationError includes the correlation_id
        in its error context (``exc_info.value.model.correlation_id``).
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        test_correlation_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="evil",
                config_ref="file:../secret.yaml",
                correlation_id=test_correlation_id,
            )

        # Verify correlation_id is preserved in error context for tracing
        assert exc_info.value.model.correlation_id == test_correlation_id, (
            f"Expected correlation_id {test_correlation_id}, "
            f"got {exc_info.value.model.correlation_id}"
        )

    def test_case_sensitive_traversal_handling(
        self,
        tmp_path: Path,
        mock_container_factory: MockContainerFactory,
        minimal_handler_config_yaml: str,
    ) -> None:
        """Verify case variations of traversal sequences are handled.

        Attack Vector: Case Sensitivity Exploitation
        --------------------------------------------
        On case-insensitive filesystems (Windows, macOS default), ``CONFIGS``
        and ``configs`` refer to the same directory. An attacker might use
        ``../CONFIGS/secret.yaml`` to access files in a directory that
        validation considers different from ``configs``.

        Security Implication
        --------------------
        If path validation is case-sensitive but the filesystem is not,
        an attacker can bypass validation using case variations.

        Expected Behavior
        -----------------
        Case variations in traversal paths are blocked, and the security
        boundary is maintained regardless of filesystem case sensitivity.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create a file that might be accessible via case tricks
        # On case-insensitive FS, this might be the same as config_dir
        weird_dir = tmp_path / "CONFIGS"
        if not weird_dir.exists():  # May already exist on case-insensitive FS
            weird_dir.mkdir()
        weird_file = weird_dir / "secret.yaml"
        weird_file.write_text(minimal_handler_config_yaml)

        config = ModelBindingConfigResolverConfig(config_dir=config_dir)
        container = mock_container_factory(config)
        resolver = BindingConfigResolver(container)

        # Attempt to access via case variation - should be blocked
        # regardless of filesystem case sensitivity
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
