# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security tests for Docker infrastructure.

These tests verify that Docker configuration follows security best practices:
- No hardcoded credentials in configuration files
- Non-root user execution in containers
- Proper secret handling without defaults
- No sensitive information exposure
- No hardcoded private IP addresses in non-comment lines

This test suite addresses PR #32 reviewer feedback requesting security-focused
tests for the Docker infrastructure implementation.
"""

from __future__ import annotations

import re

import pytest

# Import shared constants from conftest for backward compatibility.
# New tests should prefer using the docker_dir or compose_file_path fixtures instead.
from tests.unit.docker.conftest import COMPOSE_FILE_PATH, DOCKER_DIR

# Explicit marker for documentation (also auto-applied by tests/unit/conftest.py)
pytestmark = [pytest.mark.unit]


# =============================================================================
# Private IP Address Detection
# =============================================================================
#
# RFC 1918 private address ranges:
#   - 10.0.0.0/8       (10.0.0.0 - 10.255.255.255)
#   - 172.16.0.0/12    (172.16.0.0 - 172.31.255.255)
#   - 192.168.0.0/16   (192.168.0.0 - 192.168.255.255)
#
# Other private/special ranges:
#   - 127.0.0.0/8      (127.0.0.0 - 127.255.255.255) - localhost/loopback
#   - 169.254.0.0/16   (169.254.0.0 - 169.254.255.255) - link-local
#
# These patterns detect hardcoded private IPs in configuration files which
# can cause portability issues and indicate configuration that should use
# environment variables or Docker service names instead.
#
# Pattern explanation:
# - ^\s*[^#\n]* : Line start, optional whitespace, then non-comment chars
#   (ensures we don't match IPs in comment lines)
# - IP patterns match the specific private ranges
#
# Note: 172.16-31.x.x requires checking the second octet is 16-31


def build_private_ip_pattern() -> re.Pattern[str]:
    """Build regex pattern to detect private IP addresses in non-comment lines.

    This function builds a comprehensive regex pattern that matches all
    RFC 1918 private IP address ranges plus localhost and link-local ranges.

    Returns:
        Compiled regex pattern for detecting private IPs.

    Private ranges detected:
        - 10.0.0.0/8 (Class A private)
        - 172.16.0.0/12 (Class B private, 172.16.x.x - 172.31.x.x)
        - 192.168.0.0/16 (Class C private)
        - 127.0.0.0/8 (localhost/loopback)
        - 169.254.0.0/16 (link-local/APIPA)

    Note:
        The pattern only matches IPs on non-comment lines. Lines starting
        with '#' (after optional whitespace) are excluded to allow
        documentation examples and commented-out configuration.
    """
    # Individual IP range patterns (without the non-comment prefix)
    ip_10_pattern = r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # 10.x.x.x
    ip_127_pattern = r"127\.\d{1,3}\.\d{1,3}\.\d{1,3}"  # 127.x.x.x
    ip_169_254_pattern = r"169\.254\.\d{1,3}\.\d{1,3}"  # 169.254.x.x
    # 172.16.x.x to 172.31.x.x - need to match 16-19, 20-29, 30-31
    ip_172_pattern = r"172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}"
    ip_192_168_pattern = r"192\.168\.\d{1,3}\.\d{1,3}"  # 192.168.x.x

    # Combine all patterns with OR
    all_ip_patterns = "|".join(
        [
            ip_10_pattern,
            ip_127_pattern,
            ip_169_254_pattern,
            ip_172_pattern,
            ip_192_168_pattern,
        ]
    )

    # Full pattern: non-comment line containing any private IP
    # ^\s* - start of line, optional whitespace
    # [^#\n]* - any chars except # and newline (ensures not in comment)
    # (?:...) - non-capturing group for the IP patterns
    full_pattern = rf"^\s*[^#\n]*(?:{all_ip_patterns})"

    return re.compile(full_pattern, re.MULTILINE)


# Compiled pattern for use in tests
PRIVATE_IP_PATTERN = build_private_ip_pattern()


@pytest.mark.unit
class TestEnvExampleSecurity:
    """Security tests for .env.example file.

    Verifies that the example environment file:
    - Uses placeholder patterns instead of production-ready credentials
    - Contains security warnings for sensitive values
    - Does not include weak default passwords
    """

    def test_file_exists(self) -> None:
        """Verify .env.example file exists."""
        env_file = DOCKER_DIR / ".env.example"
        assert env_file.exists(), "Missing .env.example file"

    def test_no_production_ready_credentials(self) -> None:
        """Verify .env.example uses placeholder patterns, not production credentials.

        The example file should guide users to set their own values rather than
        providing working defaults that could be accidentally used in production.
        """
        env_file = DOCKER_DIR / ".env.example"
        content = env_file.read_text()

        # Check for weak password patterns as actual values (not in comments)
        # Pattern: VAR=weak_password (where weak_password is a known weak pattern)
        weak_password_assignments = [
            r"PASSWORD\s*=\s*password123",
            r"PASSWORD\s*=\s*admin123",
            r"PASSWORD\s*=\s*secret123",
            r"PASSWORD\s*=\s*test123",
            r"TOKEN\s*=\s*test_?token",
        ]

        for pattern in weak_password_assignments:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Found weak password assignment pattern: {matches}"

    def test_security_warnings_present(self) -> None:
        """Verify security warnings are present for sensitive configuration.

        The .env.example file should include clear warnings about:
        - Changing default values in production
        - Not committing real credentials
        - Proper secret handling
        """
        env_file = DOCKER_DIR / ".env.example"
        content = env_file.read_text()

        # Should have security-related keywords in comments
        security_keywords = ["SECURITY", "WARNING", "REQUIRED"]
        found_keywords = [
            keyword for keyword in security_keywords if keyword in content
        ]

        assert len(found_keywords) >= 2, (
            f"Missing security warnings (found: {found_keywords})"
        )

    def test_placeholder_patterns_for_secrets(self) -> None:
        """Verify placeholder patterns are used for GitHub tokens and sensitive values.

        GitHub tokens and other secrets should use obvious placeholder patterns
        that prevent accidental use.
        """
        env_file = DOCKER_DIR / ".env.example"
        content = env_file.read_text()

        # GitHub token line should exist and be commented or use placeholder
        github_token_lines = [
            line for line in content.split("\n") if "GITHUB_TOKEN" in line
        ]

        assert len(github_token_lines) > 0, "Missing GITHUB_TOKEN configuration"

        # At least one line should be commented out or use placeholder
        has_safe_pattern = any(
            line.strip().startswith("#") or "xxxx" in line.lower()
            for line in github_token_lines
        )

        assert has_safe_pattern, (
            "GITHUB_TOKEN should be commented or use placeholder pattern"
        )

    def test_passwords_require_explicit_values(self) -> None:
        """Verify password variables are clearly marked as requiring explicit values.

        Password and token variables should have comments indicating they must be
        set explicitly by the user.
        """
        env_file = DOCKER_DIR / ".env.example"
        content = env_file.read_text()

        # Find password-related variables
        password_vars = [
            "POSTGRES_PASSWORD",
            "REDIS_PASSWORD",
        ]

        for var in password_vars:
            # Should appear in the file
            assert var in content, f"Missing {var} configuration"

            # Find lines before the variable (up to 10 lines before)
            lines = content.split("\n")
            var_line_idx = None
            for idx, line in enumerate(lines):
                if var in line and not line.strip().startswith("#"):
                    var_line_idx = idx
                    break

            assert var_line_idx is not None, f"Could not find {var} assignment"

            # Check 10 lines before the variable for security warnings
            start_idx = max(0, var_line_idx - 10)
            section_lines = lines[start_idx : var_line_idx + 1]
            section_text = "\n".join(section_lines)

            # Should have warning comments nearby
            has_warning = (
                "WARNING" in section_text
                or "SECURITY" in section_text
                or "REQUIRED" in section_text
                or "CRITICAL" in section_text
            )

            assert has_warning, (
                f"{var} should have security warning in comments (checked {len(section_lines)} lines)"
            )


@pytest.mark.unit
class TestDockerfileSecurity:
    """Security tests for Dockerfile configuration.

    Verifies that the Dockerfile:
    - Creates and uses a non-root user
    - Does not contain hardcoded secrets
    - Follows security best practices
    """

    def test_file_exists(self) -> None:
        """Verify Dockerfile.runtime exists."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        assert dockerfile.exists(), "Missing Dockerfile.runtime"

    def test_creates_non_root_user(self) -> None:
        """Verify Dockerfile creates a non-root user.

        Containers should not run as root. The Dockerfile must create a
        dedicated user account for running the application.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should create a user with useradd or adduser
        user_creation_patterns = [
            r"useradd\s+",
            r"adduser\s+",
        ]

        found_user_creation = any(
            re.search(pattern, content) for pattern in user_creation_patterns
        )

        assert found_user_creation, (
            "Dockerfile must create a non-root user with useradd/adduser"
        )

    def test_switches_to_non_root_user(self) -> None:
        """Verify Dockerfile switches to non-root user before running application.

        After creating a non-root user, the Dockerfile must use the USER
        directive to switch to that user.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should have USER directive for non-root user
        user_directive_pattern = r"^USER\s+(?!root\b)\w+"
        found_user_directive = re.search(user_directive_pattern, content, re.MULTILINE)

        assert found_user_directive, (
            "Dockerfile must switch to non-root user with USER directive"
        )

    def test_does_not_run_as_root(self) -> None:
        """Verify Dockerfile does not explicitly run as root user.

        The final USER directive should not be 'USER root'.

        SECURITY: This test MUST fail if no USER directive exists, because
        Docker containers run as root by default when no USER is specified.
        A missing USER directive is a security vulnerability.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Split into lines and find last USER directive
        lines = content.split("\n")
        user_lines = [line for line in lines if re.match(r"^\s*USER\s+", line)]

        # SECURITY CHECK: Ensure at least one USER directive exists.
        # Without a USER directive, Docker runs containers as root by default,
        # which is a serious security vulnerability. This test MUST fail if
        # no USER directive is found - it should never silently pass.
        assert user_lines, (
            "SECURITY FAILURE: Dockerfile has no USER directive. "
            "Containers will run as root by default, which is a security risk. "
            "Add 'USER <non-root-user>' directive to the Dockerfile."
        )

        last_user_directive = user_lines[-1]
        assert "root" not in last_user_directive.lower(), (
            f"SECURITY FAILURE: Final USER directive sets root user: "
            f"'{last_user_directive.strip()}'. Container must not run as root."
        )

    def test_no_hardcoded_passwords(self) -> None:
        """Verify Dockerfile does not contain hardcoded passwords.

        The Dockerfile should not have any PASSWORD, TOKEN, or SECRET
        environment variables set to literal values.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should not contain password assignments with literal values
        secret_patterns = [
            r'PASSWORD\s*=\s*[\'"][^\'"]{3,}[\'"]',
            r'TOKEN\s*=\s*[\'"][^\'"]{3,}[\'"]',
            r'SECRET\s*=\s*[\'"][^\'"]{3,}[\'"]',
        ]

        for pattern in secret_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Found potential hardcoded secret: {matches}"

    def test_no_hardcoded_api_keys(self) -> None:
        """Verify Dockerfile does not contain hardcoded API keys.

        Common API key patterns should not appear in the Dockerfile.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Common API key patterns
        api_key_patterns = [
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub personal access token
            r"sk_live_[a-zA-Z0-9]{24,}",  # Stripe live key
            r"AKIA[0-9A-Z]{16}",  # AWS access key
        ]

        for pattern in api_key_patterns:
            matches = re.findall(pattern, content)
            assert not matches, f"Found potential hardcoded API key: {matches}"

    def test_uses_build_args_for_versions(self) -> None:
        """Verify Dockerfile uses ARG for version configuration.

        Version information should be parameterized using ARG directives
        rather than hardcoded.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should use ARG for configurable values
        assert "ARG " in content, "Dockerfile should use ARG for parameters"

    def test_has_security_documentation(self) -> None:
        """Verify Dockerfile includes security-related documentation.

        The Dockerfile should have comments explaining security practices
        like non-root user execution.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should mention security in comments
        security_keywords = ["security", "non-root", "least privilege"]
        found_security_docs = any(
            keyword in content.lower() for keyword in security_keywords
        )

        assert found_security_docs, "Dockerfile should document security practices"


@pytest.mark.unit
class TestDockerComposeSecurity:
    """Security tests for docker-compose.yml configuration.

    Verifies that the docker-compose file:
    - Requires explicit secrets without defaults
    - Uses environment variables properly
    - Does not expose unnecessary ports
    """

    def test_file_exists(self) -> None:
        """Verify docker-compose.infra.yml exists."""
        assert COMPOSE_FILE_PATH.exists(), "Missing docker-compose.infra.yml"

    def test_postgres_password_requires_explicit_value(self) -> None:
        """Verify POSTGRES_PASSWORD requires explicit value without default.

        The compose file should use ${POSTGRES_PASSWORD} without a default
        value (:-default syntax), forcing users to set it explicitly.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have POSTGRES_PASSWORD
        assert "POSTGRES_PASSWORD" in content, "Missing POSTGRES_PASSWORD configuration"

        # Should use ${POSTGRES_PASSWORD} pattern (explicit, no default)
        assert "${POSTGRES_PASSWORD}" in content, (
            "POSTGRES_PASSWORD should use ${VAR} syntax"
        )

        # Should NOT have default value pattern (${VAR:-default})
        assert "POSTGRES_PASSWORD:-" not in content, (
            "POSTGRES_PASSWORD should NOT have default value (security risk)"
        )

    def test_valkey_password_configuration(self) -> None:
        """Verify Valkey password (REDIS_PASSWORD env var) is properly configured.

        For local development, Valkey runs WITHOUT authentication (no --requirepass),
        accessible only within the Docker network. The REDIS_PASSWORD env var is
        provided for runtime services to use when connecting, with an empty default
        since the server doesn't require auth.

        Note: The env var name REDIS_PASSWORD is retained for backwards compatibility
        with existing Redis client libraries and infrastructure.

        Security rationale:
        - Valkey is only accessible within omnibase-infra-network (not exposed to host)
        - Local development does not require auth complexity
        - Production deployments should override via environment variables
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have REDIS_PASSWORD configuration for client applications
        assert "REDIS_PASSWORD" in content, "Missing REDIS_PASSWORD configuration"

        # Should use ${REDIS_PASSWORD} pattern with environment variable substitution
        assert "${REDIS_PASSWORD" in content, "REDIS_PASSWORD should use ${VAR} syntax"

        # Verify Valkey service exists and uses secure network isolation
        assert "valkey:" in content, "Missing Valkey service definition"
        assert "omnibase-infra-network" in content, (
            "Valkey should be isolated to internal network"
        )

    def test_no_hardcoded_credentials(self) -> None:
        """Verify docker-compose does not contain hardcoded credentials.

        All sensitive values should be parameterized through environment
        variables, not hardcoded in the compose file.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should not have literal password assignments
        hardcoded_patterns = [
            r"password:\s*['\"]?changeme['\"]?",
            r"password:\s*['\"]?admin['\"]?",
            r"password:\s*['\"]?secret['\"]?",
            r"token:\s*['\"]?[a-zA-Z0-9]{20,}['\"]?",
        ]

        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Found potential hardcoded credential: {matches}"

    def test_uses_secrets_configuration(self) -> None:
        """Verify docker-compose uses secrets configuration for build secrets.

        Docker Compose should define secrets for build-time credential handling.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have secrets section
        assert "secrets:" in content, "Missing secrets configuration section"

        # Should reference github_token secret for private repo access
        assert "github_token" in content, "Missing github_token secret configuration"

    def test_security_comments_present(self) -> None:
        """Verify docker-compose includes security-related comments.

        The compose file should document security requirements for passwords
        and tokens, explaining why they require explicit values.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have security-related comments
        security_keywords = ["SECURITY", "required", "explicit"]
        found_security_comments = sum(
            1 for keyword in security_keywords if keyword in content
        )

        assert found_security_comments >= 2, (
            "Missing security documentation in comments"
        )

    def test_port_exposure_is_controlled(self) -> None:
        """Verify exposed ports are intentional and documented.

        Only necessary ports should be exposed, and they should be
        configurable through environment variables.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Find all port mappings
        port_pattern = r"ports:\s*\n\s*-\s*['\"]?(\d+):"
        exposed_ports = re.findall(port_pattern, content)

        # Should expose ports through environment variables, not hardcoded
        for port in exposed_ports:
            # Check if port appears in variable substitution pattern
            env_var_pattern = rf"\$\{{[^}}]*{port}[^}}]*\}}"
            has_env_var = re.search(env_var_pattern, content)

            # Port should either be configurable or be a standard runtime port
            assert has_env_var or port == "8085", (
                f"Port {port} should be configurable via environment variable"
            )

    def test_resource_limits_defined(self) -> None:
        """Verify resource limits are defined for security and stability.

        Services should have resource limits to prevent resource exhaustion
        attacks and ensure stable operation.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have deploy resource limits
        assert "deploy:" in content, "Missing deploy configuration"
        assert "resources:" in content, "Missing resource limits"
        assert "limits:" in content, "Missing resource limits for container security"

    def test_restart_policy_is_safe(self) -> None:
        """Verify restart policy prevents infinite restart loops.

        Services should use 'unless-stopped' or 'on-failure' restart policies,
        not 'always', to prevent resource exhaustion from failing containers.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Find all restart directives
        # Note: The pattern handles both quoted and unquoted values because YAML
        # requires "no" to be quoted (otherwise it's interpreted as boolean false)
        restart_pattern = r'restart:\s*["\']?([^"\'\s]+)["\']?'
        restart_policies = re.findall(restart_pattern, content)

        # Should use safe restart policies
        safe_policies = ["unless-stopped", "on-failure", "no"]

        for policy in restart_policies:
            assert policy in safe_policies, (
                f"Unsafe restart policy: {policy} (use unless-stopped or on-failure)"
            )


@pytest.mark.unit
class TestDockerSecretsManagement:
    """Security tests for Docker secrets management.

    Verifies that:
    - BuildKit secrets are used for build-time credentials
    - No secrets are baked into images
    - Secret handling follows best practices
    """

    def test_dockerfile_uses_secret_mounts(self) -> None:
        """Verify Dockerfile uses BuildKit secret mounts for credentials.

        GitHub token should be provided via secret mount, not build args,
        to prevent secrets from being baked into image layers.
        """
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should use secret mount syntax
        assert "mount=type=secret" in content, (
            "Dockerfile should use BuildKit secret mounts"
        )

        # Should reference github_token secret
        assert "github_token" in content, "Missing github_token secret reference"

    def test_compose_defines_build_secrets(self) -> None:
        """Verify docker-compose defines secrets for build process.

        The compose file should map environment variables to build secrets.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have build secrets configuration
        assert "secrets:" in content, "Missing secrets section in docker-compose"

        # Should map GITHUB_TOKEN from environment
        assert 'environment: "GITHUB_TOKEN"' in content, (
            "Missing GITHUB_TOKEN environment mapping"
        )

    def test_no_secrets_in_build_args(self) -> None:
        """Verify secrets are not passed via build args.

        Build args are baked into image metadata and should not contain secrets.
        Check that GITHUB_TOKEN is not passed as a build arg.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should NOT have GITHUB_TOKEN as build arg
        build_args_pattern = r"args:\s*\n(?:\s*-?\s*\w+:.*\n)*"
        build_args_sections = re.findall(build_args_pattern, content)

        for section in build_args_sections:
            assert "GITHUB_TOKEN" not in section, (
                "GITHUB_TOKEN should use secret mount, not build arg"
            )


@pytest.mark.unit
class TestDockerNetworkSecurity:
    """Security tests for Docker networking configuration.

    Verifies that:
    - Services use isolated networks
    - External service access is controlled
    - Network configuration follows security best practices
    """

    def test_uses_custom_network(self) -> None:
        """Verify docker-compose defines and uses custom networks.

        Services should use isolated networks rather than the default bridge.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have networks section
        assert "networks:" in content, "Missing networks configuration"

        # Should define custom network
        assert "omnibase-infra-network" in content, "Missing custom network definition"

    def test_self_contained_infrastructure(self) -> None:
        """Verify infrastructure compose is self-contained without external host dependencies.

        The consolidated docker-compose.infra.yml provides all services locally,
        using internal Docker service names for communication. This is more secure
        than depending on external hosts because:
        - No hardcoded external IP addresses
        - Services communicate via isolated Docker network
        - External access only through configurable published ports

        Detects all RFC 1918 private IP ranges:
        - 10.0.0.0/8 (Class A private)
        - 172.16.0.0/12 (Class B private, 172.16.x.x - 172.31.x.x)
        - 192.168.0.0/16 (Class C private)
        Plus:
        - 127.0.0.0/8 (localhost/loopback)
        - 169.254.0.0/16 (link-local/APIPA)
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should NOT have hardcoded private IP addresses in non-comment lines
        # Uses PRIVATE_IP_PATTERN which covers all RFC 1918 ranges plus
        # localhost and link-local. See module-level docstring for details.
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert not matches, (
            f"Found hardcoded private IP addresses: {matches}\n"
            "Configuration should use Docker service names or environment variables "
            "instead of hardcoded IPs for portability."
        )

        # Services should use internal Docker service names, not external hosts
        # These patterns indicate proper internal service communication
        internal_service_patterns = [
            "postgres:5432",  # Internal PostgreSQL
            "redpanda:9092",  # Internal Redpanda/Kafka
            "valkey:6379",  # Internal Valkey/Redis
            "consul:8500",  # Internal Consul (optional profile)
        ]

        # Verify at least some internal service references exist
        internal_refs_found = sum(
            1 for pattern in internal_service_patterns if pattern in content
        )
        assert internal_refs_found >= 2, (
            f"Expected internal service name references (e.g., postgres:5432), "
            f"found {internal_refs_found}"
        )

    def test_network_isolation_per_service(self) -> None:
        """Verify services specify network membership explicitly.

        Each service should explicitly declare which networks it connects to.
        """
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have network references in services
        # Look for services section and verify network usage
        assert "omnibase-infra-network" in content, (
            "Services should reference custom network"
        )


# ============================================================================
# Integration Tests (require Docker to be running)
# ============================================================================


@pytest.mark.integration
class TestDockerSecurityIntegration:
    """Integration tests for Docker security (require Docker daemon).

    These tests actually build and inspect Docker images to verify security
    properties. Marked with @pytest.mark.integration to run separately.
    """

    def test_image_runs_as_non_root_user(self) -> None:
        """Verify built image actually runs as non-root user.

        This integration test builds the Docker image and inspects it to
        confirm the USER directive was correctly applied.
        """
        pytest.skip("Integration test - requires Docker daemon and build process")

    def test_image_contains_no_secrets(self) -> None:
        """Verify built image layers do not contain secrets.

        This integration test scans image layers for potential secrets.
        """
        pytest.skip(
            "Integration test - requires Docker daemon and secret scanning tools"
        )

    def test_healthcheck_endpoint_is_secure(self) -> None:
        """Verify healthcheck endpoint does not expose sensitive information.

        This integration test starts a container and verifies the health
        endpoint response does not leak internal details.
        """
        pytest.skip("Integration test - requires running container and HTTP client")


# =============================================================================
# Private IP Pattern Unit Tests
# =============================================================================


@pytest.mark.unit
class TestPrivateIpPatternDetection:
    """Unit tests for the private IP address detection pattern.

    These tests verify that PRIVATE_IP_PATTERN correctly identifies all
    RFC 1918 private IP ranges plus localhost and link-local addresses,
    while correctly ignoring IPs in comments.
    """

    def test_detects_10_range(self) -> None:
        """Test detection of 10.x.x.x addresses (Class A private)."""
        content = "server_address=10.0.0.1"
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 1, "Should detect 10.x.x.x address"

    def test_detects_10_range_variations(self) -> None:
        """Test detection of various 10.x.x.x addresses."""
        test_cases = [
            "host: 10.0.0.1",
            "addr=10.255.255.255",
            "ip: 10.100.50.25",
        ]
        for content in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            assert len(matches) == 1, f"Should detect 10.x.x.x in: {content}"

    def test_detects_127_range(self) -> None:
        """Test detection of 127.x.x.x addresses (localhost)."""
        content = "LOCALHOST=127.0.0.1"
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 1, "Should detect localhost address"

    def test_detects_127_range_variations(self) -> None:
        """Test detection of various 127.x.x.x addresses."""
        test_cases = [
            "host: 127.0.0.1",
            "addr=127.0.0.2",
            "ip: 127.255.255.255",
        ]
        for content in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            assert len(matches) == 1, f"Should detect 127.x.x.x in: {content}"

    def test_detects_169_254_range(self) -> None:
        """Test detection of 169.254.x.x addresses (link-local)."""
        content = "link_local=169.254.1.1"
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 1, "Should detect link-local address"

    def test_detects_172_16_to_31_range(self) -> None:
        """Test detection of 172.16.x.x - 172.31.x.x addresses (Class B private)."""
        # Test boundary values
        test_cases = [
            ("host: 172.16.0.1", True, "172.16 lower bound"),
            ("host: 172.31.255.255", True, "172.31 upper bound"),
            ("host: 172.20.100.50", True, "172.20 middle range"),
            ("host: 172.15.0.1", False, "172.15 below range"),
            ("host: 172.32.0.1", False, "172.32 above range"),
        ]
        for content, should_match, desc in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            if should_match:
                assert len(matches) == 1, f"Should detect {desc}"
            else:
                assert len(matches) == 0, f"Should NOT detect {desc}"

    def test_detects_192_168_range(self) -> None:
        """Test detection of 192.168.x.x addresses (Class C private)."""
        content = "server: 192.168.1.100"
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 1, "Should detect 192.168 address"

    def test_detects_192_168_range_variations(self) -> None:
        """Test detection of various 192.168.x.x addresses."""
        test_cases = [
            "host: 192.168.0.1",
            "addr=192.168.255.255",
            "ip: 192.168.86.200",
        ]
        for content in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            assert len(matches) == 1, f"Should detect 192.168.x.x in: {content}"

    def test_ignores_commented_lines(self) -> None:
        """Test that IPs in comment lines are ignored."""
        test_cases = [
            "# server: 192.168.1.1",
            "  # host=10.0.0.1",
            "    # Example: 172.16.0.1",
            "# 127.0.0.1 is localhost",
        ]
        for content in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            assert len(matches) == 0, f"Should ignore comment: {content}"

    def test_ignores_public_ips(self) -> None:
        """Test that public IP addresses are not matched."""
        test_cases = [
            "server: 8.8.8.8",  # Google DNS
            "addr=1.1.1.1",  # Cloudflare DNS
            "ip: 104.16.0.1",  # Cloudflare
            "host: 172.15.0.1",  # Below 172.16 range
            "host: 172.32.0.1",  # Above 172.31 range
            "host: 192.167.1.1",  # Not 192.168
            "host: 192.169.1.1",  # Not 192.168
        ]
        for content in test_cases:
            matches = PRIVATE_IP_PATTERN.findall(content)
            assert len(matches) == 0, f"Should not match public IP in: {content}"

    def test_multiline_content(self) -> None:
        """Test pattern works correctly with multiline content."""
        content = """
# This is a comment with 192.168.1.1
server_addr=10.0.0.1
# Another comment 172.16.0.1
redis_host=192.168.86.100
"""
        matches = PRIVATE_IP_PATTERN.findall(content)
        # Should find 10.0.0.1 and 192.168.86.100, but not the commented ones
        assert len(matches) == 2, "Should find exactly 2 non-commented IPs"

    def test_inline_comments_after_ip(self) -> None:
        """Test that IPs before inline comments are detected."""
        content = "server: 192.168.1.1  # internal server"
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 1, "Should detect IP before inline comment"

    def test_mixed_valid_and_invalid(self) -> None:
        """Test content with mix of valid and commented IPs."""
        content = """
# Configuration
# Use 192.168.86.200 for remote server
database_host: postgres
kafka_bootstrap: redpanda:9092
# Do not hardcode: 10.0.0.1
"""
        matches = PRIVATE_IP_PATTERN.findall(content)
        assert len(matches) == 0, "All IPs are in comments, should find none"
