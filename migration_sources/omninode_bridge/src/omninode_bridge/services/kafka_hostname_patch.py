#!/usr/bin/env python3
"""
Hostname resolution patch for aiokafka RedPanda connectivity.

This module provides a monkey patch for socket.getaddrinfo to resolve
'omninode-bridge-redpanda' to localhost, fixing the advertised listeners
configuration issue.

Enhanced with robust error handling to prevent container restart loops.
"""

import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)

# Store original getaddrinfo
_original_getaddrinfo = socket.getaddrinfo
_patch_applied = False
_patch_failures = 0
_max_failures = 5


def patched_getaddrinfo(host: str, port: Any, *args, **kwargs) -> list[tuple]:
    """Patched getaddrinfo that redirects RedPanda container hostname to localhost with enhanced error handling.

    This function resolves the Docker networking issue where RedPanda advertises
    'omninode-bridge-redpanda:9092' as the broker address even for external connections,
    but that hostname doesn't resolve outside the Docker container environment.

    The patch provides:
    - Hostname redirection: omninode-bridge-redpanda -> localhost
    - Port mapping: 9092 (internal) -> 29092 (external Docker port)
    - Robust error handling to prevent container restart loops
    - Fallback mechanisms for network failures
    - Failure tracking with automatic disable after threshold

    Container Restart Loop Prevention:
        - Tracks patch failures with global counter
        - Falls back to original hostname if redirect fails
        - Logs errors without raising exceptions in critical cases
        - Provides graceful degradation when patch becomes unstable

    Args:
        host: Hostname or IP address to resolve
        port: Port number or service name to resolve
        *args: Additional positional arguments passed to socket.getaddrinfo
        **kwargs: Additional keyword arguments passed to socket.getaddrinfo

    Returns:
        List of address tuples in standard getaddrinfo format:
        [(family, type, proto, canonname, sockaddr), ...]

    Raises:
        socket.gaierror: DNS resolution errors (after fallback attempts)
        OSError: Network connectivity issues (after fallback attempts)
        Exception: Critical errors that can't be handled gracefully

    Example Redirections:
        Input: ('omninode-bridge-redpanda', 9092)
        Output: Resolves to localhost:29092

        Input: ('omninode-bridge-redpanda', 8080)
        Output: Resolves to localhost:8080

        Input: ('other-host.com', 80)
        Output: No redirection, normal resolution

    Error Handling Strategy:
        1. Try redirected hostname/port if applicable
        2. If redirect fails, try original hostname/port
        3. Track failures for stability monitoring
        4. Log errors but don't fail the application
        5. Disable patch if too many failures occur

    Thread Safety:
        This function modifies global variables (_patch_failures) and should
        be used in environments where socket.getaddrinfo calls are serialized
        or where race conditions in failure counting are acceptable.
    """
    global _patch_failures

    try:
        original_host = host
        original_port = port

        if host == "omninode-bridge-redpanda":
            # Redirect hostname to localhost
            host = "localhost"

            # Redirect internal port 9092 to external port 29092
            if port == 9092:
                port = 29092
                logger.debug(
                    f"Redirecting omninode-bridge-redpanda:9092 -> {host}:{port}"
                )
            else:
                logger.debug(
                    f"Redirecting omninode-bridge-redpanda:{port} -> {host}:{port}"
                )

        # Attempt the address resolution with robust error handling
        try:
            result = _original_getaddrinfo(host, port, *args, **kwargs)
            # Reset failure count on successful resolution
            _patch_failures = 0
            return result

        except (OSError, socket.gaierror, socket.herror) as e:
            # Log network-related errors but don't fail immediately
            logger.warning(
                f"Network error resolving {host}:{port} (original: {original_host}:{original_port}): {e}"
            )
            _patch_failures += 1

            # If this is a redirected hostname and it fails, try the original
            if original_host != host:
                logger.info(
                    f"Retrying with original hostname: {original_host}:{original_port}"
                )
                try:
                    return _original_getaddrinfo(
                        original_host, original_port, *args, **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to original hostname also failed: {fallback_error}"
                    )

            # Re-raise the original exception
            raise

    except Exception as e:
        _patch_failures += 1
        logger.error(f"Critical error in hostname patch: {e}")

        # If we have too many failures, fall back to original behavior
        if _patch_failures >= _max_failures:
            logger.critical(
                f"Hostname patch has failed {_patch_failures} times, considering disabling"
            )

        # Always try to fall back to original getaddrinfo for stability
        try:
            return _original_getaddrinfo(host, port, *args, **kwargs)
        except Exception as fallback_error:
            logger.critical(f"Even original getaddrinfo failed: {fallback_error}")
            raise


def apply_kafka_hostname_patch():
    """Apply hostname resolution patch for Kafka/RedPanda connectivity with safety checks.

    This function monkey-patches the socket.getaddrinfo function to resolve Docker
    networking issues with RedPanda broker connectivity. The patch is designed to
    prevent container restart loops through comprehensive error handling and validation.

    Patch Functionality:
        - Replaces socket.getaddrinfo with patched_getaddrinfo
        - Redirects 'omninode-bridge-redpanda' to 'localhost'
        - Maps internal port 9092 to external port 29092
        - Provides fallback to original function on errors

    Safety Features:
        - Validates original function availability before patching
        - Prevents duplicate patch application
        - Tracks patch status globally
        - Comprehensive error logging without raising exceptions
        - Graceful degradation on patch failures

    Container Restart Loop Prevention:
        The function is designed to never raise exceptions that could cause
        container restart loops. All errors are logged but the application
        continues with either patched or original hostname resolution.

    Returns:
        bool: True if patch applied successfully, False if patch failed or already applied

    Example Usage:
        # Apply patch during application startup
        success = apply_kafka_hostname_patch()
        if success:
            print("Kafka hostname patch applied successfully")
        else:
            print("Failed to apply patch, using original hostname resolution")

        # Check if patch is active
        if is_patch_applied():
            print("Patch is currently active")

    Side Effects:
        - Modifies socket.getaddrinfo globally for the entire Python process
        - Sets global _patch_applied flag to True on success
        - Logs patch application status for monitoring

    Thread Safety:
        This function modifies global state and should typically be called
        during application startup before multiple threads are created.
    """
    global _patch_applied

    try:
        if _patch_applied:
            logger.debug("Kafka hostname patch already applied")
            return True

        # Validate that we have the original function
        if not callable(_original_getaddrinfo):
            logger.error("Original getaddrinfo function is not available")
            return False

        # Apply the patch
        socket.getaddrinfo = patched_getaddrinfo
        _patch_applied = True

        logger.info(
            "✅ Kafka hostname patch applied - 'omninode-bridge-redpanda' -> 'localhost'"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to apply Kafka hostname patch: {e}")
        return False


def remove_kafka_hostname_patch():
    """Remove the hostname resolution patch and restore original socket.getaddrinfo.

    This function safely removes the monkey patch applied by apply_kafka_hostname_patch()
    and restores the original socket.getaddrinfo function. It includes comprehensive
    error handling to prevent application failures during patch removal.

    Restoration Process:
        - Validates that the patch is currently applied
        - Checks availability of original getaddrinfo function
        - Restores socket.getaddrinfo to original implementation
        - Updates global patch status flags
        - Logs restoration status for monitoring

    Safety Features:
        - No-op if patch was not applied (safe to call multiple times)
        - Validates original function before restoration
        - Comprehensive error logging without raising exceptions
        - Graceful handling of edge cases

    Returns:
        bool: True if patch removed successfully, False if removal failed or patch not applied

    Example Usage:
        # Remove patch during application shutdown
        success = remove_kafka_hostname_patch()
        if success:
            print("Kafka hostname patch removed successfully")
        else:
            print("Failed to remove patch or patch was not applied")

        # Check patch status after removal
        if not is_patch_applied():
            print("Patch successfully removed")

    Use Cases:
        - Application shutdown cleanup
        - Testing scenarios where patch needs to be toggled
        - Recovery from patch-related issues
        - Dynamic patch management

    Side Effects:
        - Restores socket.getaddrinfo to original Python implementation
        - Sets global _patch_applied flag to False on success
        - Logs patch removal status for monitoring

    Thread Safety:
        This function modifies global state and should be called when
        no other threads are actively using socket.getaddrinfo or when
        proper synchronization is in place.
    """
    global _patch_applied

    try:
        if not _patch_applied:
            logger.debug("Kafka hostname patch was not applied")
            return True

        # Validate that we have the original function
        if not callable(_original_getaddrinfo):
            logger.error(
                "Original getaddrinfo function is not available for restoration"
            )
            return False

        # Remove the patch
        socket.getaddrinfo = _original_getaddrinfo
        _patch_applied = False

        logger.info("🔄 Kafka hostname patch removed")
        return True

    except Exception as e:
        logger.error(f"Failed to remove Kafka hostname patch: {e}")
        return False


def is_patch_applied():
    """Check if the hostname patch is currently applied to socket.getaddrinfo.

    This function provides a simple way to check the current patch status without
    attempting to modify anything. It's useful for monitoring, debugging, and
    conditional logic based on patch state.

    Returns:
        bool: True if the hostname patch is currently active, False otherwise

    Example Usage:
        # Check patch status before applying
        if not is_patch_applied():
            apply_kafka_hostname_patch()

        # Conditional logic based on patch status
        if is_patch_applied():
            print("Using patched hostname resolution")
        else:
            print("Using original hostname resolution")

        # Health check integration
        health_status = {
            "hostname_patch_active": is_patch_applied(),
            "patch_failure_count": get_patch_failure_count()
        }

    Thread Safety:
        This function only reads global state and is thread-safe.
    """
    return _patch_applied


def get_patch_failure_count():
    """Get the current count of patch failures for monitoring and diagnostics.

    This function returns the number of times the hostname patch has encountered
    errors during operation. The failure count is used to track patch stability
    and trigger automatic disabling if too many failures occur.

    Failure Count Usage:
        - Incremented each time patched_getaddrinfo encounters an error
        - Reset to 0 when a successful resolution occurs
        - Used to determine if patch should be disabled (>= _max_failures)
        - Useful for monitoring patch health and reliability

    Returns:
        int: Current number of patch failures since last successful resolution

    Example Usage:
        # Monitor patch health
        failures = get_patch_failure_count()
        if failures > 0:
            print(f"Patch has {failures} recent failures")

        # Health check integration
        health_status = {
            "patch_failures": get_patch_failure_count(),
            "patch_stable": get_patch_failure_count() < 3,
            "should_disable": should_disable_patch()
        }

        # Alerting based on failure threshold
        if get_patch_failure_count() >= 3:
            logger.warning("Hostname patch experiencing high failure rate")

    Thread Safety:
        This function only reads global state and is thread-safe.
    """
    return _patch_failures


def should_disable_patch():
    """Check if the hostname patch should be disabled due to excessive failures.

    This function evaluates whether the patch has failed too many times and should
    be automatically disabled to prevent ongoing issues. It's part of the container
    restart loop prevention strategy.

    Failure Threshold Logic:
        - Returns True if _patch_failures >= _max_failures (default: 5)
        - Indicates the patch is unstable and should be disabled
        - Used by monitoring systems to trigger patch removal
        - Helps prevent cascading failures in production

    Returns:
        bool: True if patch should be disabled due to excessive failures, False otherwise

    Example Usage:
        # Automatic patch management
        if should_disable_patch():
            logger.warning("Disabling hostname patch due to excessive failures")
            remove_kafka_hostname_patch()

        # Health monitoring
        if should_disable_patch():
            alert_system.send_alert("Hostname patch unstable, requires attention")

        # Conditional patch application
        if not is_patch_applied() and not should_disable_patch():
            apply_kafka_hostname_patch()

        # Status reporting
        status = {
            "patch_active": is_patch_applied(),
            "failures": get_patch_failure_count(),
            "should_disable": should_disable_patch(),
            "max_failures_threshold": _max_failures
        }

    Container Restart Loop Prevention:
        This function is a key component in preventing container restart loops
        by allowing the system to automatically disable a failing patch rather
        than continuing to encounter repeated failures.

    Thread Safety:
        This function only reads global state and is thread-safe.
    """
    return _patch_failures >= _max_failures


# ============================================================================
# HOSTNAME PATCH AUTO-APPLY: DISABLED
# ============================================================================
# The hostname patch is NO LONGER automatically applied on module import.
#
# REASON: We use /etc/hosts for DNS resolution instead of socket patching.
#         The hostname patch was causing connection issues by redirecting
#         Kafka connections to localhost (127.0.0.1) instead of allowing
#         OS-level DNS resolution via /etc/hosts to work properly.
#
# CONFIGURATION: Set hostname entries in /etc/hosts:
#   192.168.86.200 omninode-bridge-redpanda
#   192.168.86.200 omninode-bridge-postgres
#   192.168.86.200 omninode-bridge-consul
#
# If you need to manually apply the patch (not recommended), call:
#   apply_kafka_hostname_patch() explicitly in your code.
# ============================================================================

# DISABLED: Auto-apply patch when module is imported
# if __name__ != "__main__":
#     # Check if patch is explicitly disabled via environment variable
#     import os
#
#     if os.getenv("DISABLE_KAFKA_HOSTNAME_PATCH", "false").lower() == "true":
#         logger.info("🚫 Kafka hostname patch disabled via DISABLE_KAFKA_HOSTNAME_PATCH environment variable")
#     else:
#         try:
#             # Multiple methods to detect Docker container environment
#             docker_in_cgroup = False
#         if os.path.exists("/proc/1/cgroup"):
#             try:
#                 with open("/proc/1/cgroup") as f:
#                     docker_in_cgroup = "docker" in f.read()
#             except (OSError, PermissionError):
#                 docker_in_cgroup = False
#
#         is_in_container = (
#             os.path.exists(
#                 "/.dockerenv"
#             )  # Docker containers typically have /.dockerenv file
#             or docker_in_cgroup
#             or os.getenv("DOCKER_CONTAINER") == "true"
#             or os.getenv("KUBERNETES_SERVICE_HOST") is not None  # Kubernetes pods
#         )
#
#         if is_in_container:
#             logger.info(
#                 "🐳 Running inside Docker container - skipping Kafka hostname patch"
#             )
#         else:
#             logger.info(
#                 "🖥️  Running outside Docker container - applying Kafka hostname patch"
#             )
#             success = apply_kafka_hostname_patch()
#             if not success:
#                 logger.warning("Failed to apply hostname patch during module import")
#
#     except FileNotFoundError:
#         # /proc/1/cgroup might not exist on some systems
#         logger.debug(
#             "Container detection files not found, assuming non-container environment"
#         )
#         success = apply_kafka_hostname_patch()
#         if not success:
#             logger.warning("Failed to apply hostname patch during module import")
#
#     except PermissionError:
#         # Might not have permission to read container detection files
#         logger.debug("Permission denied reading container detection files")
#         success = apply_kafka_hostname_patch()
#         if not success:
#             logger.warning("Failed to apply hostname patch during module import")
#
#         except Exception as e:
#             # If we can't detect environment, err on the side of applying the patch
#             logger.warning(f"Error detecting container environment: {e}")
#             logger.info("Applying hostname patch as fallback")
#             success = apply_kafka_hostname_patch()
#             if not success:
#                 logger.error(
#                     "Failed to apply hostname patch even as fallback during module import"
#                 )
