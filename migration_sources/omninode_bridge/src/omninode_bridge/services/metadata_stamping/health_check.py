#!/usr/bin/env python3
"""
Health check script for metadata stamping service.

This script provides comprehensive health checking for Docker containers,
including readiness and liveness probes with proper exit codes.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def make_http_request(url: str, timeout: int = 10) -> Optional[dict[str, Any]]:
    """Make HTTP request with error handling.

    Args:
        url: URL to request
        timeout: Request timeout in seconds

    Returns:
        Response data or None if failed
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status == 200:
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {"status": "ok"}
            else:
                logging.error(f"HTTP request failed with status {response.status}")
                return None
    except urllib.error.URLError as e:
        logging.error(f"URL error: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


def check_readiness(host: str = "localhost", port: int = 8053) -> bool:
    """Check if service is ready to serve requests.

    Args:
        host: Service host
        port: Service port

    Returns:
        True if service is ready
    """
    logging.info("Performing readiness check...")

    # Check basic service endpoint
    url = f"http://{host}:{port}/"
    response = make_http_request(url, timeout=5)

    if not response:
        logging.error("Service root endpoint not responding")
        return False

    # Check health endpoint
    health_url = f"http://{host}:{port}/api/v1/metadata-stamping/health"
    health_response = make_http_request(health_url, timeout=10)

    if not health_response:
        logging.error("Health endpoint not responding")
        return False

    # Parse health response
    if isinstance(health_response, dict):
        if health_response.get("status") == "success":
            data = health_response.get("data", {})
            if data.get("status") in ["healthy", "degraded"]:
                logging.info(f"Service ready - status: {data.get('status')}")
                return True

    logging.error(f"Service not ready - health response: {health_response}")
    return False


def check_liveness(host: str = "localhost", port: int = 8053) -> bool:
    """Check if service is alive and functioning.

    Args:
        host: Service host
        port: Service port

    Returns:
        True if service is alive
    """
    logging.info("Performing liveness check...")

    # Check basic connectivity
    url = f"http://{host}:{port}/"
    response = make_http_request(url, timeout=3)

    if not response:
        logging.error("Service not responding to basic requests")
        return False

    logging.info("Service is alive")
    return True


def check_startup(
    host: str = "localhost", port: int = 8053, max_attempts: int = 30
) -> bool:
    """Check if service has completed startup.

    Args:
        host: Service host
        port: Service port
        max_attempts: Maximum startup attempts

    Returns:
        True if service has started successfully
    """
    logging.info(f"Performing startup check (max {max_attempts} attempts)...")

    for attempt in range(max_attempts):
        # Try basic connectivity
        url = f"http://{host}:{port}/"
        response = make_http_request(url, timeout=2)

        if response:
            # Check if health endpoint is available
            health_url = f"http://{host}:{port}/api/v1/metadata-stamping/health"
            health_response = make_http_request(health_url, timeout=5)

            if health_response:
                logging.info(f"Service startup completed after {attempt + 1} attempts")
                return True

        if attempt < max_attempts - 1:
            logging.info(
                f"Startup attempt {attempt + 1}/{max_attempts} failed, retrying in 2s..."
            )
            time.sleep(2)

    logging.error(f"Service failed to start after {max_attempts} attempts")
    return False


def check_metrics(host: str = "localhost", port: int = 9090) -> bool:
    """Check if metrics endpoint is available.

    Args:
        host: Metrics host
        port: Metrics port

    Returns:
        True if metrics are available
    """
    logging.info("Checking metrics endpoint...")

    url = f"http://{host}:{port}/metrics"
    response = make_http_request(url, timeout=5)

    if response or urllib.request.urlopen(url, timeout=5).status == 200:
        logging.info("Metrics endpoint is available")
        return True
    else:
        logging.warning("Metrics endpoint not available")
        return False


def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(
        description="Metadata Stamping Service Health Check"
    )
    parser.add_argument(
        "check_type",
        choices=["readiness", "liveness", "startup", "metrics"],
        help="Type of health check to perform",
    )
    parser.add_argument(
        "--host", default="localhost", help="Service host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8053, help="Service port (default: 8053)"
    )
    parser.add_argument(
        "--metrics-port", type=int, default=9090, help="Metrics port (default: 9090)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Startup timeout in attempts (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet mode - only errors")

    args = parser.parse_args()

    # Setup logging
    log_level = "ERROR" if args.quiet else args.log_level
    setup_logging(log_level)

    # Perform the requested check
    success = False

    try:
        if args.check_type == "readiness":
            success = check_readiness(args.host, args.port)
        elif args.check_type == "liveness":
            success = check_liveness(args.host, args.port)
        elif args.check_type == "startup":
            success = check_startup(args.host, args.port, args.timeout)
        elif args.check_type == "metrics":
            success = check_metrics(args.host, args.metrics_port)

        if success:
            if not args.quiet:
                print(f"✓ {args.check_type.capitalize()} check passed")
            sys.exit(0)
        else:
            if not args.quiet:
                print(f"✗ {args.check_type.capitalize()} check failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Health check interrupted")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Health check error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
