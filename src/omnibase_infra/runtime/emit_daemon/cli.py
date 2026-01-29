# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Command-Line Interface for Hook Event Emit Daemon.

This module provides the CLI for managing and testing the emit daemon.
The daemon provides a Unix socket interface for Claude Code hooks to emit
events to Kafka without blocking hook execution.

Commands:
    start   - Start the daemon (foreground or daemonized)
    stop    - Stop the running daemon
    status  - Check daemon status and queue metrics
    emit    - Emit a test event
    config  - Show resolved configuration

Usage Examples:
    # Start daemon in foreground
    emit-daemon start --kafka-servers localhost:9092

    # Start daemon in background
    emit-daemon start --kafka-servers localhost:9092 --daemonize

    # Check status
    emit-daemon status

    # Emit a test event
    emit-daemon emit --event-type prompt.submitted --payload '{"session_id": "test"}'

    # Stop daemon
    emit-daemon stop

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

import yaml

from omnibase_core.errors import OnexError
from omnibase_infra import __version__
from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig
from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon

# Default timeout for socket operations (seconds)
DEFAULT_SOCKET_TIMEOUT: float = 5.0

# Default graceful shutdown wait time (seconds)
DEFAULT_SHUTDOWN_WAIT: float = 30.0


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI operations.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config_file(config_path: Path) -> dict[str, object]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values. Values are str, int, float, or bool
        as parsed from YAML. Pydantic handles final type validation.

    Raises:
        SystemExit: If file cannot be read or parsed.
    """
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                print("Error: Config file must be a YAML mapping", file=sys.stderr)
                sys.exit(1)
            # Filter out None values, keep all YAML-parsed types (str, int, float, bool)
            # Pydantic handles final type validation and conversion
            result: dict[str, object] = {}
            for key, value in data.items():
                if value is not None:
                    result[key] = value
            return result
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Cannot read config file: {e}", file=sys.stderr)
        sys.exit(1)


def _build_config(args: argparse.Namespace) -> ModelEmitDaemonConfig:
    """Build configuration from CLI arguments and config file.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Validated configuration model.

    Raises:
        SystemExit: If configuration is invalid.
    """
    # Start with empty config dict (object allows all config value types)
    config_dict: dict[str, object] = {}

    # Load from config file if specified
    config_file = getattr(args, "config", None)
    if config_file:
        config_dict.update(_load_config_file(Path(config_file)))

    # Apply CLI overrides (these take precedence over config file)
    if hasattr(args, "kafka_servers") and args.kafka_servers:
        config_dict["kafka_bootstrap_servers"] = args.kafka_servers

    if hasattr(args, "socket_path") and args.socket_path:
        config_dict["socket_path"] = Path(args.socket_path)

    if hasattr(args, "pid_path") and args.pid_path:
        config_dict["pid_path"] = Path(args.pid_path)

    if hasattr(args, "spool_dir") and args.spool_dir:
        config_dict["spool_dir"] = Path(args.spool_dir)

    try:
        # Use with_env_overrides to also pick up environment variables
        return ModelEmitDaemonConfig.with_env_overrides(**config_dict)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)


def _send_socket_message(
    socket_path: Path,
    message: dict[str, object],
    timeout: float = DEFAULT_SOCKET_TIMEOUT,
) -> dict[str, object]:
    """Send a message to the daemon via Unix socket.

    Args:
        socket_path: Path to the Unix domain socket.
        message: JSON-serializable message to send.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON response from daemon.

    Raises:
        ConnectionError: If cannot connect to daemon.
        TimeoutError: If operation times out.
        OnexError: If response is invalid JSON.
    """
    if not socket_path.exists():
        raise ConnectionError(f"Socket not found: {socket_path}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(str(socket_path))

        # Send message (newline-delimited JSON)
        request = json.dumps(message) + "\n"
        sock.sendall(request.encode("utf-8"))

        # Read response
        response_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
            if b"\n" in response_data:
                break

        if not response_data:
            raise ConnectionError("Empty response from daemon")

        # Parse response
        response_str = response_data.decode("utf-8").strip()
        result = json.loads(response_str)
        if not isinstance(result, dict):
            raise OnexError("Response must be a JSON object")
        return result

    except TimeoutError as e:
        raise TimeoutError(f"Socket operation timed out: {e}") from e
    except json.JSONDecodeError as e:
        raise OnexError(f"Invalid JSON response: {e}") from e
    except OSError as e:
        raise ConnectionError(f"Socket error: {e}") from e
    finally:
        sock.close()


def _read_pid_file(pid_path: Path) -> int | None:
    """Read PID from daemon PID file.

    Args:
        pid_path: Path to the PID file.

    Returns:
        The PID if file exists and is valid, None otherwise.
    """
    if not pid_path.exists():
        return None

    try:
        pid_str = pid_path.read_text().strip()
        return int(pid_str)
    except (OSError, ValueError):
        return None


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Args:
        pid: Process ID to check.

    Returns:
        True if process is running, False otherwise.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it
        return True


def _daemonize() -> None:
    """Fork the current process into a daemon.

    Double-fork to prevent zombie processes and detach from terminal.
    After this call, the parent process should exit.

    Raises:
        SystemExit: On Windows (os.fork() is not available).
    """
    # Check platform - fork() is Unix-only
    if sys.platform == "win32":
        print("Error: --daemonize is not supported on Windows", file=sys.stderr)
        sys.exit(1)

    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent process - exit
        sys.exit(0)

    # Become session leader
    os.setsid()

    # Second fork (prevent acquiring a controlling terminal)
    pid = os.fork()
    if pid > 0:
        # First child - exit
        sys.exit(0)

    # We're now in the grandchild (daemon) process

    # Change working directory to root
    os.chdir("/")

    # Reset file creation mask to secure default (rw-r--r-- for files, rwxr-xr-x for dirs)
    os.umask(0o022)

    # Redirect standard file descriptors to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def cmd_start(args: argparse.Namespace) -> None:
    """Handle the 'start' command.

    Start the emit daemon in foreground or background mode.

    Args:
        args: Parsed CLI arguments.
    """
    _setup_logging(verbose=args.verbose)

    # Build configuration
    config = _build_config(args)

    # Check if daemon is already running
    pid = _read_pid_file(config.pid_path)
    if pid is not None and _is_process_running(pid):
        print(f"Error: Daemon already running with PID {pid}", file=sys.stderr)
        sys.exit(1)

    # Daemonize if requested
    if args.daemonize:
        log_dir = config.spool_dir.parent
        log_file = log_dir / "emit-daemon.log"
        # Create log directory BEFORE daemonizing so errors are visible to user
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: Cannot create log directory: {e}", file=sys.stderr)
            sys.exit(1)
        print("Starting emit-daemon in background...")
        print(f"  Socket: {config.socket_path}")
        print(f"  PID file: {config.pid_path}")
        print(f"  Kafka: {config.kafka_bootstrap_servers}")
        print(f"  Log file: {log_file}")
        _daemonize()
        # NOTE: After daemonize, stdio is redirected to /dev/null.
        # Startup errors will appear in the log file above.
        # Clear any existing handlers to ensure reconfiguration works
        # (logging.basicConfig() is a no-op if handlers already exist)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        # Set up file-based logging (directory already exists from above)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filename=str(log_file),
            filemode="a",
        )

    # Run the daemon
    async def run_daemon() -> None:
        daemon = EmitDaemon(config)
        try:
            await daemon.start()
            if not args.daemonize:
                print(f"Emit daemon started (PID: {os.getpid()})")
                print(f"  Socket: {config.socket_path}")
                print(f"  Kafka: {config.kafka_bootstrap_servers}")
                print("Press Ctrl+C to stop...")
            await daemon.run_until_shutdown()
        except OnexError as e:
            if not args.daemonize:
                print(f"Error: {e}", file=sys.stderr)
            logging.getLogger(__name__).exception("Daemon error")
            sys.exit(1)

    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        if not args.daemonize:
            print("\nShutting down...")
        sys.exit(0)


def cmd_stop(args: argparse.Namespace) -> None:
    """Handle the 'stop' command.

    Stop the running emit daemon gracefully.

    Args:
        args: Parsed CLI arguments.
    """
    # Get PID file path (intentional /tmp default for daemon PID tracking)
    pid_path = Path(getattr(args, "pid_path", None) or "/tmp/omniclaude-emit.pid")  # noqa: S108

    # If config file provided, use its pid_path
    if args.config:
        config_data = _load_config_file(Path(args.config))
        if "pid_path" in config_data:
            pid_path = Path(str(config_data["pid_path"]))

    # Read PID
    pid = _read_pid_file(pid_path)
    if pid is None:
        print("Daemon not running (no PID file)", file=sys.stderr)
        sys.exit(1)

    if not _is_process_running(pid):
        print(f"Daemon not running (stale PID file for PID {pid})")
        # Clean up stale PID file
        try:
            pid_path.unlink()
            print(f"Removed stale PID file: {pid_path}")
        except OSError:
            pass
        sys.exit(0)

    # Send SIGTERM for graceful shutdown
    print(f"Stopping emit-daemon (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("Daemon already stopped")
        sys.exit(0)
    except PermissionError:
        print(f"Error: Permission denied to stop daemon (PID: {pid})", file=sys.stderr)
        sys.exit(1)

    # Wait for graceful shutdown
    wait_time = DEFAULT_SHUTDOWN_WAIT
    for _ in range(int(wait_time * 10)):  # Check every 0.1s
        if not _is_process_running(pid):
            print("Daemon stopped successfully")
            # Clean up PID file if daemon didn't
            if pid_path.exists():
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            sys.exit(0)
        time.sleep(0.1)

    # Process didn't stop gracefully - warn user
    print(
        f"Warning: Daemon (PID: {pid}) did not stop within {wait_time}s",
        file=sys.stderr,
    )
    print(f"You may need to force kill with: kill -9 {pid}", file=sys.stderr)
    sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Handle the 'status' command.

    Check daemon status via ping command.

    Args:
        args: Parsed CLI arguments.
    """
    # Get socket path (intentional /tmp default for daemon socket)
    socket_path = Path(
        getattr(args, "socket_path", None) or "/tmp/omniclaude-emit.sock"  # noqa: S108
    )

    # If config file provided, use its socket_path
    if args.config:
        config_data = _load_config_file(Path(args.config))
        if "socket_path" in config_data:
            socket_path = Path(str(config_data["socket_path"]))

    try:
        response = _send_socket_message(socket_path, {"command": "ping"})

        if response.get("status") == "ok":
            print("Daemon is running")
            print(f"  Queue size (memory): {response.get('queue_size', 'unknown')}")
            print(f"  Queue size (spool):  {response.get('spool_size', 'unknown')}")

            # Print as JSON if requested
            if args.json:
                print(json.dumps(response, indent=2))

            sys.exit(0)
        else:
            print(f"Daemon returned unexpected status: {response}", file=sys.stderr)
            sys.exit(1)

    except ConnectionError as e:
        print(f"Daemon not running: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(f"Daemon not responding: {e}", file=sys.stderr)
        sys.exit(1)
    except OnexError as e:
        print(f"Invalid response from daemon: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_emit(args: argparse.Namespace) -> None:
    """Handle the 'emit' command.

    Emit a test event to the daemon.

    Args:
        args: Parsed CLI arguments.
    """
    # Get socket path (intentional /tmp default for daemon socket)
    socket_path = Path(
        getattr(args, "socket_path", None) or "/tmp/omniclaude-emit.sock"  # noqa: S108
    )

    # If config file provided, use its socket_path
    if args.config:
        config_data = _load_config_file(Path(args.config))
        if "socket_path" in config_data:
            socket_path = Path(str(config_data["socket_path"]))

    # Parse payload JSON
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            print("Error: Payload must be a JSON object", file=sys.stderr)
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON payload: {e}", file=sys.stderr)
        sys.exit(1)

    # Build event message
    message: dict[str, object] = {
        "event_type": args.event_type,
        "payload": payload,
    }

    try:
        response = _send_socket_message(socket_path, message)

        if response.get("status") == "queued":
            event_id = response.get("event_id", "unknown")
            print("Event queued successfully")
            print(f"  Event ID: {event_id}")
            print(f"  Type: {args.event_type}")

            if args.json:
                print(json.dumps(response, indent=2))

            sys.exit(0)
        elif response.get("status") == "error":
            reason = response.get("reason", "Unknown error")
            print(f"Error: {reason}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Unexpected response: {response}", file=sys.stderr)
            sys.exit(1)

    except ConnectionError as e:
        print(f"Cannot connect to daemon: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(f"Daemon not responding: {e}", file=sys.stderr)
        sys.exit(1)
    except OnexError as e:
        print(f"Invalid response from daemon: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """Handle the 'config' command.

    Show resolved configuration.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        config = _build_config(args)
    except SystemExit:
        # If config building fails, show what we can
        if args.config:
            config_data = _load_config_file(Path(args.config))
            print("# Configuration from file (not validated):")
            print(yaml.dump(config_data, default_flow_style=False))
        else:
            print("# Default configuration:")
            print("# (Requires kafka_bootstrap_servers to be set)")
            print()
            print("socket_path: /tmp/omniclaude-emit.sock")
            print("pid_path: /tmp/omniclaude-emit.pid")
            print("spool_dir: ~/.omniclaude/emit-spool")
            print("kafka_bootstrap_servers: <REQUIRED>")
            print("kafka_client_id: emit-daemon")
            print("max_payload_bytes: 1048576")
            print("max_memory_queue: 100")
            print("max_spool_messages: 1000")
            print("max_spool_bytes: 10485760")
            print("socket_timeout_seconds: 5.0")
            print("kafka_timeout_seconds: 30.0")
            print("shutdown_drain_seconds: 10.0")
        return

    # Convert config to dict for YAML output
    config_dict = {
        "socket_path": str(config.socket_path),
        "pid_path": str(config.pid_path),
        "spool_dir": str(config.spool_dir),
        "kafka_bootstrap_servers": config.kafka_bootstrap_servers,
        "kafka_client_id": config.kafka_client_id,
        "max_payload_bytes": config.max_payload_bytes,
        "max_memory_queue": config.max_memory_queue,
        "max_spool_messages": config.max_spool_messages,
        "max_spool_bytes": config.max_spool_bytes,
        "socket_timeout_seconds": config.socket_timeout_seconds,
        "kafka_timeout_seconds": config.kafka_timeout_seconds,
        "shutdown_drain_seconds": config.shutdown_drain_seconds,
        "spooling_enabled": config.spooling_enabled,
    }

    print("# Resolved configuration:")
    print(yaml.dump(config_dict, default_flow_style=False))


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="emit-daemon",
        description="Hook Event Emit Daemon - Persistent Kafka event emission for Claude Code hooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show version
  emit-daemon --version

  # Start daemon in foreground
  emit-daemon start --kafka-servers localhost:9092

  # Start daemon in background
  emit-daemon start --kafka-servers localhost:9092 --daemonize

  # Start with config file
  emit-daemon start --config /path/to/config.yaml

  # Check daemon status
  emit-daemon status

  # Emit a test event
  emit-daemon emit --event-type prompt.submitted --payload '{"session_id": "test"}'

  # Stop the daemon
  emit-daemon stop

  # Show resolved configuration
  emit-daemon config --config /path/to/config.yaml
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        required=True,
    )

    # -------------------------------------------------------------------------
    # start command
    # -------------------------------------------------------------------------
    start_parser = subparsers.add_parser(
        "start",
        help="Start the emit daemon",
        description="Start the emit daemon in foreground or background mode.",
    )
    start_parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to YAML configuration file",
    )
    start_parser.add_argument(
        "--kafka-servers",
        "-k",
        metavar="SERVERS",
        help="Kafka bootstrap servers (e.g., localhost:9092). Overrides config file.",
    )
    start_parser.add_argument(
        "--socket-path",
        "-s",
        metavar="PATH",
        help="Unix socket path. Overrides config file.",
    )
    start_parser.add_argument(
        "--pid-path",
        "-p",
        metavar="PATH",
        help="PID file path. Overrides config file.",
    )
    start_parser.add_argument(
        "--spool-dir",
        metavar="DIR",
        help="Spool directory path. Overrides config file.",
    )
    start_parser.add_argument(
        "--daemonize",
        "-d",
        action="store_true",
        help="Run daemon in background (fork and exit)",
    )
    start_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    start_parser.set_defaults(func=cmd_start)

    # -------------------------------------------------------------------------
    # stop command
    # -------------------------------------------------------------------------
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the running daemon",
        description="Stop the running emit daemon gracefully.",
    )
    stop_parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to YAML configuration file (to find PID file)",
    )
    stop_parser.add_argument(
        "--pid-path",
        "-p",
        metavar="PATH",
        help="PID file path (default: /tmp/omniclaude-emit.pid)",
    )
    stop_parser.set_defaults(func=cmd_stop)

    # -------------------------------------------------------------------------
    # status command
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Check daemon status",
        description="Check if the daemon is running and show queue metrics.",
    )
    status_parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to YAML configuration file (to find socket path)",
    )
    status_parser.add_argument(
        "--socket-path",
        "-s",
        metavar="PATH",
        help="Unix socket path (default: /tmp/omniclaude-emit.sock)",
    )
    status_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output full status as JSON",
    )
    status_parser.set_defaults(func=cmd_status)

    # -------------------------------------------------------------------------
    # emit command
    # -------------------------------------------------------------------------
    emit_parser = subparsers.add_parser(
        "emit",
        help="Emit a test event",
        description="Emit an event to the daemon for publishing to Kafka.",
    )
    emit_parser.add_argument(
        "--event-type",
        "-t",
        required=True,
        metavar="TYPE",
        help="Event type (e.g., prompt.submitted)",
    )
    emit_parser.add_argument(
        "--payload",
        "-p",
        required=True,
        metavar="JSON",
        help='Event payload as JSON string (e.g., \'{"key": "value"}\')',
    )
    emit_parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to YAML configuration file (to find socket path)",
    )
    emit_parser.add_argument(
        "--socket-path",
        "-s",
        metavar="PATH",
        help="Unix socket path (default: /tmp/omniclaude-emit.sock)",
    )
    emit_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output full response as JSON",
    )
    emit_parser.set_defaults(func=cmd_emit)

    # -------------------------------------------------------------------------
    # config command
    # -------------------------------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Show configuration",
        description="Show the resolved configuration as YAML.",
    )
    config_parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        help="Path to YAML configuration file",
    )
    config_parser.add_argument(
        "--kafka-servers",
        "-k",
        metavar="SERVERS",
        help="Kafka bootstrap servers (required if no config file)",
    )
    config_parser.set_defaults(func=cmd_config)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Call the appropriate command handler
    args.func(args)


if __name__ == "__main__":
    main()
