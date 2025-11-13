#!/usr/bin/env python3
"""
Pre-Push Git Hook - File Change Event Publisher

Lightweight git hook that publishes file change events to Kafka when files are pushed.
Designed for minimal overhead (<2s execution time) with graceful degradation.

Usage:
    This script is installed as .git/hooks/pre-push by install_git_hooks.sh

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (default: localhost:29092)
    KAFKA_ENABLE_LOGGING: Enable Kafka event logging (default: true)
    GIT_HOOK_TIMEOUT: Max execution time in seconds (default: 2)
    GIT_HOOK_DEBUG: Enable debug logging (default: false)

Exit Codes:
    0: Success (always returns 0 to allow push even if Kafka fails)

ONEX v2.0 Compliance:
- Uses ModelFileChangeEvent for standardized event format
- Publishes to omninode_file_changes_v1 topic
- Correlation ID for distributed tracing
- Graceful degradation with structured logging
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

# Configure logging
DEBUG = os.getenv("GIT_HOOK_DEBUG", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[git-hook] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Performance tracking
HOOK_TIMEOUT = int(os.getenv("GIT_HOOK_TIMEOUT", "2"))
START_TIME = time.time()


def get_elapsed_ms() -> float:
    """Get elapsed time in milliseconds since hook started."""
    return (time.time() - START_TIME) * 1000


def get_repo_info() -> dict:
    """
    Get repository information from git.

    Returns:
        Dictionary with repo_name, repo_path, branch, author_name, author_email
    """
    try:
        # Get repo root path
        repo_path = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Get repo name from path
        repo_name = Path(repo_path).name

        # Get current branch
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Get author info (if available)
        try:
            author_name = (
                subprocess.check_output(
                    ["git", "config", "user.name"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            author_name = None

        try:
            author_email = (
                subprocess.check_output(
                    ["git", "config", "user.email"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            author_email = None

        logger.debug(f"Repository info: {repo_name} at {repo_path} on branch {branch}")

        return {
            "repo_name": repo_name,
            "repo_path": repo_path,
            "branch": branch,
            "author_name": author_name,
            "author_email": author_email,
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get repository info: {e}")
        return {}


def get_changed_files_since_remote() -> list[str]:
    """
    Get list of changed files since last push to remote.

    Returns files that differ between local and remote branch.
    If remote tracking branch doesn't exist, returns files in working tree.

    Returns:
        List of file paths relative to repo root
    """
    try:
        # Get current branch
        current_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Try to get remote tracking branch
        try:
            remote_branch = (
                subprocess.check_output(
                    [
                        "git",
                        "rev-parse",
                        "--abbrev-ref",
                        f"{current_branch}@{{upstream}}",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )

            # Get diff between local and remote
            output = subprocess.check_output(
                ["git", "diff", "--name-only", remote_branch, current_branch],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")

            files = [f.strip() for f in output.split("\n") if f.strip()]
            logger.debug(f"Found {len(files)} changed files since {remote_branch}")

        except subprocess.CalledProcessError:
            # No remote tracking branch, get all staged + modified files
            logger.debug(
                f"No remote tracking branch for {current_branch}, using working tree"
            )

            # Get staged files
            staged = subprocess.check_output(
                ["git", "diff", "--name-only", "--cached"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")

            # Get modified files
            modified = subprocess.check_output(
                ["git", "diff", "--name-only"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")

            # Combine and deduplicate
            files = list(
                {f.strip() for f in (staged + modified).split("\n") if f.strip()}
            )
            logger.debug(f"Found {len(files)} changed files in working tree")

        return files

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get changed files: {e}")
        return []


async def publish_file_change_event(
    files: list[str],
    repo_info: dict,
) -> bool:
    """
    Publish file change event to Kafka.

    Args:
        files: List of changed file paths
        repo_info: Repository information dictionary

    Returns:
        True if published successfully, False otherwise
    """
    if not files:
        logger.debug("No files to publish, skipping event")
        return True

    if get_elapsed_ms() > (HOOK_TIMEOUT * 1000):
        logger.warning(f"Timeout exceeded ({HOOK_TIMEOUT}s), skipping Kafka publish")
        return False

    # Check if Kafka logging is enabled
    if os.getenv("KAFKA_ENABLE_LOGGING", "true").lower() != "true":
        logger.info("Kafka logging disabled, skipping event publish")
        return True

    try:
        # Import required modules (lazy import for performance)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from omninode_bridge.events.models.file_change_events import (
            EnumGitOperation,
            ModelFileChangeEvent,
        )
        from omninode_bridge.services.kafka_client import KafkaClient

        # Create correlation ID
        correlation_id = uuid4()

        # Create file change event
        event = ModelFileChangeEvent(
            correlation_id=correlation_id,
            files=files,
            repo_name=repo_info["repo_name"],
            repo_path=repo_info["repo_path"],
            branch=repo_info["branch"],
            operation=EnumGitOperation.PRE_PUSH,
            author_name=repo_info.get("author_name"),
            author_email=repo_info.get("author_email"),
        )

        logger.debug(f"Created file change event: {correlation_id}")

        # Initialize Kafka client
        kafka_client = KafkaClient(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
            timeout_seconds=max(1, HOOK_TIMEOUT - int(get_elapsed_ms() / 1000)),
        )

        # Connect to Kafka (with timeout)
        connect_start = time.time()
        await asyncio.wait_for(
            kafka_client.connect(),
            timeout=max(0.5, HOOK_TIMEOUT - (time.time() - START_TIME)),
        )
        connect_time_ms = (time.time() - connect_start) * 1000

        logger.debug(f"Kafka connected in {connect_time_ms:.0f}ms")

        # Publish event (with timeout)
        publish_start = time.time()
        success = await asyncio.wait_for(
            kafka_client.publish_raw_event(
                topic=event.to_kafka_topic(),
                data=event.to_dict(),
                key=str(correlation_id),
            ),
            timeout=max(0.5, HOOK_TIMEOUT - (time.time() - START_TIME)),
        )
        publish_time_ms = (time.time() - publish_start) * 1000

        # Disconnect
        await kafka_client.disconnect()

        if success:
            logger.info(
                f"Published file change event for {len(files)} files "
                f"(correlation_id={correlation_id}, publish_time={publish_time_ms:.0f}ms)"
            )
        else:
            logger.warning("Failed to publish file change event (Kafka unavailable)")

        return success

    except TimeoutError:
        logger.warning(f"Kafka publish timeout after {get_elapsed_ms():.0f}ms")
        return False

    except Exception as e:
        logger.warning(f"Failed to publish file change event: {e}")
        if DEBUG:
            import traceback

            logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """
    Main git hook entry point.

    Always exits with 0 (success) to allow push even if Kafka fails.
    """
    logger.debug("Pre-push hook started")

    try:
        # Get repository info
        repo_info = get_repo_info()
        if not repo_info:
            logger.warning("Failed to get repository info, skipping event publish")
            sys.exit(0)

        # Get changed files
        files = get_changed_files_since_remote()
        if not files:
            logger.debug("No changed files detected, skipping event publish")
            sys.exit(0)

        logger.info(f"Detected {len(files)} changed files")

        # Publish event to Kafka (async)
        success = asyncio.run(publish_file_change_event(files, repo_info))

        total_time_ms = get_elapsed_ms()
        logger.debug(
            f"Pre-push hook completed in {total_time_ms:.0f}ms (success={success})"
        )

        # Always exit with 0 to allow push (non-blocking)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Pre-push hook failed: {e}")
        if DEBUG:
            import traceback

            logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Always exit with 0 to allow push (non-blocking)
        sys.exit(0)


if __name__ == "__main__":
    main()
