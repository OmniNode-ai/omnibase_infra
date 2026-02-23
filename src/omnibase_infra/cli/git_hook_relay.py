# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""onex-git-hook-relay CLI — relay git hook events to Kafka.

Receives git hook payloads and publishes ``ModelGitHookEvent`` instances
to ``onex.evt.git.hook.v1``.

Usage
-----
    onex-git-hook-relay emit \\
        --hook pre-commit \\
        --repo OmniNode-ai/omniclaude \\
        --branch main \\
        --author jsmith \\
        --outcome allowed \\
        --gates-json '["lint", "tests"]'

    onex-git-hook-relay emit \\
        --hook post-receive \\
        --repo OmniNode-ai/omnibase_core \\
        --branch jonah/feature-x \\
        --author jsmith \\
        --outcome pass \\
        --gates-file /tmp/hook-gates.json

Non-Blocking Design
-------------------
If Kafka is unreachable within 2 seconds, the CLI exits 0 (git hooks must
not block the developer workflow) and spools the event to
``~/.onex/spool/git-hooks.jsonl`` for deferred delivery.

Repo Identity
-------------
``--repo`` MUST be in ``{owner}/{name}`` canonical format (e.g.
``OmniNode-ai/omniclaude``). Absolute paths and local names are rejected.
Pattern: ``^[\\w.-]+/[\\w.-]+$``.

Author
------
``--author`` is the GitHub username, NOT an email address. PII (email)
must not cross the event bus.

Topic
-----
``onex.evt.git.hook.v1``
Partition key: ``{repo}:{branch}``

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import click
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Topic constant (mirrors omnibase_core TOPIC_GIT_HOOK_EVENT from PR#531)
TOPIC_GIT_HOOK_EVENT = "onex.evt.git.hook.v1"

# Repo identifier pattern: {owner}/{name} — no absolute paths, no bare names
_REPO_PATTERN = re.compile(r"^[\w.\-]+/[\w.\-]+$")

# Spool file location for non-blocking fallback
_SPOOL_DIR = Path.home() / ".onex" / "spool"
_SPOOL_FILE = _SPOOL_DIR / "git-hooks.jsonl"

# Kafka publish timeout in seconds — keeps git hooks non-blocking
_KAFKA_TIMEOUT_SECONDS = 2.0


class ModelGitHookEmitParams(BaseModel):
    """Parameters for a git hook emit invocation.

    Encapsulates all fields required to build and publish a ModelGitHookEvent.
    Using a model here reduces the parameter count of ``_build_event``
    and ensures consistent validation.

    Attributes:
        hook: Hook name (e.g. ``"pre-commit"``, ``"post-receive"``).
        repo: Repository in ``{owner}/{name}`` format.
        branch: Branch name the hook fired on.
        author: GitHub username (NOT email).
        outcome: Hook outcome string.
        gates: Gate names evaluated during the hook run.
        bootstrap_servers: Kafka bootstrap servers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook: str = Field(..., description="Hook name.")
    repo: str = Field(..., description="Repository in '{owner}/{name}' format.")
    branch: str = Field(..., description="Branch name.")
    author: str = Field(..., description="GitHub username (NOT email).")
    outcome: str = Field(..., description="Hook outcome.")
    gates: list[str] = Field(default_factory=list, description="Gate names evaluated.")
    bootstrap_servers: str = Field(
        default="localhost:29092",
        description="Kafka bootstrap servers.",
    )


def _validate_repo(repo: str) -> None:
    """Raise ValueError if repo is not in {owner}/{name} format."""
    if not _REPO_PATTERN.match(repo):
        raise ValueError(
            f"Invalid repo format: {repo!r}. "
            "Must be '{owner}/{name}' (e.g. 'OmniNode-ai/omniclaude'). "
            "Never use absolute paths or bare repo names."
        )


def _spool_event(event: JsonType) -> None:
    """Write event to spool file for deferred delivery.

    Spool is an append-only JSONL file at ~/.onex/spool/git-hooks.jsonl.
    """
    try:
        _SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        with _SPOOL_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        logger.info("Event spooled to %s", _SPOOL_FILE)
    except OSError as exc:
        logger.warning("Failed to spool event: %s", exc)


async def _publish_event(
    event: JsonType,
    partition_key: str,
    bootstrap_servers: str,
) -> bool:
    """Publish event to Kafka with a 2-second timeout.

    Returns True on success, False if Kafka is unreachable or times out.
    The timeout ensures the CLI never blocks git operations.
    """
    try:
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.event_bus.adapters import AdapterProtocolEventPublisherKafka
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka.default()
        await asyncio.wait_for(bus.start(), timeout=_KAFKA_TIMEOUT_SECONDS)
        try:
            container = ModelONEXContainer()
            adapter = AdapterProtocolEventPublisherKafka(
                container=container,
                bus=bus,
                service_name="onex-git-hook-relay",
            )
            success = await asyncio.wait_for(
                adapter.publish(
                    event_type=TOPIC_GIT_HOOK_EVENT,
                    payload=event,
                    partition_key=partition_key,
                ),
                timeout=_KAFKA_TIMEOUT_SECONDS,
            )
            return bool(success)
        finally:
            await bus.close()
    except TimeoutError:
        logger.warning(
            "Kafka publish timed out after %.1fs — spooling event",
            _KAFKA_TIMEOUT_SECONDS,
        )
        return False
    except Exception as exc:
        logger.warning("Kafka publish failed: %s — spooling event", exc)
        return False


def _build_event(params: ModelGitHookEmitParams) -> JsonType:
    """Build the event payload dict matching ModelGitHookEvent schema.

    Args:
        params: Validated emit parameters.

    Returns:
        JsonType event payload dict.
    """
    gates_json_value: JsonType = list(params.gates)
    return {
        "event_type": TOPIC_GIT_HOOK_EVENT,
        "hook": params.hook,
        "repo": params.repo,
        "branch": params.branch,
        "author": params.author,
        "outcome": params.outcome,
        "gates": gates_json_value,
        "correlation_id": str(uuid4()),
        "emitted_at": datetime.now(tz=UTC).isoformat(),
    }


def _resolve_gates(gates_json: str | None, gates_file: Path | None) -> list[str]:
    """Resolve gate names from --gates-json or --gates-file.

    Returns the parsed list of gate name strings.
    Calls sys.exit(1) on parse error.
    """
    if gates_json is not None and gates_file is not None:
        click.echo("ERROR: Provide only one of --gates-json or --gates-file.", err=True)
        sys.exit(1)

    if gates_json is not None:
        try:
            parsed = json.loads(gates_json)
            if not isinstance(parsed, list):
                raise ValueError("gates-json must be a JSON array")
            return [str(g) for g in parsed]
        except (json.JSONDecodeError, ValueError) as exc:
            click.echo(f"ERROR: Invalid --gates-json: {exc}", err=True)
            sys.exit(1)

    if gates_file is not None:
        try:
            parsed = json.loads(gates_file.read_text(encoding="utf-8"))
            if not isinstance(parsed, list):
                raise ValueError("gates-file must contain a JSON array")
            return [str(g) for g in parsed]
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            click.echo(f"ERROR: Invalid --gates-file: {exc}", err=True)
            sys.exit(1)

    return []


@click.group()
@click.option(
    "--bootstrap-servers",
    default=lambda: os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
    show_default=True,
    help="Kafka bootstrap servers (overrides KAFKA_BOOTSTRAP_SERVERS env var).",
)
@click.pass_context
def cli(ctx: click.Context, bootstrap_servers: str) -> None:
    """onex-git-hook-relay — relay git hook events to Kafka.

    Non-blocking: if Kafka is unreachable within 2s, events are spooled to
    ~/.onex/spool/git-hooks.jsonl and the CLI exits 0.
    """
    ctx.ensure_object(dict)
    ctx.obj["bootstrap_servers"] = bootstrap_servers


@cli.command("emit")
@click.option(
    "--hook", required=True, help="Hook name (e.g. pre-commit, post-receive)."
)
@click.option(
    "--repo",
    required=True,
    help="Repository in '{owner}/{name}' format (e.g. OmniNode-ai/omniclaude).",
)
@click.option("--branch", required=True, help="Branch name the hook fired on.")
@click.option(
    "--author",
    required=True,
    help="GitHub username of the committer/pusher (NOT email).",
)
@click.option(
    "--outcome",
    required=True,
    type=click.Choice(["pass", "fail", "allowed", "blocked"], case_sensitive=False),
    help="Hook outcome.",
)
@click.option(
    "--gates-json",
    default=None,
    help='Inline JSON array of gate names (e.g. \'["lint","tests"]\').',
)
@click.option(
    "--gates-file",
    default=None,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help="Path to JSON file containing gate names array.",
)
@click.pass_context
def emit(ctx: click.Context, /, **kwargs: object) -> None:
    """Emit a git hook event to Kafka.

    Provide gate names via --gates-json or --gates-file (mutually exclusive).

    Examples::

        onex-git-hook-relay emit --hook pre-commit --repo OmniNode-ai/omniclaude \\
            --branch main --author jsmith --outcome allowed --gates-json '[]'
    """
    hook = str(kwargs["hook"])
    repo = str(kwargs["repo"])
    branch = str(kwargs["branch"])
    author = str(kwargs["author"])
    outcome = str(kwargs["outcome"])
    gates_json_raw = kwargs.get("gates_json")
    gates_file_raw = kwargs.get("gates_file")
    gates_json: str | None = str(gates_json_raw) if gates_json_raw is not None else None
    gates_file: Path | None = (
        gates_file_raw if isinstance(gates_file_raw, Path) else None
    )

    # Validate repo format
    try:
        _validate_repo(repo)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    gates = _resolve_gates(gates_json, gates_file)
    params = ModelGitHookEmitParams(
        hook=hook,
        repo=repo,
        branch=branch,
        author=author,
        outcome=outcome.lower(),
        gates=gates,
        bootstrap_servers=ctx.obj["bootstrap_servers"],
    )

    event = _build_event(params)
    partition_key = f"{repo}:{branch}"

    success = asyncio.run(
        _publish_event(event, partition_key, params.bootstrap_servers)
    )

    if not success:
        _spool_event(event)
        click.echo(
            f"[onex-git-hook-relay] Kafka unavailable — event spooled to {_SPOOL_FILE}",
            err=True,
        )
    else:
        click.echo(f"[onex-git-hook-relay] Published {hook} event for {repo}:{branch}")

    # Always exit 0 — git hooks must not be blocked by Kafka unavailability
    sys.exit(0)


def main() -> None:
    """Entry point for onex-git-hook-relay CLI."""
    logging.basicConfig(level=logging.WARNING)
    cli()


__all__ = ["cli", "main", "ModelGitHookEmitParams"]
