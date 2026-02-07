# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Main CLI entry point for ``onex-infra-test``.

Provides end-to-end integration testing commands for the ONEX infrastructure
stack. All subcommands are registered here.

Usage::

    onex-infra-test env up              # Start local infra
    onex-infra-test env down            # Stop local infra
    onex-infra-test introspect          # Trigger introspection
    onex-infra-test verify registry     # Check registry state
    onex-infra-test verify topics       # Check topic compliance
    onex-infra-test verify snapshots    # Check snapshot topic state
    onex-infra-test verify idempotency  # Run twice, assert no duplicates
    onex-infra-test run --suite smoke   # Happy path
    onex-infra-test run --suite failure # Runtime-down-then-recovery
"""

from __future__ import annotations

import click

from omnibase_infra.cli.infra_test.env import env
from omnibase_infra.cli.infra_test.introspect import introspect
from omnibase_infra.cli.infra_test.run_suite import run_suite
from omnibase_infra.cli.infra_test.verify import verify


@click.group()
@click.option(
    "--compose-file",
    default="docker/docker-compose.e2e.yml",
    envvar="ONEX_COMPOSE_FILE",
    help="Path to Docker Compose file.",
    show_default=True,
)
@click.option(
    "--project-name",
    default="omnibase-infra",
    envvar="ONEX_PROJECT_NAME",
    help="Docker Compose project name.",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, compose_file: str, project_name: str) -> None:
    """ONEX Infrastructure Integration Test CLI."""
    ctx.ensure_object(dict)
    ctx.obj["compose_file"] = compose_file
    ctx.obj["project_name"] = project_name


cli.add_command(env)
cli.add_command(introspect)
cli.add_command(verify)
cli.add_command(run_suite)


if __name__ == "__main__":
    cli()
