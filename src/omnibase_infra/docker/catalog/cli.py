# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: T201
"""CLI for catalog-driven infrastructure management.

Usage:
    python -m omnibase_infra.docker.catalog.cli generate core [--output path]
    python -m omnibase_infra.docker.catalog.cli validate core
    python -m omnibase_infra.docker.catalog.cli up runtime
    python -m omnibase_infra.docker.catalog.cli down
    python -m omnibase_infra.docker.catalog.cli status
    python -m omnibase_infra.docker.catalog.cli read-stack
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.docker.catalog.validator import validate_env

# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CATALOG_DIR = str(_REPO_ROOT / "docker" / "catalog")
_DEFAULT_OUTPUT = str(_REPO_ROOT / "docker" / "docker-compose.generated.yml")
_STACK_FILE = str(_REPO_ROOT / ".onex" / "stack.yml")


def _resolve_and_generate(bundles: list[str], output: str) -> int:
    """Resolve bundles, generate compose, write to output path."""
    resolver = CatalogResolver(catalog_dir=_CATALOG_DIR)
    resolved = resolver.resolve(bundles=bundles)
    compose = generate_compose(resolved)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Generated compose with {len(resolved.manifests)} entries -> {output}")
    return 0


def _save_stack(bundles: list[str]) -> None:
    """Persist selected bundles to .onex/stack.yml."""
    stack_path = Path(_STACK_FILE)
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stack_path, "w") as f:
        yaml.dump({"bundles": bundles}, f, default_flow_style=False)


def _load_stack() -> list[str]:
    """Load bundles from .onex/stack.yml."""
    stack_path = Path(_STACK_FILE)
    if not stack_path.exists():
        return ["core"]
    with open(stack_path) as f:
        data = yaml.safe_load(f)
    return list(data.get("bundles", ["core"]))


def cmd_generate(args: list[str]) -> int:
    """Generate compose from selected bundles."""
    output = _DEFAULT_OUTPUT
    bundles = []
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            bundles.append(args[i])
            i += 1
    if not bundles:
        bundles = _load_stack()
    return _resolve_and_generate(bundles, output)


def cmd_validate(args: list[str]) -> int:
    """Validate env vars for selected bundles."""
    bundles = args if args else _load_stack()
    resolver = CatalogResolver(catalog_dir=_CATALOG_DIR)
    resolved = resolver.resolve(bundles=bundles)
    result = validate_env(resolved.required_env)
    if result.ok:
        print("All required env vars are set.")
        return 0
    print("Missing required env vars:", file=sys.stderr)
    for var in result.missing:
        print(f"  - {var}", file=sys.stderr)
    return 1


def cmd_up(args: list[str]) -> int:
    """Validate, generate, and start compose stack."""
    bundles = args if args else _load_stack()

    # Save stack selection
    if args:
        _save_stack(bundles)

    # Validate
    resolver = CatalogResolver(catalog_dir=_CATALOG_DIR)
    resolved = resolver.resolve(bundles=bundles)
    result = validate_env(resolved.required_env)
    if not result.ok:
        print("Cannot start: missing required env vars:", file=sys.stderr)
        for var in result.missing:
            print(f"  - {var}", file=sys.stderr)
        return 1

    # Generate
    rc = _resolve_and_generate(bundles, _DEFAULT_OUTPUT)
    if rc != 0:
        return rc

    # Start
    proc = subprocess.run(
        ["docker", "compose", "-f", _DEFAULT_OUTPUT, "up", "-d"],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    return proc.returncode


def cmd_down(_args: list[str]) -> int:
    """Stop compose stack."""
    proc = subprocess.run(
        ["docker", "compose", "-f", _DEFAULT_OUTPUT, "down"],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    return proc.returncode


def cmd_status(_args: list[str]) -> int:
    """Show compose stack status."""
    proc = subprocess.run(
        ["docker", "compose", "-f", _DEFAULT_OUTPUT, "ps"],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    return proc.returncode


def cmd_read_stack(_args: list[str]) -> int:
    """Print current stack selection."""
    bundles = _load_stack()
    print(" ".join(bundles))
    return 0


def main() -> None:
    """Entry point for the catalog CLI."""
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m omnibase_infra.docker.catalog.cli <command> [args]")
        print("Commands: generate, validate, up, down, status, read-stack")
        sys.exit(1)

    command = args[0]
    rest = args[1:]

    commands = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "up": cmd_up,
        "down": cmd_down,
        "status": cmd_status,
        "read-stack": cmd_read_stack,
    }

    if command not in commands:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)

    sys.exit(commands[command](rest))


if __name__ == "__main__":
    main()
