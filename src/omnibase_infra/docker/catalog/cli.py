# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: T201
"""CLI for catalog-driven infrastructure management.

Usage:
    python -m omnibase_infra.docker.catalog.cli generate core [--output path]
    python -m omnibase_infra.docker.catalog.cli validate core
    python -m omnibase_infra.docker.catalog.cli up runtime
    python -m omnibase_infra.docker.catalog.cli up runtime --seed
    python -m omnibase_infra.docker.catalog.cli up runtime --build
    python -m omnibase_infra.docker.catalog.cli down
    python -m omnibase_infra.docker.catalog.cli status
    python -m omnibase_infra.docker.catalog.cli seed
    python -m omnibase_infra.docker.catalog.cli read-stack
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.docker.catalog.validator import validate_env

_OMNIBASE_ENV = Path.home() / ".omnibase" / ".env"


def _load_omnibase_env() -> None:
    """Load ~/.omnibase/.env into os.environ if it exists.

    Only sets vars not already present in the environment (existing values
    take precedence). This makes the CLI self-contained without requiring
    the caller to manually ``source ~/.omnibase/.env``.
    """
    if not _OMNIBASE_ENV.exists():
        return
    with open(_OMNIBASE_ENV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CATALOG_DIR = str(_REPO_ROOT / "docker" / "catalog")
_DEFAULT_OUTPUT = str(_REPO_ROOT / "docker" / "docker-compose.generated.yml")
_STACK_FILE = str(_REPO_ROOT / ".onex" / "stack.yml")
_SEED_SCRIPT = str(_REPO_ROOT / "scripts" / "seed-infisical.py")
_CONTRACTS_DIR = str(_REPO_ROOT / "src" / "omnibase_infra" / "nodes")


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
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return ["core"]
    bundles = data.get("bundles", ["core"])
    return list(bundles) if isinstance(bundles, list) else ["core"]


def cmd_generate(args: list[str]) -> int:
    """Generate compose from selected bundles."""
    output = _DEFAULT_OUTPUT
    bundles = []
    i = 0
    while i < len(args):
        if args[i] == "--output":
            if i + 1 >= len(args):
                print("Missing value for --output", file=sys.stderr)
                return 1
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
    _load_omnibase_env()
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


def _run_seed() -> int:
    """Run Infisical seed if INFISICAL_ADDR is set.

    Returns 0 on success or if Infisical is not configured (opt-in).
    Returns non-zero on seed failure.
    """
    infisical_addr = os.environ.get("INFISICAL_ADDR", "")
    if not infisical_addr:
        print("Skipping Infisical seed (INFISICAL_ADDR not set)")
        return 0

    seed_path = Path(_SEED_SCRIPT)
    if not seed_path.exists():
        print(f"Seed script not found: {_SEED_SCRIPT}", file=sys.stderr)
        return 1

    print("Seeding Infisical with contract config keys...")
    proc = subprocess.run(
        [
            sys.executable,
            _SEED_SCRIPT,
            "--contracts-dir",
            _CONTRACTS_DIR,
            "--create-missing-keys",
            "--execute",
        ],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        print("WARNING: Infisical seed failed (non-fatal)", file=sys.stderr)
    return 0


def _current_git_sha() -> str:
    """Return the current git SHA for the omnibase_infra repo (OMN-9330)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cmd_seed(_args: list[str]) -> int:
    """Seed Infisical with config keys from contracts."""
    return _run_seed()


def cmd_up(args: list[str]) -> int:
    """Validate, generate, and start compose stack."""
    _load_omnibase_env()
    # Parse flags
    run_seed = "--seed" in args
    force_build = "--build" in args
    flag_args = {"--seed", "--build"}
    bundles = [a for a in args if a not in flag_args] if args else _load_stack()

    # Save stack selection (exclude flags from saved bundles)
    if bundles:
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

    # Pre-cleanup: remove dead/exited containers to prevent restart delays (OMN-5468)
    # and name collisions when core infra is already running (OMN-5469).
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            _DEFAULT_OUTPUT,
            "rm",
            "-f",
            "--stop",
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
    )

    # Build if requested (OMN-7214)
    if force_build:
        print("Rebuilding images (--build)...")
        git_sha = _current_git_sha()
        build_proc = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                _DEFAULT_OUTPUT,
                "build",
                "--build-arg",
                f"GIT_SHA={git_sha}",
            ],
            cwd=str(_REPO_ROOT),
            check=False,
        )
        if build_proc.returncode != 0:
            print("Image build failed", file=sys.stderr)
            return build_proc.returncode

    # Start
    proc = subprocess.run(
        ["docker", "compose", "-f", _DEFAULT_OUTPUT, "up", "-d"],
        cwd=str(_REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        return proc.returncode

    # OMN-5831: Seed Infisical after stack is up when --seed flag is passed
    # or when runtime bundle is included. This ensures runtime services can
    # prefetch config from Infisical on startup.
    has_runtime = "runtime" in bundles
    if run_seed or has_runtime:
        _run_seed()

    return 0


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
        print("Commands: generate, validate, up, down, status, seed, read-stack")
        sys.exit(1)

    command = args[0]
    rest = args[1:]

    commands = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "up": cmd_up,
        "down": cmd_down,
        "status": cmd_status,
        "seed": cmd_seed,
        "read-stack": cmd_read_stack,
    }

    if command not in commands:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)

    sys.exit(commands[command](rest))


if __name__ == "__main__":
    main()
