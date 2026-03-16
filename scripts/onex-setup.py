#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Interactive CLI for bootstrapping the OmniNode platform infrastructure.

Prompts for preset/custom selection, writes ~/.omnibase/topology.yaml, and
invokes NodeSetupOrchestrator, printing events as they flow.

Invariants:
    I7 — resolve_compose_file() lives only here. Handlers receive an
         already-resolved string path.
    I8 — Cloud selection stores mode=CLOUD in topology.yaml and shows a
         coming-soon notice. Does NOT convert cloud to disabled.

Ticket: OMN-3496

Usage:
    uv run python scripts/onex-setup.py --preset minimal --dry-run
    uv run python scripts/onex-setup.py --preset standard --no-interactive
    uv run python scripts/onex-setup.py --topology-file ~/.omnibase/topology.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOUD_COMING_SOON = (
    "\n\u26a0  Cloud mode: your preference is stored in topology.yaml, "
    "but cloud provisioning is not yet implemented (coming soon).\n"
)

# Human-readable messages for each event type emitted by the orchestrator.
_EVENT_MESSAGES: dict[str, str] = {
    "setup.preflight.started": "Running preflight checks...",
    "setup.preflight.completed": "All checks passed",
    "setup.preflight.failed": "Preflight checks failed",
    "setup.provision.started": "Starting Docker services...",
    "setup.provision.completed": "Docker services started",
    "setup.provision.failed": "Docker provisioning failed",
    "setup.infisical.started": "Bootstrapping Infisical...",
    "setup.infisical.completed": "Infisical ready",
    "setup.infisical.skipped": "Infisical skipped (not in topology)",
    "setup.infisical.failed": "Infisical bootstrap failed",
    "setup.validate.started": "Validating provisioned services...",
    "setup.validate.completed": "All services healthy",
    "setup.validate.failed": "Service validation failed",
    "setup.completed": "Platform ready \u2713",
    "setup.cloud.unavailable": "Cloud provisioning not available",
    "setup.aborted": "Setup aborted",
}


# ---------------------------------------------------------------------------
# I7 — resolve_compose_file lives only here (not in handlers)
# ---------------------------------------------------------------------------


def resolve_compose_file(cli_arg: str | None) -> str:
    """Resolve the path to the Docker Compose infra file.

    Resolution order:
        1. ``cli_arg`` (explicit ``--compose-file`` argument)
        2. ``ONEX_COMPOSE_FILE`` environment variable
        3. Upward search from CWD for ``docker/docker-compose.infra.yml``

    Args:
        cli_arg: Value of the ``--compose-file`` CLI argument, or None.

    Returns:
        Resolved absolute path string.

    Raises:
        RuntimeError: If no compose file is found by any method.
    """
    if cli_arg:
        return cli_arg

    env_path = os.environ.get("ONEX_COMPOSE_FILE")
    if env_path:
        return env_path

    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "docker" / "docker-compose.infra.yml"
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "Cannot locate docker-compose.infra.yml. "
        "Set ONEX_COMPOSE_FILE or pass --compose-file."
    )


def _omnibase_dir() -> Path:
    """Return the ~/.omnibase directory path.

    Respects the ``OMNIBASE_DIR`` environment variable for testing isolation.
    """
    env = os.environ.get("OMNIBASE_DIR")
    return Path(env) if env else Path.home() / ".omnibase"


# ---------------------------------------------------------------------------
# Topology builder helpers
# ---------------------------------------------------------------------------


def _topology_for_preset(preset: str) -> object:
    """Return a ModelDeploymentTopology for the given preset name.

    Args:
        preset: One of ``minimal``, ``standard``, or ``full``.

    Returns:
        ModelDeploymentTopology instance.

    Raises:
        ValueError: If preset name is not recognised.
    """
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

    factories = {
        "minimal": ModelDeploymentTopology.default_minimal,
        "standard": ModelDeploymentTopology.default_standard,
        "full": ModelDeploymentTopology.default_full,
    }
    factory = factories.get(preset)
    if factory is None:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose one of: {', '.join(factories)}."
        )
    return factory()


def _print_topology_summary(topology: object) -> None:
    """Print a human-readable summary of the topology to stdout."""
    from omnibase_core.enums.enum_deployment_mode import EnumDeploymentMode
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

    assert isinstance(topology, ModelDeploymentTopology)

    preset_label = topology.active_preset or "custom"
    print(f"\nTopology preset: {preset_label}")
    print(f"{'Service':<20} {'Mode':<12}")
    print("-" * 34)
    for name, svc in sorted(topology.services.items()):
        mode_label = svc.mode.value
        print(f"{name:<20} {mode_label:<12}")

    cloud_services = [
        name
        for name, svc in topology.services.items()
        if svc.mode == EnumDeploymentMode.CLOUD
    ]
    if cloud_services:
        print(CLOUD_COMING_SOON)


# ---------------------------------------------------------------------------
# Event printing
# ---------------------------------------------------------------------------


def _print_event(event_type: str, payload: dict[str, object]) -> None:
    """Print a single setup event in the canonical CLI format.

    Format: ``[event_type]   message``
    """
    message = _EVENT_MESSAGES.get(event_type, "")
    if payload and event_type == "setup.cloud.unavailable":
        gated = payload.get("gated_services", [])
        message = f"Cloud provisioning not available: {', '.join(str(s) for s in gated)} (stored for future use)"
    elif payload and event_type == "setup.preflight.completed":
        # Optionally show check count if available in payload
        pass
    print(f"[{event_type}]  {message}")


# ---------------------------------------------------------------------------
# Orchestrator invocation (dry-run aware)
# ---------------------------------------------------------------------------


async def _run_orchestrator(
    topology: object,
    compose_file_path: str,
    dry_run: bool,
) -> bool:
    """Invoke HandlerSetupOrchestrator and print events as they flow.

    In dry-run mode, all steps are skipped and a ``setup.completed`` event
    is synthesised to indicate success.

    Args:
        topology: Validated ModelDeploymentTopology.
        compose_file_path: Resolved path to docker-compose.infra.yml.
        dry_run: If True, skip all real provisioning.

    Returns:
        True on success (``setup.completed`` received), False otherwise.
    """
    from omnibase_core.enums.enum_deployment_mode import EnumDeploymentMode
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

    assert isinstance(topology, ModelDeploymentTopology)

    if dry_run:
        print("\n[dry-run] Skipping provisioning — topology summary only.")
        _print_event("setup.completed", {})
        return True

    # Check for cloud-only gate before invoking the orchestrator.
    cloud_services = [
        name
        for name, svc in topology.services.items()
        if svc.mode == EnumDeploymentMode.CLOUD
    ]
    if cloud_services:
        _print_event("setup.cloud.unavailable", {"gated_services": cloud_services})
        return False

    # Import effect node implementations lazily to avoid import-time side-effects.
    # Build a minimal stub container (no services required for CLI invocation).
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_setup_infisical_effect.handlers.handler_infisical_full_setup import (
        HandlerInfisicalFullSetup,
    )
    from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision import (
        HandlerLocalProvision,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
        HandlerSetupOrchestrator,
    )
    from omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check import (
        HandlerPreflightCheck,
    )
    from omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate import (
        HandlerServiceValidate,
    )

    container = ModelONEXContainer()

    handler = HandlerSetupOrchestrator(
        container=container,
        preflight=HandlerPreflightCheck(container=container),
        provision=HandlerLocalProvision(container=container),
        infisical=HandlerInfisicalFullSetup(container=container),
        validate=HandlerServiceValidate(container=container),
    )
    await handler.initialize({})

    corr_id = uuid4()
    result = await handler.handle(topology, corr_id, compose_file_path)

    success = False
    for event in result.result.events:
        _print_event(event.event_type, dict(event.payload))
        if event.event_type == "setup.completed":
            success = True

    return success


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="onex-setup",
        description="Interactive CLI for bootstrapping the OmniNode platform infrastructure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=["minimal", "standard", "full"],
        help="Skip interactive prompt and use the specified preset.",
    )
    parser.add_argument(
        "--topology-file",
        metavar="PATH",
        help="Use an existing topology.yaml file instead of prompting.",
    )
    parser.add_argument(
        "--compose-file",
        metavar="PATH",
        help="Override the Docker Compose file path (I7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip file writes and Docker operations — topology summary only.",
    )
    parser.add_argument(
        "--skip-infisical",
        action="store_true",
        default=False,
        help="Skip the Infisical bootstrap step.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        default=False,
        help="Skip post-provision validation.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Use standard defaults; never prompt for input.",
    )
    return parser


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------


def _prompt_preset() -> str:
    """Interactively ask the user to choose a preset.

    Returns:
        One of ``minimal``, ``standard``, ``full``.
    """
    presets = ["minimal", "standard", "full"]
    print("\nChoose a topology preset:")
    for i, name in enumerate(presets, start=1):
        descriptions = {
            "minimal": "3 services — postgres, redpanda, valkey",
            "standard": "4 services — minimal + infisical (secrets)",
            "full": "5 services — standard + keycloak",
        }
        print(f"  {i}. {name:<12}  {descriptions[name]}")
    while True:
        raw = input("\nPreset [1-3, default=1]: ").strip()
        if not raw:
            return presets[0]
        try:
            idx = int(raw) - 1
        except ValueError:
            print("Please enter a number between 1 and 3.")
            continue
        if 0 <= idx < len(presets):
            return presets[idx]
        print("Please enter a number between 1 and 3.")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point.

    Returns:
        0 on success, 1 on failure or cloud gate.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Build topology
    # ------------------------------------------------------------------
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

    topology: ModelDeploymentTopology

    if args.topology_file:
        topology_path = Path(args.topology_file).expanduser()
        try:
            topology = ModelDeploymentTopology.from_yaml(topology_path)
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(f"Error loading topology file: {exc}", file=sys.stderr)
            return 1
    elif args.preset:
        try:
            topology = _topology_for_preset(args.preset)  # type: ignore[assignment]
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    elif args.no_interactive:
        topology = ModelDeploymentTopology.default_standard()  # type: ignore[assignment]
    else:
        preset = _prompt_preset()
        topology = _topology_for_preset(preset)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Step 2: Print topology summary
    # ------------------------------------------------------------------
    _print_topology_summary(topology)

    # ------------------------------------------------------------------
    # Step 3: Confirm (skip if --no-interactive or --dry-run)
    # ------------------------------------------------------------------
    if not args.no_interactive and not args.dry_run:
        try:
            answer = input("\nProceed with setup? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer and answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    # ------------------------------------------------------------------
    # Step 4: Write topology.yaml (skip if --dry-run)
    # ------------------------------------------------------------------
    if not args.dry_run:
        omnibase = _omnibase_dir()
        omnibase.mkdir(parents=True, exist_ok=True)
        topo_path = omnibase / "topology.yaml"
        try:
            topology.to_yaml(topo_path)
            print(f"\nTopology written to {topo_path}")
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(f"Error writing topology file: {exc}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Step 5: Invoke orchestrator and print events
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    try:
        compose_file = resolve_compose_file(args.compose_file)
    except RuntimeError as exc:
        if args.dry_run:
            compose_file = "docker/docker-compose.infra.yml"  # stub for dry-run
        else:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        success = asyncio.run(
            _run_orchestrator(
                topology=topology,
                compose_file_path=compose_file,
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 6: Exit code
    # ------------------------------------------------------------------
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
