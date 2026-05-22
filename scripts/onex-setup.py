#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Interactive CLI for bootstrapping the OmniNode platform infrastructure.

Prompts for preset/custom selection, writes ~/.omnibase/topology.yaml, and
publishes a typed setup orchestration command to the ONEX event bus.

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
import json
import os
import sys
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOUD_COMING_SOON = (
    "\n\u26a0  Cloud mode: your preference is stored in topology.yaml, "
    "but cloud provisioning is not yet implemented (coming soon).\n"
)

SETUP_ORCHESTRATION_TOPIC: Final[str] = (
    "onex.cmd.omnibase-infra.setup-orchestration-start.v1"
)
"""Command topic consumed by the setup orchestration runtime."""


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
# Typed bus command publishing
# ---------------------------------------------------------------------------


def _build_setup_command(
    topology: object,
    compose_file_path: str,
    dry_run: bool,
    correlation_id: UUID | None = None,
) -> object:
    """Build the typed setup orchestrator command payload."""
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_input import (
        ModelSetupOrchestratorInput,
    )

    assert isinstance(topology, ModelDeploymentTopology)

    return ModelSetupOrchestratorInput(
        topology=topology,
        correlation_id=correlation_id or uuid4(),
        compose_file_path=compose_file_path,
        dry_run=dry_run,
    )


def _build_command_envelope_json(command: object) -> str:
    """Serialize the setup command as a typed ONEX event envelope."""
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_input import (
        ModelSetupOrchestratorInput,
    )

    assert isinstance(command, ModelSetupOrchestratorInput)
    envelope: ModelEventEnvelope[ModelSetupOrchestratorInput] = ModelEventEnvelope[
        ModelSetupOrchestratorInput
    ](
        payload=command,
        correlation_id=command.correlation_id,
        source_tool="onex-setup",
        target_tool="node_setup_orchestrator",
        event_type=SETUP_ORCHESTRATION_TOPIC,
        payload_type=type(command).__name__,
    )
    return envelope.model_dump_json()


async def _publish_setup_command(
    topology: object,
    compose_file_path: str,
    dry_run: bool,
) -> UUID:
    """Publish the setup orchestration command to Kafka.

    Returns:
        Correlation ID of the published setup command.
    """
    from aiokafka import AIOKafkaProducer

    from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_input import (
        ModelSetupOrchestratorInput,
    )

    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap_servers:
        raise RuntimeError(
            "KAFKA_BOOTSTRAP_SERVERS is not set. "
            "Set it to the Redpanda/Kafka bootstrap address before running setup."
        )

    command = _build_setup_command(topology, compose_file_path, dry_run)
    assert isinstance(command, ModelSetupOrchestratorInput)
    payload = _build_command_envelope_json(command).encode("utf-8")

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers, acks="all")
    await producer.start()
    try:
        await producer.send_and_wait(
            SETUP_ORCHESTRATION_TOPIC,
            payload,
            key=str(command.correlation_id).encode("utf-8"),
        )
    finally:
        await producer.stop()

    return command.correlation_id


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
    # Step 5: Publish setup command
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
        if args.dry_run:
            command = _build_setup_command(
                topology=topology,
                compose_file_path=compose_file,
                dry_run=True,
            )
            print("\n[dry-run] Setup command payload:")
            print(
                json.dumps(json.loads(_build_command_envelope_json(command)), indent=2)
            )
            print("[dry-run] Skipping Kafka publish.")
            return 0

        correlation_id = asyncio.run(
            _publish_setup_command(
                topology=topology,
                compose_file_path=compose_file,
                dry_run=False,
            )
        )
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 6: Exit code
    # ------------------------------------------------------------------
    print(f"Published setup command to {SETUP_ORCHESTRATION_TOPIC}")
    print(f"Correlation ID: {correlation_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
