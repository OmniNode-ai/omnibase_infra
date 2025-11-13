"""
Generate command for ONEX node creation.

Provides CLI command and async function for generating ONEX nodes
via event-driven orchestration.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import click

from omninode_bridge.events.codegen import ModelEventNodeGenerationRequested

from ..client import CLIKafkaClient
from ..config import CodegenCLIConfig
from ..protocols import KafkaClientProtocol, ProgressDisplayProtocol
from ..ui import ProgressDisplay


@dataclass
class GenerationResult:
    """
    Result of node generation operation.

    Attributes:
        success: Whether generation succeeded
        workflow_id: Workflow identifier
        files: List of generated file paths
        duration_seconds: Total duration
        quality_score: Quality score (0.0-1.0)
        error: Error message if failed
    """

    success: bool
    workflow_id: str | None = None
    files: list[str] | None = None
    duration_seconds: float = 0.0
    quality_score: float = 0.0
    error: str | None = None


async def generate_node_async(
    prompt: str,
    output_dir: str,
    kafka_client: KafkaClientProtocol,
    progress_display: ProgressDisplayProtocol,
    node_type: str | None = None,
    interactive: bool = False,
    enable_intelligence: bool = True,
    enable_quorum: bool = False,
    timeout_seconds: int = 300,
) -> GenerationResult:
    """
    Generate an ONEX node via event-driven orchestration.

    This function is fully testable with dependency injection.

    Args:
        prompt: Natural language description of the node
        output_dir: Output directory for generated files
        kafka_client: Kafka client for event publishing/consumption
        progress_display: Progress display for tracking
        node_type: Optional node type hint (effect|orchestrator|reducer|compute)
        interactive: Enable interactive checkpoints
        enable_intelligence: Use RAG intelligence gathering
        enable_quorum: Use AI quorum validation
        timeout_seconds: Timeout for generation (default: 5 minutes)

    Returns:
        GenerationResult with generation outcome

    Raises:
        TimeoutError: If generation exceeds timeout
        RuntimeError: If generation fails
    """
    correlation_id = uuid4()

    # Ensure progress display uses the same correlation ID
    if hasattr(progress_display, "correlation_id"):
        progress_display.correlation_id = correlation_id

    try:
        # Start consuming events in background
        consumer_task = asyncio.create_task(
            kafka_client.consume_progress_events(
                correlation_id=correlation_id,
                callback=progress_display.on_event,
            )
        )

        print("\n🚀 Generating ONEX node...")
        print(f"   Correlation ID: {correlation_id}")
        print(f"   Prompt: {prompt}")
        print(f"   Output: {output_dir}\n")

        # Publish generation request event
        request_event = ModelEventNodeGenerationRequested(
            correlation_id=correlation_id,
            prompt=prompt,
            output_directory=output_dir,
            node_type=node_type,
            interactive_mode=interactive,
            enable_intelligence=enable_intelligence,
            enable_quorum=enable_quorum,
        )

        await kafka_client.publish_request(request_event)

        # Wait for completion with progress updates
        result = await progress_display.wait_for_completion(
            timeout_seconds=timeout_seconds
        )

        # Cancel consumer task
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        return GenerationResult(
            success=True,
            workflow_id=result.get("workflow_id"),
            files=result.get("generated_files", []),
            duration_seconds=result.get("total_duration_seconds", 0.0),
            quality_score=result.get("quality_score", 0.0),
        )

    except TimeoutError as e:
        return GenerationResult(
            success=False,
            error=str(e),
        )
    except RuntimeError as e:
        return GenerationResult(
            success=False,
            error=str(e),
        )


@click.command()
@click.argument("prompt", type=str)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for generated files (default: ./generated_nodes)",
)
@click.option(
    "--node-type",
    type=click.Choice(["effect", "orchestrator", "reducer", "compute"]),
    default=None,
    help="Node type hint (optional, will be inferred if not provided)",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Enable interactive checkpoints for validation",
)
@click.option(
    "--enable-intelligence",
    is_flag=True,
    default=True,
    help="Use RAG intelligence gathering (default: enabled)",
)
@click.option(
    "--disable-intelligence",
    "enable_intelligence",
    flag_value=False,
    help="Disable RAG intelligence gathering",
)
@click.option(
    "--enable-quorum",
    is_flag=True,
    help="Use AI quorum validation (multi-model consensus)",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Timeout in seconds (default: 300 = 5 minutes)",
)
@click.option(
    "--kafka-servers",
    type=str,
    default=None,
    help=(
        "Kafka bootstrap servers (default: KAFKA_BOOTSTRAP_SERVERS env "
        "or omninode-bridge-redpanda:9092 for remote infrastructure)"
    ),
)
def generate_command(
    prompt: str,
    output_dir: str | None,
    node_type: str | None,
    interactive: bool,
    enable_intelligence: bool,
    enable_quorum: bool,
    timeout: int | None,
    kafka_servers: str | None,
) -> int:
    """
    Generate ONEX nodes via event-driven orchestration.

    PROMPT: Natural language description of the node to generate

    Examples:
        omninode-generate "Create PostgreSQL CRUD Effect"
        omninode-generate "Create ML inference Orchestrator" --interactive
        omninode-generate "Create metrics Reducer" --enable-intelligence

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate inputs
    # 1. Prompt validation
    if not prompt or not prompt.strip():
        click.echo("Error: Prompt cannot be empty", err=True)
        return 1

    prompt = prompt.strip()
    MAX_PROMPT_LENGTH = 10000  # 10KB limit
    if len(prompt) > MAX_PROMPT_LENGTH:
        click.echo(
            f"Error: Prompt too long ({len(prompt)} chars). "
            f"Maximum {MAX_PROMPT_LENGTH} characters allowed.",
            err=True,
        )
        return 1

    # 2. Output directory path validation (prevent path traversal)
    if output_dir:
        output_path = Path(output_dir).resolve()
        try:
            # Check for path traversal attempts
            if (
                ".." in output_dir
                or output_dir.startswith("/etc")
                or output_dir.startswith("/sys")
            ):
                click.echo(
                    "Error: Invalid output directory path. "
                    "Path traversal patterns and system directories are not allowed.",
                    err=True,
                )
                return 1
        except (ValueError, OSError) as e:
            click.echo(f"Error: Invalid output directory: {e}", err=True)
            return 1

    # 3. Timeout validation
    if timeout is not None and (timeout < 1 or timeout > 3600):
        click.echo(
            f"Error: Timeout must be between 1 and 3600 seconds (got {timeout})",
            err=True,
        )
        return 1

    # Load configuration
    config = CodegenCLIConfig.from_env()

    # Apply command-line overrides
    if kafka_servers:
        config = config.with_overrides(kafka_bootstrap_servers=kafka_servers)
    if output_dir:
        config = config.with_overrides(default_output_dir=output_dir)
    if timeout:
        config = config.with_overrides(default_timeout_seconds=timeout)

    # Use config defaults for unspecified values
    output_dir = output_dir or config.default_output_dir
    timeout = timeout or config.default_timeout_seconds

    async def run() -> int:
        # Initialize Kafka client and progress display
        kafka_client = CLIKafkaClient(bootstrap_servers=config.kafka_bootstrap_servers)
        await kafka_client.connect()

        try:
            correlation_id = uuid4()
            progress_display = ProgressDisplay(correlation_id=correlation_id)

            result = await generate_node_async(
                prompt=prompt,
                output_dir=output_dir,
                kafka_client=kafka_client,
                progress_display=progress_display,
                node_type=node_type,
                interactive=interactive,
                enable_intelligence=enable_intelligence,
                enable_quorum=enable_quorum,
                timeout_seconds=timeout,
            )

            if result.success:
                # Print results
                print("\n✅ Generation complete!")
                print(f"   Duration: {result.duration_seconds:.1f}s")
                print(f"   Quality Score: {result.quality_score:.2f}")
                print(f"   Files Generated: {len(result.files or [])}")
                if result.files:
                    print("\n   Generated files:")
                    for file_path in result.files:
                        print(f"   - {file_path}")
                return 0
            else:
                print(f"\n❌ Generation failed: {result.error}", file=sys.stderr)
                return 1

        except KeyboardInterrupt:
            print("\n\n⚠️  Generation cancelled by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
            return 1
        finally:
            await kafka_client.disconnect()

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130
