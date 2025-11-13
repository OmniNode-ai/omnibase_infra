"""
ONEX Node Generation CLI.

Provides command-line interface and library functions for generating
ONEX nodes via event-driven orchestration.

Key Components:
- main: Entry point for CLI execution
- commands: CLI command implementations
- client: Kafka client for event publishing/consumption
- ui: Progress display and user interaction
- config: Configuration management
- protocols: Protocol definitions for testability

Usage:
    # As CLI
    omninode-generate "Create PostgreSQL CRUD Effect"

    # As library
    from omninode_bridge.cli.codegen import generate_node_async
    result = await generate_node_async(...)
"""

from .commands import generate_command, generate_node_async
from .config import CodegenCLIConfig
from .main import main
from .protocols import KafkaClientProtocol, ProgressDisplayProtocol

__all__ = [
    "main",
    "generate_command",
    "generate_node_async",
    "CodegenCLIConfig",
    "KafkaClientProtocol",
    "ProgressDisplayProtocol",
]

__version__ = "0.1.0"
