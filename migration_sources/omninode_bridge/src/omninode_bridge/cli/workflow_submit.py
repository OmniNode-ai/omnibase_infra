#!/usr/bin/env python3
"""
Event-Driven Workflow Submission CLI Tool

A reusable command-line interface for submitting workflows via Kafka
instead of REST endpoints. Provides a clean, event-driven way to
execute workflows through the message bus.

Usage:
    python -m omninode_bridge.cli.workflow_submit --workflow examples/test_workflow.json
    python -m omninode_bridge.cli.workflow_submit --quick-task "Analyze project dependencies"
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from ..constants import ComplexityLevels, Defaults, KafkaTopics
from ..models.workflow import WorkflowDefinition, WorkflowTask
from ..workflow.events import WorkflowExecutionEvent, WorkflowExecutionResponseEvent

logger = logging.getLogger(__name__)


class WorkflowSubmissionCLI:
    """Event-driven workflow submission CLI."""

    def __init__(self, kafka_bootstrap_servers: str = "localhost:29092"):
        """Initialize the CLI with Kafka configuration."""
        self.bootstrap_servers = kafka_bootstrap_servers
        self.producer: AIOKafkaProducer | None = None
        self.consumer: AIOKafkaConsumer | None = None
        self.correlation_id: str | None = None

    async def connect(self) -> None:
        """Connect to Kafka cluster."""
        try:
            # Initialize producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda v: str(v).encode("utf-8") if v else None,
                acks="all",
            )
            await self.producer.start()

            # Initialize consumer for responses
            self.consumer = AIOKafkaConsumer(
                KafkaTopics.WORKFLOW_EXECUTION_RESPONSES,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"workflow_cli_{uuid.uuid4().hex[:8]}",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
            )
            await self.consumer.start()

            print(f"✅ Connected to Kafka at {self.bootstrap_servers}")

        except Exception as e:
            print(f"❌ Failed to connect to Kafka: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        print("✅ Disconnected from Kafka")

    async def submit_workflow_from_file(
        self,
        workflow_file: Path,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        """Submit workflow from JSON file."""
        try:
            with open(workflow_file) as f:
                workflow_data = json.load(f)

            workflow_definition = WorkflowDefinition(**workflow_data)
            return await self._submit_workflow(workflow_definition, input_data or {})

        except FileNotFoundError:
            print(f"❌ Workflow file not found: {workflow_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in workflow file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading workflow: {e}")
            sys.exit(1)

    async def submit_quick_task(
        self,
        task_description: str,
        complexity: str = ComplexityLevels.MODERATE,
        ai_task_type: str = "analysis",
    ) -> str:
        """Submit a simple single-task workflow."""
        task = WorkflowTask(
            task_id=f"quick_task_{uuid.uuid4().hex[:8]}",
            task_type="ai_task",
            prompt=task_description,
            config={
                "complexity": complexity,
                "ai_task_type": ai_task_type,
                "timeout": Defaults.TASK_TIMEOUT,
                "retry_limit": Defaults.MAX_RETRIES,
            },
        )

        workflow_definition = WorkflowDefinition(
            workflow_id=f"quick_workflow_{uuid.uuid4().hex[:8]}",
            name=f"Quick Task: {task_description[:50]}...",
            description=f"Single-task workflow: {task_description}",
            tasks=[task],
            dependencies={},
        )

        return await self._submit_workflow(workflow_definition, {})

    async def _submit_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        input_data: dict[str, Any],
    ) -> str:
        """Submit workflow and return correlation ID."""
        self.correlation_id = str(uuid.uuid4())

        # Create workflow execution event
        event = WorkflowExecutionEvent(
            correlation_id=self.correlation_id,
            workflow_definition=workflow_definition,
            input_data=input_data,
        )

        try:
            # Publish to Kafka
            await self.producer.send(
                KafkaTopics.WORKFLOW_EXECUTION_REQUESTS,
                value=event.model_dump(),
                key=self.correlation_id,
            )

            print("🚀 Workflow submitted successfully!")
            print(f"   Workflow ID: {workflow_definition.workflow_id}")
            print(f"   Correlation ID: {self.correlation_id}")
            print(f"   Tasks: {len(workflow_definition.tasks)}")
            print(f"   Description: {workflow_definition.description}")

            return self.correlation_id

        except KafkaError as e:
            print(f"❌ Failed to submit workflow to Kafka: {e}")
            raise

    async def wait_for_response(self, timeout: int = 300) -> None:
        """Wait for workflow execution response."""
        if not self.correlation_id:
            print("❌ No correlation ID available. Submit a workflow first.")
            return

        print(f"⏳ Waiting for workflow response (timeout: {timeout}s)...")

        try:
            async for message in self.consumer:
                response_data = message.value

                # Check if this response matches our correlation ID
                if response_data.get("correlation_id") == self.correlation_id:
                    response = WorkflowExecutionResponseEvent(**response_data)

                    print("\n📋 Workflow Response Received:")
                    print(f"   Status: {response.status}")
                    print(f"   Workflow ID: {response.workflow_id}")

                    if response.status == "completed":
                        print("✅ Workflow completed successfully!")
                        if response.results:
                            print(
                                f"   Results: {len(response.results)} tasks completed",
                            )
                            for task_id, result in response.results.items():
                                success = result.get("success", False)
                                status_icon = "✅" if success else "❌"
                                print(
                                    f"   {status_icon} Task {task_id}: {result.get('summary', 'No summary')}",
                                )

                    elif response.status == "failed":
                        print("❌ Workflow failed!")
                        if response.error:
                            print(f"   Error: {response.error}")

                    elif response.status == "started":
                        print("🏃 Workflow execution started...")
                        continue  # Keep waiting for completion

                    if response.execution_metrics:
                        print(
                            f"   Duration: {response.execution_metrics.get('total_duration_ms', 0)}ms",
                        )
                        print(
                            f"   Tasks Executed: {response.execution_metrics.get('tasks_executed', 0)}",
                        )

                    # Only break for terminal statuses
                    if response.status in ["completed", "failed"]:
                        break

        except TimeoutError:
            print(f"⏰ Timeout waiting for workflow response after {timeout}s")
        except Exception as e:
            print(f"❌ Error waiting for response: {e}")


def create_example_workflow() -> None:
    """Create an example workflow file for reference."""
    example_workflow = {
        "workflow_id": "example_analysis_workflow",
        "name": "Project Analysis Workflow",
        "description": "Comprehensive project analysis with security and performance review",
        "tasks": [
            {
                "task_id": "dependency_analysis",
                "task_type": "ai_task",
                "prompt": "Analyze the project dependencies in package.json and identify any security vulnerabilities or outdated packages.",
                "config": {
                    "complexity": "moderate",
                    "ai_task_type": "security_analysis",
                    "timeout": 180,
                    "retry_limit": 2,
                },
                "validation_contract": {
                    "type": "required_fields",
                    "fields": [
                        "vulnerabilities",
                        "outdated_packages",
                        "recommendations",
                    ],
                },
                "definition_of_done": {
                    "min_content_length": 100,
                    "required_keywords": ["security", "analysis"],
                    "quality_threshold": 0.8,
                },
            },
            {
                "task_id": "code_quality_review",
                "task_type": "ai_task",
                "prompt": "Review the codebase for code quality issues, potential bugs, and adherence to best practices.",
                "config": {
                    "complexity": "moderate",
                    "ai_task_type": "code_review",
                    "timeout": 240,
                    "retry_limit": 2,
                },
                "validation_contract": {
                    "type": "required_fields",
                    "fields": ["quality_score", "issues_found", "recommendations"],
                },
            },
            {
                "task_id": "performance_optimization",
                "task_type": "ai_task",
                "prompt": "Identify performance bottlenecks and suggest optimization strategies for the application.",
                "config": {
                    "complexity": "complex",
                    "ai_task_type": "performance_analysis",
                    "timeout": 300,
                    "retry_limit": 3,
                },
                "model_fallback_config": {
                    "strategy": "next_tier",
                    "fallback_models": [
                        "llama3.1:8b-instruct-q6_k",
                        "mixtral:8x7b-instruct-v0.1-q4_K_M",
                    ],
                    "max_attempts": 2,
                },
            },
        ],
        "dependencies": {
            "code_quality_review": ["dependency_analysis"],
            "performance_optimization": ["dependency_analysis"],
        },
    }

    example_file = Path("examples/example_workflow.json")
    example_file.parent.mkdir(exist_ok=True)

    with open(example_file, "w") as f:
        json.dump(example_workflow, f, indent=2)

    print(f"📄 Example workflow created: {example_file}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Event-driven workflow submission CLI for OmniNode Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit workflow from file
  python -m omninode_bridge.cli.workflow_submit --workflow examples/test_workflow.json

  # Submit quick single task
  python -m omninode_bridge.cli.workflow_submit --quick-task "Analyze project security"

  # Submit with custom input data
  python -m omninode_bridge.cli.workflow_submit --workflow my_workflow.json --input '{"env": "production"}'

  # Create example workflow file
  python -m omninode_bridge.cli.workflow_submit --create-example
        """,
    )

    # Workflow submission options
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument(
        "--workflow",
        "-w",
        type=Path,
        help="Path to workflow JSON file",
    )
    workflow_group.add_argument(
        "--quick-task",
        "-q",
        type=str,
        help="Quick single-task workflow description",
    )
    workflow_group.add_argument(
        "--create-example",
        action="store_true",
        help="Create an example workflow file",
    )

    # Additional options
    parser.add_argument("--input", "-i", type=str, help="Input data as JSON string")
    parser.add_argument(
        "--complexity",
        "-c",
        choices=[
            ComplexityLevels.SIMPLE,
            ComplexityLevels.MODERATE,
            ComplexityLevels.COMPLEX,
            ComplexityLevels.CRITICAL,
        ],
        default=ComplexityLevels.MODERATE,
        help="Task complexity level (for quick tasks)",
    )
    parser.add_argument(
        "--kafka-servers",
        "-k",
        default="localhost:29092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout for workflow response (seconds)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit workflow and exit without waiting for response",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Handle create example
    if args.create_example:
        create_example_workflow()
        return

    # Parse input data
    input_data = {}
    if args.input:
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid input JSON: {e}")
            sys.exit(1)

    # Initialize CLI
    cli = WorkflowSubmissionCLI(args.kafka_servers)

    try:
        await cli.connect()

        # Submit workflow
        if args.workflow:
            correlation_id = await cli.submit_workflow_from_file(
                args.workflow,
                input_data,
            )
        elif args.quick_task:
            correlation_id = await cli.submit_quick_task(
                args.quick_task,
                args.complexity,
            )

        # Wait for response unless --no-wait
        if not args.no_wait:
            await cli.wait_for_response(args.timeout)
        else:
            print(
                f"🏃 Workflow submitted. Use correlation ID {correlation_id} to track progress.",
            )

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ CLI error: {e}")
        sys.exit(1)
    finally:
        await cli.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
