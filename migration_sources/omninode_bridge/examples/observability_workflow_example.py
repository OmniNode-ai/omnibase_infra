"""
Example: Using Observability Features in Workflows

This example demonstrates how to integrate logging, tracing, and correlation
tracking into a typical workflow using the observability infrastructure.
"""

import asyncio
import logging
from uuid import uuid4

from omninode_bridge.events.versioning import EventSchemaVersion, get_topic_name
from omninode_bridge.observability.logging_config import (
    add_extra_context,
    clear_extra_context,
    configure_logging,
    correlation_context,
    get_logger,
)
from omninode_bridge.observability.tracing import (
    add_span_attributes,
    add_span_event,
    get_tracer,
    set_span_error,
    set_span_success,
    trace_async,
)

# Configure structured logging
configure_logging(level=logging.INFO, use_json=False)
logger = get_logger(__name__)
tracer = get_tracer(__name__)


@trace_async(span_name="validate_input", add_correlation=True)
async def validate_input(data: dict) -> bool:
    """Validate input data with tracing and logging."""
    logger.info("Validating input data", extra={"data_keys": list(data.keys())})

    add_span_attributes(
        validation_type="schema",
        field_count=len(data),
    )

    # Simulate validation
    await asyncio.sleep(0.1)

    if "required_field" not in data:
        logger.error("Validation failed: missing required_field")
        add_span_event("validation_failed", {"reason": "missing_required_field"})
        return False

    add_span_event("validation_passed")
    logger.info("Input validation successful")
    return True


@trace_async(span_name="process_data", add_correlation=True)
async def process_data(data: dict) -> dict:
    """Process data with observability."""
    logger.info("Processing data")

    add_span_attributes(
        processing_stage="transformation",
        data_size=len(str(data)),
    )

    # Add extra context for this processing step
    add_extra_context(
        batch_size=1,
        processing_type="standard",
    )

    try:
        # Simulate processing
        await asyncio.sleep(0.2)

        result = {
            "processed": True,
            "original_data": data,
            "timestamp": asyncio.get_event_loop().time(),
        }

        add_span_event("processing_completed", {"result_keys": list(result.keys())})
        logger.info("Data processing completed successfully")
        set_span_success()

        return result

    except Exception as e:
        logger.error("Processing failed", exc_info=True)
        set_span_error(e)
        raise
    finally:
        clear_extra_context()


@trace_async(span_name="publish_event", add_correlation=True)
async def publish_event(event_data: dict, version: EventSchemaVersion) -> str:
    """Publish event to Kafka topic with versioning."""
    topic = get_topic_name("example-event", version)

    logger.info("Publishing event", extra={"topic": topic, "version": version.value})

    add_span_attributes(
        topic_name=topic,
        event_version=version.value,
        event_type="EXAMPLE_EVENT",
    )

    # Simulate publishing
    await asyncio.sleep(0.05)

    add_span_event("event_published", {"topic": topic})
    logger.info("Event published successfully", extra={"topic": topic})

    return topic


async def execute_workflow(workflow_id: str, input_data: dict):
    """
    Execute a complete workflow with full observability.

    This function demonstrates:
    - Correlation context management
    - Distributed tracing
    - Structured logging
    - Event versioning
    """
    correlation_id = uuid4()

    # Establish correlation context for entire workflow
    async with correlation_context(
        correlation_id=correlation_id,
        workflow_id=uuid4(),
        stage_name="initialization",
    ):
        logger.info(
            "Starting workflow",
            extra={
                "workflow_id": workflow_id,
                "input_keys": list(input_data.keys()),
            },
        )

        # Create workflow span
        with tracer.start_as_current_span("workflow_execution") as span:
            span.set_attribute("workflow.id", workflow_id)
            span.set_attribute("workflow.type", "example")

            try:
                # Stage 1: Validation
                add_span_event("stage_started", {"stage": "validation"})

                is_valid = await validate_input(input_data)

                if not is_valid:
                    logger.warning("Workflow terminated: validation failed")
                    span.set_attribute("workflow.status", "failed")
                    return {"success": False, "reason": "validation_failed"}

                # Stage 2: Processing
                add_span_event("stage_started", {"stage": "processing"})

                processed_data = await process_data(input_data)

                # Stage 3: Publishing
                add_span_event("stage_started", {"stage": "publishing"})

                topic = await publish_event(processed_data, EventSchemaVersion.V1)

                # Workflow completed successfully
                add_span_event("workflow_completed", {"topic": topic})
                span.set_attribute("workflow.status", "completed")
                set_span_success()

                logger.info(
                    "Workflow completed successfully",
                    extra={
                        "workflow_id": workflow_id,
                        "published_topic": topic,
                    },
                )

                return {
                    "success": True,
                    "correlation_id": str(correlation_id),
                    "topic": topic,
                }

            except Exception as e:
                logger.error("Workflow failed with error", exc_info=True)
                span.set_attribute("workflow.status", "error")
                set_span_error(e)
                raise


async def main():
    """Main execution function."""
    print("=" * 80)
    print("Observability Workflow Example")
    print("=" * 80)

    # Example 1: Successful workflow
    print("\n1. Executing successful workflow...")
    result1 = await execute_workflow(
        workflow_id="workflow-001",
        input_data={
            "required_field": "value",
            "optional_field": "data",
        },
    )
    print(f"Result: {result1}")

    # Example 2: Failed validation
    print("\n2. Executing workflow with validation failure...")
    result2 = await execute_workflow(
        workflow_id="workflow-002",
        input_data={
            "optional_field": "data",
            # Missing required_field
        },
    )
    print(f"Result: {result2}")

    # Example 3: Multiple parallel workflows
    print("\n3. Executing multiple parallel workflows...")
    workflows = [
        execute_workflow(
            workflow_id=f"workflow-{i:03d}",
            input_data={"required_field": f"value-{i}"},
        )
        for i in range(3, 6)
    ]

    results = await asyncio.gather(*workflows, return_exceptions=True)
    for i, result in enumerate(results, start=3):
        print(f"Workflow {i}: {result}")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
