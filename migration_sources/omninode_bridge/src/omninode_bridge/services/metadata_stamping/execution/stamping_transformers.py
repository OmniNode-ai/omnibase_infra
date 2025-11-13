"""
Metadata stamping transformers for O.N.E. v0.1 protocol.

This module provides transformer implementations for metadata stamping
and validation operations using the schema-first execution pattern.
"""

import hashlib
import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from .transformer import ExecutionContext, transformer

logger = logging.getLogger(__name__)


# Input/Output schemas for metadata stamping
class StampingInput(BaseModel):
    """Input schema for metadata stamping."""

    content: str = Field(..., description="Content to stamp")
    file_path: Optional[str] = Field(None, description="Optional file path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    stamp_type: str = Field("lightweight", description="Type of stamp")
    namespace: str = Field("omninode.services.metadata", description="Namespace")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class StampingOutput(BaseModel):
    """Output schema for metadata stamping."""

    success: bool = Field(..., description="Operation success")
    hash: str = Field(..., description="Generated hash")
    stamped_content: str = Field(..., description="Content with stamp")
    stamp: str = Field(..., description="Stamp value")
    stamp_type: str = Field(..., description="Type of stamp applied")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    stamp_metadata: dict[str, Any] = Field(..., description="Stamp metadata")
    namespace: str = Field(..., description="Namespace")
    op_id: str = Field(..., description="Operation ID")


class ValidationInput(BaseModel):
    """Input schema for stamp validation."""

    content: str = Field(..., description="Content to validate")
    expected_hash: Optional[str] = Field(None, description="Expected hash")
    namespace: Optional[str] = Field(None, description="Namespace filter")


class ValidationOutput(BaseModel):
    """Output schema for stamp validation."""

    valid: bool = Field(..., description="Validation result")
    found_stamps: int = Field(..., description="Number of stamps found")
    stamps: list[str] = Field(default_factory=list, description="Found stamp hashes")
    validation_details: dict[str, Any] = Field(..., description="Validation details")


class BatchStampingInput(BaseModel):
    """Input schema for batch stamping."""

    items: list[StampingInput] = Field(..., description="Items to stamp")
    parallel: bool = Field(True, description="Process in parallel")


class BatchStampingOutput(BaseModel):
    """Output schema for batch stamping."""

    success: bool = Field(..., description="Overall success")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully stamped items")
    failed_items: int = Field(..., description="Failed items")
    results: list[StampingOutput] = Field(..., description="Individual results")
    total_execution_time_ms: float = Field(..., description="Total execution time")


# Stamping transformer implementation
@transformer(StampingInput, StampingOutput, "metadata_stamper", "1.0.0")
async def metadata_stamping_transformer(
    input_data: StampingInput, context: ExecutionContext
) -> dict[str, Any]:
    """
    Transform content into stamped content with metadata.

    Args:
        input_data: Stamping input
        context: Execution context

    Returns:
        dict: Stamping output
    """
    start_time = time.perf_counter()

    try:
        # Import blake3 if available, fallback to hashlib
        try:
            import blake3

            hasher = blake3.blake3()
            hasher.update(input_data.content.encode("utf-8"))
            content_hash = hasher.hexdigest()
        except ImportError:
            # Fallback to SHA256
            content_hash = hashlib.sha256(
                input_data.content.encode("utf-8")
            ).hexdigest()

        # Create stamp metadata
        stamp_data = {
            "hash": content_hash,
            "file_path": input_data.file_path,
            "file_size": input_data.file_size or len(input_data.content),
            "stamp_type": input_data.stamp_type,
            "namespace": input_data.namespace,
            "timestamp": time.time(),
            "execution_id": context.execution_id,
            "op_id": context.execution_id,  # Use execution_id as op_id
        }

        # Add custom metadata
        stamp_data.update(input_data.metadata)

        # Create stamp based on type
        if input_data.stamp_type == "lightweight":
            stamp = f"<!-- METADATA_STAMP: {content_hash} -->"
            stamp_footer = f"\n<!-- NAMESPACE: {input_data.namespace} -->"
        elif input_data.stamp_type == "detailed":
            import json

            stamp_json = json.dumps(stamp_data, separators=(",", ":"))
            stamp = f"<!-- METADATA_STAMP_START -->\n<!-- {stamp_json} -->\n<!-- METADATA_STAMP_END -->"
            stamp_footer = ""
        else:
            stamp = f"[STAMP:{content_hash}]"
            stamp_footer = f"[NS:{input_data.namespace}]"

        # Create stamped content
        stamped_content = f"{stamp}\n{input_data.content}{stamp_footer}"

        execution_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Content stamped successfully: hash={content_hash[:16]}..., "
            f"time={execution_time:.2f}ms"
        )

        return {
            "success": True,
            "hash": content_hash,
            "stamped_content": stamped_content,
            "stamp": stamp,
            "stamp_type": input_data.stamp_type,
            "execution_time_ms": execution_time,
            "stamp_metadata": stamp_data,
            "namespace": input_data.namespace,
            "op_id": context.execution_id,
        }

    except Exception as e:
        logger.error(f"Stamping failed: {e}")
        execution_time = (time.perf_counter() - start_time) * 1000

        return {
            "success": False,
            "hash": "",
            "stamped_content": input_data.content,
            "stamp": "",
            "stamp_type": input_data.stamp_type,
            "execution_time_ms": execution_time,
            "stamp_metadata": {"error": str(e)},
            "namespace": input_data.namespace,
            "op_id": context.execution_id,
        }


# Validation transformer implementation
@transformer(ValidationInput, ValidationOutput, "stamp_validator", "1.0.0")
async def stamp_validation_transformer(
    input_data: ValidationInput, context: ExecutionContext
) -> dict[str, Any]:
    """
    Transform content validation request into validation result.

    Args:
        input_data: Validation input
        context: Execution context

    Returns:
        dict: Validation output
    """
    try:
        # Find stamps in content using various patterns
        patterns = [
            r"<!-- METADATA_STAMP: ([a-fA-F0-9]+) -->",
            r"\[STAMP:([a-fA-F0-9]+)\]",
            r'<!-- "hash": "([a-fA-F0-9]+)".*? -->',
        ]

        found_stamps = []
        for pattern in patterns:
            matches = re.findall(pattern, input_data.content)
            found_stamps.extend(matches)

        # Remove duplicates
        found_stamps = list(set(found_stamps))

        # Check namespace if provided
        namespace_valid = True
        if input_data.namespace:
            namespace_pattern = (
                rf"<!-- NAMESPACE: {re.escape(input_data.namespace)} -->"
            )
            namespace_pattern2 = rf"\[NS:{re.escape(input_data.namespace)}\]"

            namespace_valid = (
                re.search(namespace_pattern, input_data.content) is not None
                or re.search(namespace_pattern2, input_data.content) is not None
            )

        # Validate against expected hash if provided
        hash_matches = False
        if input_data.expected_hash and found_stamps:
            hash_matches = input_data.expected_hash in found_stamps

        # Determine overall validity
        valid = len(found_stamps) > 0 and namespace_valid
        if input_data.expected_hash:
            valid = valid and hash_matches

        validation_result = {
            "valid": valid,
            "found_stamps": len(found_stamps),
            "stamps": found_stamps,
            "validation_details": {
                "stamps_found": found_stamps,
                "expected_hash": input_data.expected_hash,
                "hash_matches": hash_matches,
                "namespace": input_data.namespace,
                "namespace_valid": namespace_valid,
                "execution_id": context.execution_id,
            },
        }

        logger.info(
            f"Validation completed: valid={valid}, " f"stamps_found={len(found_stamps)}"
        )

        return validation_result

    except Exception as e:
        logger.error(f"Validation failed: {e}")

        return {
            "valid": False,
            "found_stamps": 0,
            "stamps": [],
            "validation_details": {
                "error": str(e),
                "execution_id": context.execution_id,
            },
        }


# Batch stamping transformer
async def batch_stamping_transformer(
    input_data: BatchStampingInput, context: ExecutionContext
) -> dict[str, Any]:
    """
    Transform batch stamping request into batch results.

    Args:
        input_data: Batch stamping input
        context: Execution context

    Returns:
        dict: Batch stamping output
    """
    import asyncio

    from .transformer import get_transformer

    start_time = time.perf_counter()
    results = []
    successful = 0
    failed = 0

    try:
        # Get the metadata stamper
        stamper = get_transformer("metadata_stamper")

        if not stamper:
            raise ValueError("Metadata stamper not found")

        # Process items
        if input_data.parallel:
            # Process in parallel
            tasks = []
            for item in input_data.items:
                item_context = ExecutionContext(
                    execution_id=f"{context.execution_id}-{len(tasks)}",
                    input_schema="StampingInput",
                    output_schema="StampingOutput",
                    simulation_mode=context.simulation_mode,
                )
                task = stamper.execute_with_validation(item, item_context)
                tasks.append(task)

            # Wait for all tasks
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in task_results:
                if isinstance(result, Exception):
                    failed += 1
                    results.append(
                        {
                            "success": False,
                            "hash": "",
                            "stamped_content": "",
                            "stamp": "",
                            "stamp_type": "",
                            "execution_time_ms": 0.0,
                            "stamp_metadata": {"error": str(result)},
                            "namespace": "",
                            "op_id": "",
                        }
                    )
                else:
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                    results.append(result.model_dump())

        else:
            # Process sequentially
            for i, item in enumerate(input_data.items):
                item_context = ExecutionContext(
                    execution_id=f"{context.execution_id}-{i}",
                    input_schema="StampingInput",
                    output_schema="StampingOutput",
                    simulation_mode=context.simulation_mode,
                )

                try:
                    result = await stamper.execute_with_validation(item, item_context)
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                    results.append(result.model_dump())

                except Exception as e:
                    failed += 1
                    results.append(
                        {
                            "success": False,
                            "hash": "",
                            "stamped_content": "",
                            "stamp": "",
                            "stamp_type": "",
                            "execution_time_ms": 0.0,
                            "stamp_metadata": {"error": str(e)},
                            "namespace": "",
                            "op_id": "",
                        }
                    )

        total_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Batch stamping completed: total={len(input_data.items)}, "
            f"successful={successful}, failed={failed}, "
            f"time={total_time:.2f}ms"
        )

        return {
            "success": failed == 0,
            "total_items": len(input_data.items),
            "successful_items": successful,
            "failed_items": failed,
            "results": results,
            "total_execution_time_ms": total_time,
        }

    except Exception as e:
        logger.error(f"Batch stamping failed: {e}")
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "success": False,
            "total_items": len(input_data.items),
            "successful_items": 0,
            "failed_items": len(input_data.items),
            "results": [],
            "total_execution_time_ms": total_time,
        }
