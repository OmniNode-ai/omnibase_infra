# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``mcp.tools`` entry of a generated MCP wrapper contract (OMN-15920)."""

from __future__ import annotations

from typing import Self

from omnibase_core.models.contracts.model_contract_mcp_tool import ModelContractMcpTool
from omnibase_core.types import JsonType
from omnibase_infra.models.mcp.model_generated_compute_node_spec import (
    ModelGeneratedComputeNodeSpec,
)


class ModelMCPWrapperToolEntry(ModelContractMcpTool):
    """One tool entry for the wrapper orchestrator of a generated COMPUTE node."""

    @classmethod
    def from_spec(
        cls,
        spec: ModelGeneratedComputeNodeSpec,
        invoke_topic: str,
        result_topic: str,
    ) -> Self:
        """Build the entry from the spec and the validated invoke and result topics.

        The failed topic is the result topic's event name with ``-failed``; the
        core model refuses a suffix that is not canonical.
        """
        return cls(
            tool_name=spec.tool_name,
            description=spec.description,
            timeout_seconds=spec.tool_timeout_seconds,
            input_model=(
                f"{spec.invocation_input_model_module}"
                f".{spec.invocation_input_model_name}"
            ),
            output_model=(
                f"{spec.invocation_output_model_module}"
                f".{spec.invocation_output_model_name}"
            ),
            command_topic=invoke_topic,
            completed_topic=result_topic,
            failed_topic=result_topic.removesuffix("-result.v1") + "-failed.v1",
        )

    @classmethod
    def mcp_block(
        cls,
        spec: ModelGeneratedComputeNodeSpec,
        invoke_topic: str,
        result_topic: str,
    ) -> dict[str, JsonType]:
        """Return the wrapper contract ``mcp`` block: expose plus its one entry."""
        entry = cls.from_spec(spec, invoke_topic, result_topic)
        return {"expose": True, "tools": [entry.model_dump(mode="json")]}


__all__ = ["ModelMCPWrapperToolEntry"]
