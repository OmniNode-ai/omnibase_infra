"""Scratch prototype: stdio MCP server with one tool, delegate, that shells out to onex delegate.

Customer rule: only the final answer or one error line reaches the agent. Receipts, run ids,
cost and diagnostics stay on disk (onex writes them under its state root).
"""

import re
import subprocess

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

mcp = MCPServer("omninode-delegate")

PUBLIC_TASK_TYPES = [
    "code_generation",
    "complex_reasoning",
    "document",
    "planning",
    "reasoning",
    "refactor",
    "research",
    "review",
    "summarization",
    "test",
]
TIMEOUT_S = 300  # task-class ceiling is 240s; leave room for queue and process start


def _err(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=_one_line(text))], isError=True
    )


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return re.sub(
        r"\s*\(receipt: [^)]*\)\s*$", "", line
    )  # no receipt path to the agent


@mcp.tool()
def delegate(
    prompt: str,
    task_type: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> CallToolResult:
    """Send a task to OmniNode delegation and return the final answer.

    task_type is optional; one of: code_generation, complex_reasoning, document, planning,
    reasoning, refactor, research, review, summarization, test. Omit to let routing decide.
    model optionally names the model your own provider key should run.
    """
    if task_type is not None and task_type not in PUBLIC_TASK_TYPES:
        return _err(f"Error: task_type must be one of: {', '.join(PUBLIC_TASK_TYPES)}")
    cmd = ["onex", "delegate", "-", "--result-only", "--source", "external-client"]
    if task_type:
        cmd += ["--task-type", task_type]
    if model:
        cmd += ["--model", model]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _err(f"Error: delegation did not finish within {TIMEOUT_S}s")
    except FileNotFoundError:
        return _err("Error: the onex command is not installed")
    if proc.returncode != 0:
        return _err(proc.stderr or f"Error: onex delegate exited {proc.returncode}")
    return CallToolResult(content=[TextContent(type="text", text=proc.stdout)])


if __name__ == "__main__":
    mcp.run("stdio")
