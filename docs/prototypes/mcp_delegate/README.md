# MCP delegate tool (prototype, review only)

**This is a prototype for review. It is not for merging and it is not shipped code.** The
permanent version follows a design that is under review; the decisions that design must settle
are listed in the pull request description.

## What it does

`server.py` is a stdio MCP server with one tool:

```
delegate(prompt, task_type?, model?, max_tokens?)
```

It runs `onex delegate - --result-only --source external-client` with the prompt on stdin and
returns only the final answer. On any failure it returns one error line with `isError` set, and
the receipt path is stripped. Receipts, run ids and costs stay on disk under the onex state
root and never reach the agent.

`client.py` is a small SDK client that lists the tools, calls `delegate` over stdio, and prints
the run ids the call created.

## How to run

Needs `uv` and a working `onex` command on `PATH` that can reach a delegation route.

```
cd docs/prototypes/mcp_delegate
uv run python client.py "In two short sentences, explain why unit tests are useful." reasoning
```

To register the server with a host, point the host at
`uv run --project <this directory> python <this directory>/server.py`.

Use sentence-shaped prompts for smoke tests. The quality gate refuses bare one-word replies.

## Measured runs

Lab route kafka / dev / deployed-lane, SDK client over stdio, mcp SDK 2.3.0:

| Call | Result | Time | Run id |
|---|---|---|---|
| task_type reasoning, two-sentence prompt | success, answer text only | 50.1 s | 19e900ca-7428-45e7-be11-286759db593e |
| same prompt, while another delegation ran | success | 88.4 s | d1f13521-35af-4dfc-8fe1-8ee2b657db4f |

A bare one-word prompt was refused by the quality gate; the agent received `isError: true` and
one line starting `Error: returned_to_caller:`.

## Findings

- mcp 2.x renamed FastMCP to `MCPServer`; code written for 1.x does not import.
- In mcp 2.x a tool that raises gives the agent only a generic error. Return
  `CallToolResult(isError=True)` to deliver the one error line.
- Lab latency is 25 to 150 s with no progress signal, so a permanent version needs a stated
  ceiling and progress notifications.

## Status

Prototype. Not wired into the `onex` CLI, no unit tests, no registration file. The permanent
version (home, cloud variant, receipts, long runs, source stamp, mcp major) follows the design
under review.
