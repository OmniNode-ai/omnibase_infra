import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def say(*parts: object) -> None:
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


HERE = Path(__file__).parent
STATE = HERE / "state"
PROMPT = sys.argv[1]
TT = sys.argv[2] if len(sys.argv) > 2 else None


def runs():
    return (
        {p.name for p in (STATE / "runs").glob("*")}
        if (STATE / "runs").exists()
        else set()
    )


async def main():
    before = runs()
    env = {**os.environ, "ONEX_STATE_DIR": str(STATE)}
    params = StdioServerParameters(
        command="uv",
        args=["run", "--project", str(HERE), "python", str(HERE / "server.py")],
        env=env,
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            for t in tools.tools:
                say(
                    "TOOL",
                    t.name,
                    json.dumps(
                        t.input_schema
                        if hasattr(t, "input_schema")
                        else getattr(t, "inputSchema", None)
                    )[:300],
                )
            args = {"prompt": PROMPT}
            if TT:
                args["task_type"] = TT
            t0 = time.time()
            res = await s.call_tool("delegate", args)
            dt = time.time() - t0
            say("CALL_SECONDS", round(dt, 1))
            say("IS_ERROR", getattr(res, "is_error", getattr(res, "isError", None)))
            for c in res.content:
                say("CONTENT", getattr(c, "text", c))
    new = sorted(runs() - before)
    for rid in new:
        rec = json.loads((STATE / "runs" / rid / "receipt.json").read_text())
        say(
            "RUN_ID",
            rid,
            "correlation_id",
            rec.get("correlation_id"),
            "status",
            rec.get("status"),
        )


asyncio.run(main())
