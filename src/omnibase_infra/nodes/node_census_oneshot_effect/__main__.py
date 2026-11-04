# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CLI adapter dispatching the collector follow-up through RuntimeLocal's bus."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.handlers.handler_census_oneshot import load_policy
from omnibase_infra.models.model_census_oneshot_request import ModelCensusOneshotRequest
from omnibase_infra.models.model_census_oneshot_result import ModelCensusOneshotResult


def main() -> int:
    contract_path = Path(sys.argv[2])
    policy = load_policy(contract_path)
    request = ModelCensusOneshotRequest(
        script_directory=sys.argv[1], policy=policy, arguments=tuple(sys.argv[3:])
    )
    with TemporaryDirectory(prefix="census-oneshot-bus-") as scratch:
        root = Path(scratch)
        input_path = root / "request.json"
        input_path.write_text(request.model_dump_json(), encoding="utf-8")
        runtime = RuntimeLocal(
            contract_path,
            input_path=input_path,
            state_root=root / "state",
            backend_overrides={"event_bus": "inmemory"},
            timeout=policy.dispatch_timeout_seconds,
            host_handlers=True,
        )
        runtime.run()
        result = runtime.handler_result
    if not isinstance(result, ModelCensusOneshotResult):
        sys.stderr.write(f"census follow-up dispatch failed: {runtime.last_error}\n")
        return 3
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
