# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node runtime for ``runner.action-archive-cache.fill`` (OMN-24299).

Run as ``python -m omnibase_infra.nodes.node_deploy_runners.runtime_action_archive_cache_fill``.
"""

from __future__ import annotations

import argparse
import sys

from omnibase_infra.handlers.handler_action_archive_cache_fill import (
    HandlerActionArchiveCacheFill,
)
from omnibase_infra.models.model_action_archive_cache_fill_request import (
    ModelActionArchiveCacheFillRequest,
)


def main(argv: list[str] | None = None) -> int:
    """Exit 0 filled, 1 any ref failed, 2 the config or its block is absent or inactive."""
    parser = argparse.ArgumentParser(
        description="Fill the action archive cache for self-hosted runners."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    handler = HandlerActionArchiveCacheFill()
    result = handler.handle(
        ModelActionArchiveCacheFillRequest(
            config_path=args.config,
            clone_root=args.clone_root,
            dry_run=args.dry_run,
        )
    )
    sys.stdout.write(f"[action-archive-cache-fill] {result.model_dump_json()}\n")
    if any(f.reason_code == "CONFIG_ABSENT" for f in result.failed):
        return 2
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
