# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Claim one canonical authorization through an overlay-selected Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.runtime.action_authorization_claim.client import (
    claim_action_authorization_via_unix_socket,
)
from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_outcome import (
    EnumActionAuthorizationClaimOutcome,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    ModelActionAuthorizationClaimRequest,
)

_MAX_STDIN_BYTES = 16_384


def main(argv: list[str] | None = None) -> int:
    """Read one canonical JSON request from stdin and print its typed result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        required=True,
        help="Unix socket path resolved by the approved claim-tool overlay.",
    )
    arguments = parser.parse_args(argv)
    raw = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
    if not raw or len(raw) > _MAX_STDIN_BYTES:
        return 2
    try:
        request = ModelActionAuthorizationClaimRequest.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return 2
    result = asyncio.run(
        claim_action_authorization_via_unix_socket(
            socket_path=Path(arguments.socket), request=request
        )
    )
    sys.stdout.write(
        json.dumps(result.model_dump(mode="json"), separators=(",", ":")) + "\n"
    )
    return 0 if result.outcome is EnumActionAuthorizationClaimOutcome.CLAIMED else 1


if __name__ == "__main__":
    raise SystemExit(main())
