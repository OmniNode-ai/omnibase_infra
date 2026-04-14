#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Sign and publish a rebuild command to the deploy-agent topic.

Usage:
    python scripts/deploy-trigger.py \\
        --scope full \\
        --requested-by claude \\
        [--git-ref origin/main] \\
        [--services svc1 svc2]

Requires DEPLOY_AGENT_HMAC_SECRET and KAFKA_BOOTSTRAP_SERVERS in the environment
(source ~/.omnibase/.env first).
Generate a secret with: openssl rand -hex 32
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime


def sign_envelope(envelope: dict, secret: str) -> dict:
    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {**envelope, "_signature": signature}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign and send a deploy rebuild command"
    )
    parser.add_argument("--scope", choices=["full", "runtime", "core"], default="full")
    parser.add_argument("--requested-by", default="deploy-trigger-script")
    parser.add_argument("--git-ref", default="origin/main")
    parser.add_argument("--services", nargs="*", default=[])
    parser.add_argument(
        "--broker",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS"),
    )
    parser.add_argument("--correlation-id", default=str(uuid.uuid4()))
    args = parser.parse_args()

    if not args.broker:
        print(
            "ERROR: KAFKA_BOOTSTRAP_SERVERS not set. Source ~/.omnibase/.env first.",
            file=sys.stderr,
        )
        sys.exit(1)

    secret = os.environ.get("DEPLOY_AGENT_HMAC_SECRET")
    if not secret:
        print(
            "ERROR: DEPLOY_AGENT_HMAC_SECRET not set. Source ~/.omnibase/.env first.",
            file=sys.stderr,
        )
        print("       Generate with: openssl rand -hex 32", file=sys.stderr)
        sys.exit(1)

    envelope = {
        "correlation_id": args.correlation_id,
        "requested_by": args.requested_by,
        "scope": args.scope,
        "services": args.services,
        "git_ref": args.git_ref,
        "requested_at": datetime.now(UTC).isoformat(),
    }
    signed = sign_envelope(envelope, secret)

    payload = json.dumps(signed) + "\n"
    print(f"Sending signed rebuild command (correlation_id={args.correlation_id})")
    print(f"  scope={args.scope}  git_ref={args.git_ref}  broker={args.broker}")

    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "omnibase-infra-redpanda",
            "rpk",
            "topic",
            "produce",
            "onex.cmd.deploy.rebuild-requested.v1",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0:
        print(f"Published. correlation_id={args.correlation_id}")
    else:
        print(f"ERROR: rpk produce failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
