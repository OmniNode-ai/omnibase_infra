# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CLI entrypoint for the edge delegation worker.

Every network destination is a required, explicit flag. Nothing here reads
``ONEX_API_BASE_URL`` or any other environment variable as a silent
fallback -- that variable name is documented as a localhost trap (memory
``project_beta_outside_submit_credential_gap``), and a worker that guessed
its control-plane address from ambient environment state could attach to
the wrong tenant without any operator signal.

This module is import-safe (argument parsing + wiring only) and is not
invoked against a live endpoint anywhere in this build -- see the package
docstring for the build/test-only boundary this stays inside of.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx

from scripts.edge_delegation_worker.delegation_channel import build_kafka_channel
from scripts.edge_delegation_worker.worker_cycle import run_worker_loop

_DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-delegation-worker",
        description=(
            "Claim mirrored delegation-inference requests from the local bus, "
            "run them against a local OpenAI-compatible model, and publish "
            "the result back for the forwarder to mirror to the cloud edge."
        ),
    )
    parser.add_argument(
        "--api-base",
        required=True,
        help=(
            "Base URL of the onex-api deployment exposing /v1/gateway/*"
            " (e.g. https://api.example.onexcloud.dev). Required -- never"
            " read from an environment variable."
        ),
    )
    parser.add_argument(
        "--model-base",
        required=True,
        help=(
            "Base URL of the local OpenAI-compatible model server "
            "(e.g. http://127.0.0.1:8099). Required -- never defaulted."
        ),
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to send in the chat-completion request body.",
    )
    parser.add_argument(
        "--credential-file",
        required=True,
        type=Path,
        help=(
            "Path to a 0600 file holding either one opaque pre-issued bearer "
            "token, or a JSON object with client_id/client_secret/"
            "token_endpoint for a client_credentials grant. Never pass a "
            "credential inline."
        ),
    )
    parser.add_argument(
        "--edge-instance-id",
        required=True,
        help="Stable identifier for this edge worker instance.",
    )
    parser.add_argument(
        "--local-bus-brokers",
        required=True,
        help=(
            "Comma-separated bootstrap servers for the LOCAL Kafka/Redpanda "
            "broker the forwarder mirrors onto (never the cloud edge)."
        ),
    )
    parser.add_argument(
        "--consumer-group",
        required=True,
        help="Kafka consumer group id for this worker's local-bus subscription.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Sleep duration between claim attempts when no message is available.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional bound on claim/infer/publish/ack cycles, for bounded test runs.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=_DEFAULT_HTTP_TIMEOUT_SECONDS,
        help="Timeout applied to every outbound HTTP call this worker makes.",
    )
    return parser


async def _amain(args: argparse.Namespace) -> None:
    channel = build_kafka_channel(
        brokers=args.local_bus_brokers,
        consumer_group=args.consumer_group,
    )
    async with httpx.AsyncClient(timeout=args.http_timeout_seconds) as http_client:
        await channel.start()
        try:
            await run_worker_loop(
                api_base=args.api_base,
                model_base=args.model_base,
                model_name=args.model_name,
                credential_path=args.credential_file,
                edge_instance_id=args.edge_instance_id,
                channel=channel,
                http_client=http_client,
                poll_interval_seconds=args.poll_interval_seconds,
                max_cycles=args.max_cycles,
            )
        finally:
            await channel.stop()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
