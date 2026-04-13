# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entry point for deploy agent."""

import argparse
import asyncio

from deploy_agent.agent import DeployAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniNode deploy agent")
    parser.add_argument(
        "--skip-self-update",
        action="store_true",
        default=False,
        help="Bypass the self-update check on each deploy (emergency override)",
    )
    args = parser.parse_args()
    agent = DeployAgent(skip_self_update=args.skip_self_update)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
