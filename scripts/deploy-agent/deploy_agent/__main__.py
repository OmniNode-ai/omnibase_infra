# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entry point for deploy agent."""

import asyncio

from deploy_agent.agent import DeployAgent


def main() -> None:
    agent = DeployAgent()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
