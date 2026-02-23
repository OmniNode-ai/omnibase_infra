#!/usr/bin/env python3
"""Configure infinite retention on platform topics for ledger support.

WARNING: This script overrides retention.ms=-1 on ALL platform topics,
including topic-catalog topics that have intentional bounded retention
(1h for query/response, 7d for changed events).
"""

import subprocess

from omnibase_infra.topics import ALL_PLATFORM_SUFFIXES


def main() -> None:
    print(
        f"Configuring infinite retention on {len(ALL_PLATFORM_SUFFIXES)} platform topics"
    )
    print()

    for suffix in ALL_PLATFORM_SUFFIXES:
        # Topics are realm-agnostic — no env prefix. Suffix IS the topic name.
        print(f"  Setting infinite retention on: {suffix}")
        subprocess.run(
            [
                "rpk",
                "topic",
                "alter",
                suffix,
                "--set",
                "retention.ms=-1",
                "--set",
                "retention.bytes=-1",
            ],
            check=True,
        )

    print()
    print(f"Done. {len(ALL_PLATFORM_SUFFIXES)} topics configured.")


if __name__ == "__main__":
    main()
