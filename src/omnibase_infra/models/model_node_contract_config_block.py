# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One block of a node contract's ``config`` section, read from disk.

Pure nodes may not touch the filesystem, so the read lives here, outside the
node packages, and the node's models receive the parsed block.

Related Tickets:
    - OMN-21623: model names leave the analyzer; the node contract declares them
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ModelNodeContractConfigBlock:
    """Reads ``config[key]`` from a node contract file."""

    @staticmethod
    def load(contract_path: Path, key: str) -> object:
        """Return ``config[key]``, or ``None`` when the document, its ``config``
        block or the key is absent, so the caller's validation rejects it.
        """
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        config = document.get("config") if isinstance(document, dict) else None
        return config.get(key) if isinstance(config, dict) else None


__all__: list[str] = ["ModelNodeContractConfigBlock"]
