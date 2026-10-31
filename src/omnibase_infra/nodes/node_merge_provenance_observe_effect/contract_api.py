# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the GitHub REST API base from the node contract (OMN-21688)."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONTRACT_PATH = Path(__file__).resolve().parent / "contract.yaml"


def github_api_url(contract_path: Path = _CONTRACT_PATH) -> str:
    """Return the GitHub REST API base declared under ``github_api.base_url``."""
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    block = raw.get("github_api") if isinstance(raw, dict) else None
    base_url = block.get("base_url") if isinstance(block, dict) else None
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"{contract_path}: github_api.base_url must be a string")
    return base_url


__all__ = ["github_api_url"]
