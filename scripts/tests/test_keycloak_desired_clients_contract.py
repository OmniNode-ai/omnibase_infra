#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for the canonical Keycloak client configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parents[2] / "docker/keycloak/desired-clients.json"


def _client(config: dict[str, Any], client_id: str) -> dict[str, Any]:
    matches = [
        client for client in config["clients"] if client["clientId"] == client_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_omniweb_allows_the_managed_staging_callback() -> None:
    config = json.loads(_CONFIG_PATH.read_text())
    omniweb = _client(config, "omniweb")

    assert "https://dev.app.omninode.ai/*" in omniweb["redirectUris"]
    assert "https://dev.app.omninode.ai" in omniweb["webOrigins"]
