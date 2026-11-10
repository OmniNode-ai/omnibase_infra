# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node onex-api image import check -- gate a candidate pin before it is recreated (OMN-23039).

Declarative effect node bound to ``contract.yaml``. All behaviour is in
``HandlerOnexApiImageImportCheck``; the deploy agent loads that handler from its
own checkout (its venv carries a released omnibase_infra that predates it).
"""

from omnibase_infra.nodes.node_onex_api_image_import_check_effect.node import (
    NodeOnexApiImageImportCheckEffect,
)

__all__ = ["NodeOnexApiImageImportCheckEffect"]
