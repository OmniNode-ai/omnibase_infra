# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node onex-api image build -- the image the dev pin delivery advances to (OMN-23521).

Declarative effect node bound to ``contract.yaml``. All behaviour is in
``HandlerOnexApiImageBuild``; the deploy agent loads that handler from its
own checkout (its venv carries a released omnibase_infra that predates it).
"""

from omnibase_infra.nodes.node_onex_api_image_build_effect.node import (
    NodeOnexApiImageBuildEffect,
)

__all__ = ["NodeOnexApiImageBuildEffect"]
