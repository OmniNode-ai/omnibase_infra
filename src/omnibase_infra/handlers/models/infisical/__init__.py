# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The import path omnimarket's installed tools still load (OMN-22478).

omnibase_infra#5683 moved this model to ``omnibase_infra.models.security`` and
renamed it ``ModelInfisicalConfig``. The move is right and stays. Every omnimarket
build up to and including the one installed in the dispatch venvs imports the old
name at module scope, on the import path of the ledger append, so removing it broke
ledger writes on every host whose venv takes omnibase_infra from a moving clone.
Delete this package once an omnimarket build importing the new name is what those
venvs install.
"""

from omnibase_infra.models.security.model_infisical_config import ModelInfisicalConfig

ModelInfisicalHandlerConfig = ModelInfisicalConfig

__all__: list[str] = ["ModelInfisicalHandlerConfig"]
