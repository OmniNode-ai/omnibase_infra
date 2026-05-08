# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import logging

from omnibase_core.models.contracts.subcontracts.model_hook_activation import (
    ModelHookActivation,
)

_log = logging.getLogger(__name__)


class HookActivationResolver:
    """Resolves package hook_activations into a composite bitmask."""

    def __init__(self, activations: list[ModelHookActivation]) -> None:
        self._activations = activations

    def compute_mask(self) -> int:
        """Return bitmask with bits set for every enabled_by_default activation."""
        mask = 0
        for activation in self._activations:
            try:
                if activation.enabled_by_default:
                    mask |= int(activation.hook_bit)
            except Exception as exc:  # noqa: BLE001 — lenient rollout-safety policy: log and skip, never crash runtime
                _log.warning(
                    "Skipping unresolvable hook activation %s: %s",
                    activation.hook_bit,
                    exc,
                )
        return mask
