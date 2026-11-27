# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-bound diagnosis for a delegation doctor install with no local model."""

from __future__ import annotations

import importlib

from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

# ``onex models list`` reads its rows through this omnimarket seam. This package may
# not import omnimarket at module scope, so it is resolved when the check runs, the
# way the identity check resolves ``read_install_identity``.
_LIST_SEAM = "omnimarket.handlers.models_only_command"
# The provider name omnimarket gives a Local Model; every other row is hosted.
_LOCAL_PROVIDER = "local"

_NO_SETUP_FIX = (
    "Run `onex models add <provider>` to add a hosted provider, or `onex setup`."
)

__all__ = [
    "HandlerCheckDelegationLocalModel",
    "evaluate_no_local_model",
    "read_hosted_providers",
]


class HandlerCheckDelegationLocalModel:
    """Own the judgement of an install that declares no local chat model."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @staticmethod
    def handle(request: object) -> object:
        """Runtime adapter entry point; doctor check uses the static seam."""
        return request

    @staticmethod
    def read_hosted_providers() -> tuple[str, ...] | None:
        """Return the hosted providers ``onex models list`` shows as set up.

        ``None`` means the list could not be read, so the check does not judge.
        """
        try:
            listed = importlib.import_module(_LIST_SEAM).listed
            return tuple(
                str(row.provider)
                for row in listed(True)
                if row.set_up and str(row.provider) != _LOCAL_PROVIDER
            )
        except (ImportError, AttributeError, OSError, ValueError):
            return None

    @staticmethod
    def evaluate_no_local_model(
        hosted: tuple[str, ...] | None,
    ) -> tuple[ModelDelegationDiagnosis, bool]:
        """A local model is optional: fail only when no model at all is set up."""
        if hosted is None:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail=(
                        "Delegation local model was not judged because the "
                        "set-up models could not be read."
                    ),
                    fix="",
                ),
                False,
            )
        if hosted:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail=(
                        "No local model is set up; delegation uses your hosted "
                        f"provider(s): {', '.join(hosted)}. A local model is optional."
                    ),
                    fix="",
                ),
                True,
            )
        return (
            ModelDelegationDiagnosis(
                fault=EnumDelegationDoctorFault.NO_LOCAL_MODEL,
                detail="No model is set up, local or hosted.",
                fix=_NO_SETUP_FIX,
            ),
            True,
        )


def read_hosted_providers() -> tuple[str, ...] | None:
    """Keep the baselined doctor check thin."""
    return HandlerCheckDelegationLocalModel.read_hosted_providers()


def evaluate_no_local_model(
    hosted: tuple[str, ...] | None,
) -> tuple[ModelDelegationDiagnosis, bool]:
    """Keep the baselined doctor check thin."""
    return HandlerCheckDelegationLocalModel.evaluate_no_local_model(hosted)
