# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One resolved ``Gate:`` binding."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.gate_binding.enum_gate_binding_probe import EnumGateBindingProbe


class ModelGateBinding(BaseModel):
    """A binding line resolved against the declared grammar.

    ``raw`` is what the description carried, ``normalized`` what the declared
    normalizations turned it into, and ``groups`` the named captures of the
    matching form. A caller reads ``probe`` to decide whether anything has to
    be read live at all, and ``groups`` to know what to read.
    """

    model_config = ConfigDict(frozen=True)

    form: str
    probe: EnumGateBindingProbe
    raw: str
    normalized: str
    groups: dict[str, str]
