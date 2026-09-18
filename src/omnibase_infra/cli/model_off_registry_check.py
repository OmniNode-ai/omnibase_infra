# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The off-registry omnimarket drift verdict, and the line that states it
(OMN-17255).

Rendered once, consumed twice: as a structured stderr line during the run, and
as a receipt fragment that outlives the terminal. Both come from this one
object so the two cannot disagree about what the guard found.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibase_infra.cli.enum_off_registry_reason import EnumOffRegistryReason
from omnibase_infra.cli.enum_off_registry_verdict import EnumOffRegistryVerdict

__all__ = ["ModelOffRegistryCheck"]


@dataclass(frozen=True)
class ModelOffRegistryCheck:
    """One off-registry verdict, and the line that states it.

    ``pin_name`` / ``expected`` / ``installed`` always describe the SAME
    subject -- the deciding requirement -- so no field changes meaning between
    verdicts: the first unsatisfied pin when there is one, otherwise the pin on
    ``omnibase-infra`` (the layer this guard ships in) or the first pin by name.
    ``pins`` and ``unsatisfied`` carry the summary the single deciding pin
    cannot.
    """

    verdict: EnumOffRegistryVerdict
    reason: EnumOffRegistryReason
    omnibase_infra_version: str | None = None
    omnimarket_version: str | None = None
    anchor: str | None = None
    anchor_version: str | None = None
    pin_name: str | None = None
    expected: str | None = None
    installed: str | None = None
    pins: int = 0
    unsatisfied: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        """The one structured line, emitted for every verdict."""
        anchor = (
            f"{self.anchor}@{self.anchor_version}"
            if self.anchor is not None and self.anchor_version is not None
            else "NONE"
        )
        return (
            "drift_guard: "
            f"mode=off-registry "
            f"omnimarket={self.omnimarket_version or 'ABSENT'} "
            f"anchor={anchor} "
            f"pin={self.pin_name or 'NONE'} "
            f"expected={self.expected or 'NONE'} "
            f"installed={self.installed or ('ABSENT' if self.pin_name else 'NONE')} "
            f"pins={self.pins} "
            f"unsatisfied={len(self.unsatisfied)} "
            f"verdict={self.verdict.value} "
            f"reason={self.reason.value}"
        )

    def as_receipt_fields(self) -> dict[str, object]:
        """The same facts, for a run receipt.

        A customer reads the receipt after the fact; the stderr line is gone by
        then. Both are rendered from this one object so they cannot disagree.
        """
        return {
            "mode": "off-registry",
            "verdict": self.verdict.value,
            "reason": self.reason.value,
            "omnimarket_version": self.omnimarket_version,
            "anchor": self.anchor,
            "anchor_version": self.anchor_version,
            "pin": self.pin_name,
            "expected": self.expected,
            "installed": self.installed,
            "pins": self.pins,
            "unsatisfied": list(self.unsatisfied),
            "line": self.line,
        }
