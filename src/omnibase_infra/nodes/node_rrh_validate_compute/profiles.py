# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Built-in RRH validation profiles.

Profiles are defined as Python constants rather than YAML files because:

1. The validate compute handler must be **pure** (no file I/O).
2. YAML files under ``nodes/`` would be discovered by the ONEX contract
   validator, which expects every ``.yaml`` to be a valid contract.
3. Python constants are type-checked at import time.

Profile Precedence:
    PROFILE baseline -> CONTRACT can only TIGHTEN -> Final rule set
"""

from __future__ import annotations

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
from omnibase_infra.models.rrh.model_rrh_rule_severity import ModelRRHRuleSeverity

# ------------------------------------------------------------------
# Helper to reduce repetition
# ------------------------------------------------------------------

_S = ModelRRHRuleSeverity
_F = EnumVerdict.FAIL
_W = EnumVerdict.WARN


def _rule(
    rule_id: str, *, enabled: bool = True, severity: EnumVerdict = _F
) -> ModelRRHRuleSeverity:
    return _S(rule_id=rule_id, enabled=enabled, severity=severity)


# ------------------------------------------------------------------
# default — baseline profile
# ------------------------------------------------------------------

PROFILE_DEFAULT = ModelRRHProfile(
    name="default",
    description=(
        "Default RRH profile.  Covers core repo, environment, and toolchain "
        "rules.  Conditional rules (Kafka, Kubernetes, pytest) are disabled "
        "unless the contract governance activates them."
    ),
    rules=(
        # Repo checks
        _rule("RRH-1001", severity=_F),
        _rule("RRH-1002", severity=_W),
        # Environment checks
        _rule("RRH-1101", severity=_W),
        _rule("RRH-1102", enabled=False, severity=_W),
        # Kafka checks (conditional)
        _rule("RRH-1201", enabled=False, severity=_F),
        # Kubernetes checks (conditional)
        _rule("RRH-1301", enabled=False, severity=_F),
        # Toolchain checks
        _rule("RRH-1401", severity=_W),
        _rule("RRH-1402", severity=_W),
        _rule("RRH-1403", enabled=False, severity=_F),
        _rule("RRH-1404", severity=_W),
        # Cross-checks
        _rule("RRH-1501", severity=_W),
        _rule("RRH-1601", severity=_F),
        # Repo-boundary
        _rule("RRH-1701", enabled=False, severity=_F),
    ),
)

# ------------------------------------------------------------------
# ticket-pipeline — stricter than default on repo state
# ------------------------------------------------------------------

PROFILE_TICKET_PIPELINE = ModelRRHProfile(
    name="ticket-pipeline",
    description=(
        "Ticket pipeline profile.  Requires clean working tree, branch match, "
        "and toolchain presence.  Conditional rules activated by contract "
        "governance fields."
    ),
    rules=(
        # Repo checks
        _rule("RRH-1001", severity=_F),
        _rule("RRH-1002", severity=_F),
        # Environment checks
        _rule("RRH-1101", severity=_W),
        _rule("RRH-1102", severity=_W),
        # Kafka checks (conditional)
        _rule("RRH-1201", enabled=False, severity=_F),
        # Kubernetes checks (conditional)
        _rule("RRH-1301", enabled=False, severity=_F),
        # Toolchain checks
        _rule("RRH-1401", severity=_F),
        _rule("RRH-1402", severity=_F),
        _rule("RRH-1403", enabled=False, severity=_F),
        _rule("RRH-1404", severity=_F),
        # Cross-checks
        _rule("RRH-1501", severity=_F),
        _rule("RRH-1601", severity=_F),
        # Repo-boundary
        _rule("RRH-1701", enabled=False, severity=_F),
    ),
)

# ------------------------------------------------------------------
# ci-repair — relaxed for CI repair workflows
# ------------------------------------------------------------------

PROFILE_CI_REPAIR = ModelRRHProfile(
    name="ci-repair",
    description=(
        "CI repair profile.  Allows dirty working tree (RRH-1001 disabled) "
        "to support emergency CI fixes.  Toolchain checks are warnings only."
    ),
    rules=(
        # Repo checks — dirty tree ALLOWED for CI repair
        _rule("RRH-1001", enabled=False, severity=_W),
        _rule("RRH-1002", severity=_W),
        # Environment checks
        _rule("RRH-1101", severity=_W),
        _rule("RRH-1102", enabled=False, severity=_W),
        # Kafka checks (conditional)
        _rule("RRH-1201", enabled=False, severity=_W),
        # Kubernetes checks (conditional)
        _rule("RRH-1301", enabled=False, severity=_W),
        # Toolchain checks — warnings only for CI repair
        _rule("RRH-1401", severity=_W),
        _rule("RRH-1402", severity=_W),
        _rule("RRH-1403", enabled=False, severity=_W),
        _rule("RRH-1404", severity=_W),
        # Cross-checks
        _rule("RRH-1501", severity=_W),
        _rule("RRH-1601", severity=_F),
        # Repo-boundary
        _rule("RRH-1701", enabled=False, severity=_W),
    ),
)

# ------------------------------------------------------------------
# seam-ticket — all rules active at FAIL severity
# ------------------------------------------------------------------

PROFILE_SEAM_TICKET = ModelRRHProfile(
    name="seam-ticket",
    description=(
        "Seam ticket profile.  All 13 rules active at FAIL severity.  "
        "Used when is_seam_ticket governance flag is true, indicating "
        "a cross-repo change that requires maximum validation coverage."
    ),
    rules=(
        # Repo checks
        _rule("RRH-1001", severity=_F),
        _rule("RRH-1002", severity=_F),
        # Environment checks
        _rule("RRH-1101", severity=_F),
        _rule("RRH-1102", severity=_F),
        # Kafka checks — ALWAYS active for seam tickets
        _rule("RRH-1201", severity=_F),
        # Kubernetes checks — ALWAYS active for seam tickets
        _rule("RRH-1301", severity=_F),
        # Toolchain checks — all active
        _rule("RRH-1401", severity=_F),
        _rule("RRH-1402", severity=_F),
        _rule("RRH-1403", severity=_F),
        _rule("RRH-1404", severity=_F),
        # Cross-checks — all active
        _rule("RRH-1501", severity=_F),
        _rule("RRH-1601", severity=_F),
        # Repo-boundary — active for seam tickets
        _rule("RRH-1701", severity=_F),
    ),
)

# ------------------------------------------------------------------
# Profile registry
# ------------------------------------------------------------------

PROFILES: dict[str, ModelRRHProfile] = {
    "default": PROFILE_DEFAULT,
    "ticket-pipeline": PROFILE_TICKET_PIPELINE,
    "ci-repair": PROFILE_CI_REPAIR,
    "seam-ticket": PROFILE_SEAM_TICKET,
}


def get_profile(name: str) -> ModelRRHProfile:
    """Retrieve a built-in RRH profile by name.

    Args:
        name: Profile name (``default``, ``ticket-pipeline``,
            ``ci-repair``, ``seam-ticket``).

    Returns:
        The matching ``ModelRRHProfile``.

    Raises:
        KeyError: If the profile name is not recognized.
    """
    try:
        return PROFILES[name]
    except KeyError:
        available = ", ".join(sorted(PROFILES))
        msg = f"Unknown RRH profile '{name}'. Available: {available}"
        raise KeyError(msg) from None


__all__: list[str] = [
    "PROFILE_CI_REPAIR",
    "PROFILE_DEFAULT",
    "PROFILE_SEAM_TICKET",
    "PROFILE_TICKET_PIPELINE",
    "PROFILES",
    "get_profile",
]
