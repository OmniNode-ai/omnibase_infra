# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check the five compose passwords against one stated format contract, up front.

OMN-19087. A lane deploy consumes five passwords. Before this check, exactly three
of them had a format contract, and it was enforced in one place: the forward
migration (``scripts/run-forward-migrations.sh``), which runs after Postgres,
Redpanda and Valkey are already up. On 2026-09-21 a dogfood cold bring-up used a
mixed alphanumeric generator, every infrastructure service came up healthy, and
the deploy died minutes later in a different script, naming a third variable.

THE CONTRACT (this table is the one statement of it)
----------------------------------------------------
  variable                            format    why
  POSTGRES_PASSWORD                   url-safe  interpolated unescaped into every
                                                postgresql:// DSN the compose files render
  VALKEY_PASSWORD                     url-safe  interpolated unescaped into redis:// URLs
                                                and a shell --requirepass
  OMNINODE_RUNTIME_PASSWORD           hex       interpolated into a SQL string literal
  TENANT_PROJECTION_WRITER_PASSWORD   hex       by the bootstrap and the forward
  CHAIN_CANARY_READER_PASSWORD        hex       migration (validate_password and
                                                reassert_login_only_role_credential)

"url-safe" is ``[A-Za-z0-9._-]+``: characters that need no escaping in a URL, in a
compose interpolation, in a sourced env file or in a quoted shell argument (``~`` is
left out because an unquoted assignment tilde-expands it). "hex" is ``[0-9a-fA-F]+``.
``openssl rand -hex 32`` satisfies both, so it is the one generator the examples
name. The bootstrap's placeholder form, ``__REPLACE_WITH_...__``, is refused for all
five, as the bootstrap refuses it for the login roles.

WHY NOT HEX FOR ALL FIVE. The two infrastructure passwords are not hex on any lab
host today (read 2026-09-23 on .201, .101 and .105, by character class only), and
``POSTGRES_PASSWORD`` cannot be changed once the Postgres data volume has
initialised: the volume keeps the value it was created with. A hex rule on it would
refuse every existing lane, and the obvious remedy, regenerating it, is the trap
that turns a format refusal into an authentication failure. The login-role hex rule
stays exactly as it is.

WHAT THIS CHECKS AND WHAT IT DOES NOT. Format only, and only for a value that is
set and non-empty. Presence is already enforced, by the compose ``${VAR:?}`` guards
and by ``scripts/preflight_required_compose_env.py``; a login-role password that is
unset is a documented skip in the migration runner. A set value outside its format
is refused, every such variable is named in one message, and no value is ever
printed.

Callers, both before any container starts:
  - ``scripts/deploy-runtime.sh`` (``guard_password_contract``)
  - ``scripts/preflight_required_compose_env.py`` (the deploy agent and
    ``refresh_dev_lane.sh`` run it before compose validation)

Pinned by ``tests/scripts/test_password_contract_omn19087.py`` against the
migration runner, the bootstrap and every tracked env example.

Exit codes:
  0 - every set password matches its format
  1 - at least one does not (all of them are named)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PasswordFormat:
    """One named format a password must match in full."""

    name: str
    pattern: re.Pattern[str]
    rule: str


HEX = PasswordFormat(
    name="hex",
    pattern=re.compile(r"[0-9a-fA-F]+"),
    rule="hexadecimal digits only, [0-9a-fA-F]",
)

URL_SAFE = PasswordFormat(
    name="url-safe",
    pattern=re.compile(r"[A-Za-z0-9._-]+"),
    rule="ASCII letters, digits, '.', '_' and '-' only, [A-Za-z0-9._-]",
)

# The bootstrap's placeholder test (validate_password in
# docker/migrations/forward/000_create_multiple_databases.sh), applied to all five.
PLACEHOLDER = re.compile(r"__REPLACE_WITH_.*__")

GENERATOR = "openssl rand -hex 32"

# The three login-role variables, in the order run-forward-migrations.sh and the
# bootstrap's LOGIN_ONLY_ROLE_MAP list them.
LOGIN_ROLE_VARS: tuple[str, ...] = (
    "OMNINODE_RUNTIME_PASSWORD",
    "TENANT_PROJECTION_WRITER_PASSWORD",
    "CHAIN_CANARY_READER_PASSWORD",
)

INFRA_VARS: tuple[str, ...] = (
    "POSTGRES_PASSWORD",
    "VALKEY_PASSWORD",
)

CONTRACT: dict[str, PasswordFormat] = {
    **dict.fromkeys(INFRA_VARS, URL_SAFE),
    **dict.fromkeys(LOGIN_ROLE_VARS, HEX),
}

VOLUME_FIXED_VAR = "POSTGRES_PASSWORD"


def violations(environ: Mapping[str, str]) -> list[str]:
    """Names of the contract variables whose set value is outside its format.

    Unset and empty values are not judged here (see the module docstring).
    """
    found: list[str] = []
    for var, fmt in CONTRACT.items():
        value = environ.get(var, "")
        if not value:
            continue
        if PLACEHOLDER.fullmatch(value) or fmt.pattern.fullmatch(value) is None:
            found.append(var)
    return found


def render_env_example_block() -> str:
    """The contract as every tracked env example states it, verbatim.

    Generated from ``CONTRACT`` so the examples cannot state a different rule
    from the one enforced; the pin test compares each example to this text.
    """
    lines = [
        "# PASSWORD FORMAT CONTRACT (OMN-19087). Checked before any container starts",
        "# by scripts/preflight_password_contract.py, which names every variable that",
        "# breaks it. A set value outside its format is refused; the placeholder",
        "# form __REPLACE_WITH_...__ is refused for all five.",
    ]
    for var, fmt in CONTRACT.items():
        lines.append(f"#   {var} is {fmt.name} ({fmt.rule})")
    lines.append(f"# `{GENERATOR}` satisfies every format above.")
    lines.append(
        f"# {VOLUME_FIXED_VAR} is fixed once the Postgres data volume initialises: a"
    )
    lines.append(
        "# changed value fails authentication against the existing volume. Set it"
    )
    lines.append(
        "# once, before the first bring-up, and change only a variable a refusal"
    )
    lines.append("# names.")
    return "\n".join(lines)


def render_refusal(names: list[str], lane: str) -> str:
    """The refusal message. Names and contracts only, never a value."""
    lines = [
        f"ERROR: PASSWORD_CONTRACT_VIOLATION - {len(names)} of {len(CONTRACT)} "
        f"compose passwords for lane '{lane}' are outside their stated format. "
        "Refused before any container starts (OMN-19087).",
        "",
    ]
    for var in names:
        fmt = CONTRACT[var]
        lines.append(f"    {var}")
        lines.append(f"        contract : {fmt.name} - {fmt.rule}")
        lines.append(f"        generate : {GENERATOR} (satisfies every format here)")
    lines.append("")
    lines.append(
        "  Change ONLY the variables named above. Regenerating the others is not"
    )
    lines.append("  needed and can break a lane that already has a data volume.")
    if VOLUME_FIXED_VAR in names:
        lines.append("")
        lines.append(
            f"  {VOLUME_FIXED_VAR} is fixed once the Postgres data volume initialises."
        )
        lines.append(
            "  If this lane's volume already exists, a new value fails authentication:"
        )
        lines.append(
            "  the volume must be recreated (losing its data) or the value it was"
        )
        lines.append(
            "  initialised with restored. On a fresh lane, just set a new one."
        )
    lines.append("")
    lines.append(
        "  The contract is stated once, in scripts/preflight_password_contract.py."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refuse a deploy whose compose passwords are set outside their "
            "stated format, naming every such variable. Values are never printed."
        ),
    )
    parser.add_argument(
        "--lane",
        default="dev",
        help="Lane name, for the message only (default: dev).",
    )
    args = parser.parse_args(argv)

    names = violations(os.environ)
    if not names:
        print(
            f"OK: every set compose password for lane '{args.lane}' matches its "
            f"format ({len(CONTRACT)} checked)."
        )
        return 0
    print(render_refusal(names, args.lane), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
