# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Safe PostgreSQL routine identity-signature type."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator

_UNQUOTED_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.$ [],"
)


def _validate_function_signature(value: str) -> str:
    """Reject target-list escapes while retaining catalog identity arguments."""
    if len(value) < 2 or value[0] != "(" or value[-1] != ")":
        raise ValueError("function_signature must have one outer argument list")

    body = value[1:-1]
    if not body:
        return value

    arguments: list[str] = []
    current: list[str] = []
    quoted = False
    index = 0
    while index < len(body):
        character = body[index]
        if quoted:
            if character in {"\n", "\r", "\x00"}:
                raise ValueError(
                    "function_signature quoted identifiers cannot contain controls"
                )
            if character == '"':
                if index + 1 < len(body) and body[index + 1] == '"':
                    current.extend(('"', '"'))
                    index += 2
                    continue
                quoted = False
            current.append(character)
            index += 1
            continue

        if character == '"':
            quoted = True
            current.append(character)
        elif character in {"(", ")"}:
            raise ValueError("function_signature cannot contain nested argument lists")
        elif character == ",":
            argument = "".join(current).strip()
            if not argument:
                raise ValueError("function_signature arguments cannot be empty")
            arguments.append(argument)
            current = []
        elif character not in _UNQUOTED_CHARACTERS:
            raise ValueError(
                f"function_signature contains unsafe character {character!r}"
            )
        else:
            current.append(character)
        index += 1

    if quoted:
        raise ValueError("function_signature contains an unterminated identifier")
    final_argument = "".join(current).strip()
    if not final_argument:
        raise ValueError("function_signature arguments cannot be empty")
    arguments.append(final_argument)
    if any(
        "[" in argument.replace("[]", "") or "]" in argument.replace("[]", "")
        for argument in arguments
    ):
        raise ValueError("function_signature array brackets must be complete [] pairs")
    return value


type ApplicationDatabaseFunctionSignature = Annotated[
    str,
    AfterValidator(_validate_function_signature),
]

__all__ = ["ApplicationDatabaseFunctionSignature"]
