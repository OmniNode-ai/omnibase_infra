# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose additive, branch-scoped ticket fragments into ordinary contract YAML.

Preparation for shared-ticket evidence gates; this does not enable fragments in
the external reusable workflows. See CLAUDE.md, "Shared ticket contract fragments", before
authoring a fragment. Output is returned as YAML, never written to the root.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_ticket_contract_compose_request import (
    ModelTicketContractComposeRequest,
)
from omnibase_infra.models.model_ticket_contract_compose_result import (
    ModelTicketContractComposeResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

_TICKET = re.compile(r"OMN-[0-9]+")
_SCOPE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_LIST_FIELDS = ("requirements", "evidence_requirements", "dod_evidence")
_FRAGMENT_FIELDS = {"schema_version", "ticket_id", *_LIST_FIELDS}


def _identity_key(field: str, identity: str) -> str:
    if field == "acceptance":
        return re.sub(r"[-_ .]", "", identity).upper()
    return identity


def _items(document: dict[str, object], field: str) -> list[dict[str, object]]:
    value = document.get(field, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of mappings")
    return cast("list[dict[str, object]]", value)


def _load(path: Path, ticket: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{path.name}: symlink contracts are not supported")
    raw = cast(
        "dict[str, object]",
        load_yaml_mapping_no_duplicates(
            path.read_text(encoding="utf-8"), source=path.name
        ),
    )
    if raw.get("ticket_id") != ticket:
        raise ValueError(f"{path.name}: ticket_id must be {ticket}")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{path.name}: contract keys must be strings")
    for field in _LIST_FIELDS:
        _items(raw, field)
    return raw


def _acceptance(document: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for requirement in _items(document, "requirements"):
        result.extend(_items(requirement, "acceptance"))
    return result


def _identities(document: dict[str, object]) -> dict[str, list[str]]:
    result = {}
    for field, items in (
        ("requirements", _items(document, "requirements")),
        ("acceptance", _acceptance(document)),
        ("dod_evidence", _items(document, "dod_evidence")),
    ):
        identities = []
        for item in items:
            identity = item.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{field} id must be a non-empty string")
            identities.append(identity)
        result[field] = identities
    return result


def _validate_fragment(fragment: dict[str, object], scope: str) -> None:
    unknown = set(fragment) - _FRAGMENT_FIELDS
    if unknown:
        raise ValueError(f"{scope}: unsupported fragment fields: {sorted(unknown)}")
    for field in ("requirements", "dod_evidence"):
        if not fragment.get(field):
            raise ValueError(f"{scope}: {field} must be non-empty")
    identities = _identities(fragment)
    for field, prefix in (
        ("requirements", f"req-{scope}-"),
        ("dod_evidence", f"dod-{scope}-"),
    ):
        seen = set()
        for identity in identities[field]:
            if (
                not isinstance(identity, str)
                or not identity.startswith(prefix)
                or identity == prefix
            ):
                raise ValueError(
                    f"{scope}: {field} id must have prefix {prefix!r} and a suffix"
                )
            if identity in seen:
                raise ValueError(f"{scope}: duplicate {field} id {identity!r}")
            seen.add(identity)
    # ModelTicketContract accepts bare numeric criterion labels. Scope names
    # belong on requirement/evidence IDs; dotted AC labels fail that schema.
    seen_criteria: set[str] = set()
    for identity in identities["acceptance"]:
        if not re.fullmatch(r"AC[1-9][0-9]*", identity):
            raise ValueError(f"{scope}: acceptance id must be AC<positive integer>")
        if identity in seen_criteria:
            raise ValueError(f"{scope}: duplicate acceptance id {identity!r}")
        seen_criteria.add(identity)
    criteria = set(identities["acceptance"])
    if not criteria:
        raise ValueError(f"{scope}: requirements must declare acceptance criteria")
    for item in [*_items(fragment, "requirements"), *_acceptance(fragment)]:
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"{scope}: requirements and acceptance need a statement")
    bound = set()
    for item in _items(fragment, "dod_evidence"):
        checks = _items(item, "checks")
        if (
            not isinstance(checks, list)
            or not checks
            or any(
                not isinstance(check, dict)
                or not check.get("check_type")
                or not (check.get("check_value") or check.get("command"))
                for check in checks
            )
        ):
            raise ValueError(f"{scope}: {item['id']} needs non-empty checks")
        labels = item.get("binds_ac")
        if not isinstance(labels, list) or any(
            not isinstance(label, str) for label in labels
        ):
            raise ValueError(
                f"{scope}: {item['id']} binds_ac must be a list of strings"
            )
        typed_labels = cast("list[str]", labels)
        unknown_labels = set(typed_labels) - criteria
        if unknown_labels:
            raise ValueError(
                f"{scope}: {item['id']} binds unknown criterion {sorted(unknown_labels)}"
            )
        bound.update(typed_labels)
    if criteria - bound:
        raise ValueError(
            f"{scope}: unbound acceptance criteria: {sorted(criteria - bound)}"
        )


def compose_ticket_contract(contracts_dir: Path, ticket: str) -> dict[str, object]:
    """Keep root metadata, append sorted fragments, refuse ambiguous additions.

    Legacy root identities are preserved, including historic repeated criterion
    labels. Fragments cannot repeat any existing identity or bind another scope.
    Checks are preserved, never executed or judged as proof by this composer.
    """
    if not _TICKET.fullmatch(ticket):
        raise ValueError("ticket must match OMN-<digits>")
    root = _load(contracts_dir / f"{ticket}.yaml", ticket)
    fragments = contracts_dir / f"{ticket}.d"
    if fragments.is_symlink():
        raise ValueError(
            f"{fragments.name}: symlink fragment directories are not supported"
        )
    if not fragments.exists():
        return root
    known = {
        field: {_identity_key(field, identity) for identity in ids}
        for field, ids in _identities(root).items()
    }
    for path in sorted(fragments.iterdir()):
        if (
            path.suffix != ".yaml"
            or not _SCOPE.fullmatch(path.stem)
            or not path.is_file()
        ):
            raise ValueError(
                f"{path.name}: expected a <lowercase-scope>.yaml fragment file"
            )
        fragment = _load(path, ticket)
        if fragment.get("schema_version") != root.get("schema_version"):
            raise ValueError(
                f"{path.name}: schema_version must match the root contract"
            )
        _validate_fragment(fragment, path.stem)
        for field, identities in _identities(fragment).items():
            keys = {_identity_key(field, identity) for identity in identities}
            duplicates = known[field].intersection(keys)
            if duplicates:
                raise ValueError(
                    f"{path.name}: duplicate {field} id {sorted(duplicates)}"
                )
            known[field].update(keys)
        for field in _LIST_FIELDS:
            if field in fragment:
                root[field] = [*_items(root, field), *_items(fragment, field)]
    return root


class HandlerTicketContractCompose:
    """Read root/fragments and return an ordinary contract without writing it."""

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelTicketContractComposeRequest
    ) -> ModelTicketContractComposeResult:
        try:
            contract = compose_ticket_contract(
                Path(request.contracts_dir), request.ticket_ref
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            return ModelTicketContractComposeResult(
                success=False, error_message=str(error)
            )
        return ModelTicketContractComposeResult(
            success=True, contract_yaml=yaml.safe_dump(contract, sort_keys=False)
        )
