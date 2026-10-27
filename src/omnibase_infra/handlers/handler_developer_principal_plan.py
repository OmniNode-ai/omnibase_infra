# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerDeveloperPrincipalPlan: plan a developer machine's broker principal.

Planning a principal (its name, create or rotate, the exact grant list and the
digest of the declaration) was pure logic that lived only in
``scripts/lab_principal_issuer.py`` on the lab host (``load_declaration``,
``principal_name``). This handler holds the same rules as a compute node, so a
developer-run broker can plan the principal the same way the lab does. The two
are held to each other by a parity test over the same fixtures.

Pure: the caller supplies the declaration's text and the text of every derived
grant document it names, so the handler reads no file, environment, clock or
socket. The same request returns the same plan.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelDeveloperPrincipalPlan``.

Ticket: OMN-20821
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Literal

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.developer_principal.model_developer_principal_grant import (
    ModelDeveloperPrincipalGrant,
)
from omnibase_infra.models.developer_principal.model_developer_principal_plan import (
    ModelDeveloperPrincipalPlan,
)
from omnibase_infra.models.developer_principal.model_developer_principal_plan_request import (
    ModelDeveloperPrincipalPlanRequest,
)
from omnibase_infra.models.developer_principal.topic_constants import (
    SCOPED_WRITE_TOPICS,
)

SCHEMA = "developer-principal-grants.v1"
PRINCIPAL_MAX_LEN = 63
MIN_PREFIXED_READ_GROUP_LEN = 16
ALLOWED_OPERATIONS = frozenset({"read", "write", "describe"})
ALLOWED_RESOURCES = frozenset({"topic", "group"})
ALLOWED_PATTERNS = frozenset({"literal", "prefixed"})
_PREFIX = re.compile(r"[a-z][a-z0-9-]{0,15}")
_EXACT_PRINCIPAL = re.compile(r"[a-z][a-z0-9-]{0,62}")
_PLACEHOLDER = "{principal}"


class DeclarationRefusedError(Exception):
    """The declaration asks for something the lab issuer would refuse."""


@dataclasses.dataclass(frozen=True)
class PlannedGrant:
    resource: str
    name: str
    pattern: str
    operations: tuple[str, ...]
    templated: bool
    create_topic: bool


@dataclasses.dataclass(frozen=True)
class PlannedDeclaration:
    lane: str
    principal_prefix: str
    max_devices_per_login: int
    allowed_login_domains: tuple[str, ...]
    grants: tuple[PlannedGrant, ...]
    digest: str
    principal_grants: tuple[tuple[str, tuple[PlannedGrant, ...]], ...]

    def grants_for(self, principal: str) -> tuple[PlannedGrant, ...]:
        scoped = (
            grant
            for owner, grants in self.principal_grants
            if owner == principal
            for grant in grants
        )
        return tuple(dict.fromkeys((*self.grants, *scoped)))


def _declaration(text: str, documents: Mapping[str, str]) -> PlannedDeclaration:
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise DeclarationRefusedError(f"schema must be {SCHEMA}")
    lane = doc.get("lane")
    prefix = doc.get("principal_prefix")
    max_devices = doc.get("max_devices_per_login")
    domains = doc.get("allowed_login_domains") or []
    if not isinstance(lane, str) or not lane:
        raise DeclarationRefusedError("lane must be a non-empty string")
    if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix):
        raise DeclarationRefusedError(
            "principal_prefix must match [a-z][a-z0-9-]{0,15}"
        )
    if not isinstance(max_devices, int) or not 1 <= max_devices <= 20:
        raise DeclarationRefusedError(
            "max_devices_per_login must be an integer from 1 to 20"
        )
    if not isinstance(domains, list) or not all(
        isinstance(d, str) and d for d in domains
    ):
        raise DeclarationRefusedError("allowed_login_domains must be a list of domains")
    digest = hashlib.sha256(text.encode("utf-8"))
    grants_raw: object
    if "grants_from" in doc:
        if "grants" in doc:
            raise DeclarationRefusedError(
                "grants_from and grants are exclusive; a derived grant is never typed here"
            )
        grants_raw, derived = _derived_grants(doc["grants_from"], documents, lane)
        digest.update(derived)
    else:
        grants_raw = doc.get("grants")
    if not isinstance(grants_raw, list) or not grants_raw:
        raise DeclarationRefusedError("grants must be a non-empty list")
    grants = tuple(_grant(i, g) for i, g in enumerate(grants_raw))
    scoped = doc.get("principal_grants_from", {})
    if not isinstance(scoped, dict):
        raise DeclarationRefusedError("principal_grants_from must be a mapping")
    if scoped and lane != "dev":
        raise DeclarationRefusedError("principal additions are dev-only")
    principal_grants: list[tuple[str, tuple[PlannedGrant, ...]]] = []
    for principal, source in scoped.items():
        if (
            not isinstance(principal, str)
            or not principal.startswith(prefix)
            or not _EXACT_PRINCIPAL.fullmatch(principal)
        ):
            raise DeclarationRefusedError("principal additions need an exact principal")
        rows, derived = _derived_grants(source, documents, lane, principal=principal)
        additions = tuple(_grant(i, row) for i, row in enumerate(rows))
        if not additions or any(
            g.templated or ("write" in g.operations and not _scoped_write_allowed(g))
            for g in additions
        ):
            raise DeclarationRefusedError(
                "principal additions must be non-empty, concrete reads "
                f"(a write only on {', '.join(sorted(SCOPED_WRITE_TOPICS))})"
            )
        principal_grants.append((principal, additions))
        digest.update(derived)
    return PlannedDeclaration(
        lane=lane,
        principal_prefix=prefix,
        max_devices_per_login=max_devices,
        allowed_login_domains=tuple(d.lower() for d in domains),
        grants=grants,
        digest=digest.hexdigest(),
        principal_grants=tuple(principal_grants),
    )


def _scoped_write_allowed(grant: PlannedGrant) -> bool:
    return (
        grant.resource == "topic"
        and grant.pattern == "literal"
        and grant.name in SCOPED_WRITE_TOPICS
    )


def _derived_grants(
    grants_from: object,
    documents: Mapping[str, str],
    lane: str,
    *,
    principal: str = _PLACEHOLDER,
) -> tuple[list[dict[str, object]], bytes]:
    """Fold a derived grant document into one grant per resource, as the issuer does."""
    if not isinstance(grants_from, str) or not grants_from:
        raise DeclarationRefusedError("grants_from must be a file name")
    if "/" in grants_from or grants_from in {".", ".."}:
        raise DeclarationRefusedError("grants_from must name a file beside it")
    if grants_from not in documents:
        raise DeclarationRefusedError(
            f"{grants_from}: the derived document was not supplied"
        )
    text = documents[grants_from]
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DeclarationRefusedError(f"{grants_from}: not valid JSON") from exc
    if (
        not isinstance(doc, dict)
        or set(doc) != {"principal", "grants"}
        or not isinstance(doc["grants"], list)
    ):
        raise DeclarationRefusedError(
            f"{grants_from}: holds exactly principal and grants, as the grant derivation emits"
        )
    if doc["principal"] != principal:
        raise DeclarationRefusedError(
            f"{grants_from}: derived principal must be {principal}"
        )
    folded: dict[tuple[str, str, str], set[str]] = {}
    for index, row in enumerate(doc["grants"]):
        if not isinstance(row, dict) or set(row) != {
            "broker",
            "resource_type",
            "resource",
            "operation",
            "pattern",
        }:
            raise DeclarationRefusedError(
                f"{grants_from}: grants[{index}] is not a derived grant"
            )
        if row["broker"] != lane:
            raise DeclarationRefusedError(
                f"{grants_from}: grants[{index}] is for broker {row['broker']}, not lane {lane}"
            )
        key = (
            str(row["resource_type"]).lower(),
            str(row["resource"]),
            str(row["pattern"]).lower(),
        )
        folded.setdefault(key, set()).add(str(row["operation"]).lower())
    grants: list[dict[str, object]] = []
    for (resource, name, pattern), operations in sorted(folded.items()):
        grant: dict[str, object] = {
            "resource": resource,
            "name": name,
            "pattern": pattern,
            "operations": sorted(operations),
        }
        if _PLACEHOLDER in name:
            grant["create_topic"] = "write" in operations
        grants.append(grant)
    return grants, text.encode("utf-8")


def _grant(index: int, raw: object) -> PlannedGrant:
    where = f"grants[{index}]"
    if not isinstance(raw, dict):
        raise DeclarationRefusedError(f"{where} must be a mapping")
    resource, name, pattern = raw.get("resource"), raw.get("name"), raw.get("pattern")
    operations = raw.get("operations")
    if resource not in ALLOWED_RESOURCES:
        raise DeclarationRefusedError(
            f"{where}: resource must be one of {sorted(ALLOWED_RESOURCES)}"
        )
    if not isinstance(name, str) or not name:
        raise DeclarationRefusedError(f"{where}: name must be a non-empty string")
    if pattern not in ALLOWED_PATTERNS:
        raise DeclarationRefusedError(
            f"{where}: pattern must be one of {sorted(ALLOWED_PATTERNS)}"
        )
    remainder = name.replace(_PLACEHOLDER, "")
    if "{" in remainder or "}" in remainder:
        raise DeclarationRefusedError(
            f"{where}: only the '{{principal}}' placeholder is allowed"
        )
    templated = _PLACEHOLDER in name
    if templated:
        if name.count(_PLACEHOLDER) != 1:
            raise DeclarationRefusedError(
                f"{where}: '{{principal}}' must appear exactly once"
            )
        if resource != "topic" or pattern != "literal":
            raise DeclarationRefusedError(
                f"{where}: a templated grant must be a literal topic"
            )
    create_topic = raw.get("create_topic", False)
    if not isinstance(create_topic, bool):
        raise DeclarationRefusedError(f"{where}: create_topic must be a bool")
    if "create_topic" in raw and not templated:
        raise DeclarationRefusedError(
            f"{where}: create_topic is allowed only on templated topic grants"
        )
    if (
        not isinstance(operations, list)
        or not operations
        or not set(operations) <= ALLOWED_OPERATIONS
    ):
        raise DeclarationRefusedError(
            f"{where}: operations must be a subset of {sorted(ALLOWED_OPERATIONS)}"
        )
    ops = tuple(sorted(set(operations)))
    if name == "*" and ops != ("describe",):
        raise DeclarationRefusedError(f"{where}: '*' may carry describe only")
    if pattern == "prefixed" and name == "*":
        raise DeclarationRefusedError(f"{where}: '*' is a literal, not a prefix")
    if (
        resource == "group"
        and pattern == "prefixed"
        and "read" in ops
        and len(name) < MIN_PREFIXED_READ_GROUP_LEN
    ):
        raise DeclarationRefusedError(
            f"{where}: a prefixed group read grant must be at least "
            f"{MIN_PREFIXED_READ_GROUP_LEN} characters, so it cannot reach a lab runtime's group"
        )
    return PlannedGrant(
        resource=resource,
        name=name,
        pattern=pattern,
        operations=ops,
        templated=templated,
        create_topic=create_topic,
    )


def slug(value: str) -> str:
    """Lower-case, [a-z0-9-] only, no leading/trailing/double hyphens."""
    out = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", out)


def principal_name(prefix: str, login: str, device: str) -> str:
    """``<prefix><login local part>-<device>``, capped at the broker-safe length.

    Empty when the login or the device has no letter or digit.
    """
    who = slug(login.split("@", 1)[0])
    what = slug(device)
    if not who or not what:
        return ""
    name = f"{prefix}{who}-{what}"
    if len(name) > PRINCIPAL_MAX_LEN:
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        name = f"{name[: PRINCIPAL_MAX_LEN - 9].rstrip('-')}-{digest}"
    return name


def _refused(
    kind: Literal["declaration", "identity", "device_cap"],
    refusal: str,
    *,
    declaration: PlannedDeclaration | None = None,
    principal: str = "",
) -> ModelDeveloperPrincipalPlan:
    return ModelDeveloperPrincipalPlan(
        decision="refused",
        principal=principal,
        lane=declaration.lane if declaration else "",
        declaration_digest=declaration.digest if declaration else "",
        refusal_kind=kind,
        refusal=refusal,
    )


def _resolved(
    grants: Sequence[PlannedGrant], principal: str
) -> tuple[ModelDeveloperPrincipalGrant, ...]:
    resolved = [
        ModelDeveloperPrincipalGrant.model_validate(
            {
                "resource": g.resource,
                "name": g.name.replace(_PLACEHOLDER, principal)
                if g.templated
                else g.name,
                "pattern": g.pattern,
                "operations": g.operations,
                "create_topic": g.create_topic,
            }
        )
        for g in grants
    ]
    return tuple(sorted(resolved, key=lambda g: (g.resource, g.name, g.pattern)))


class HandlerDeveloperPrincipalPlan:
    """Pure compute handler: one developer principal's plan from a grants declaration.

    Classification: ``COMPUTE_HANDLER`` + ``COMPUTE``. No I/O and no side
    effects; the plan is a function of the request alone.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelDeveloperPrincipalPlanRequest
    ) -> ModelDeveloperPrincipalPlan:
        """Plan the principal: refuse, create, or rotate one the broker already holds."""
        try:
            declaration = _declaration(
                request.declaration_yaml, request.derived_documents
            )
        except DeclarationRefusedError as exc:
            return _refused("declaration", str(exc))
        login = request.login.strip().lower()
        if not login:
            return _refused(
                "identity",
                "no login to name the principal for",
                declaration=declaration,
            )
        domains = declaration.allowed_login_domains
        if domains and login.rsplit("@", 1)[-1] not in domains:
            return _refused(
                "identity",
                f"login {login} is not in an allowed domain",
                declaration=declaration,
            )
        principal = principal_name(declaration.principal_prefix, login, request.device)
        if not principal:
            return _refused(
                "identity",
                "login and device must each contain a letter or digit",
                declaration=declaration,
            )
        held = {
            name
            for name in request.existing_principals
            if name.startswith(declaration.principal_prefix)
        }
        if principal not in held and len(held) >= declaration.max_devices_per_login:
            return _refused(
                "device_cap",
                f"{len(held)} principals with prefix {declaration.principal_prefix} are already "
                f"held, the most one login may hold ({declaration.max_devices_per_login}); "
                "retire one on the broker before adding another",
                declaration=declaration,
                principal=principal,
            )
        return ModelDeveloperPrincipalPlan(
            decision="rotate" if principal in held else "create",
            principal=principal,
            lane=declaration.lane,
            grants=_resolved(declaration.grants_for(principal), principal),
            declaration_digest=declaration.digest,
        )


__all__: list[str] = ["HandlerDeveloperPrincipalPlan", "principal_name", "slug"]
