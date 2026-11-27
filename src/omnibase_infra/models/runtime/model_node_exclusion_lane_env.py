# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The nodes a lane turns off, typed and checked against the known nodes (OMN-26232).

``ONEX_RUNTIME_NODE_EXCLUSION`` is the runtime's one switch for leaving a node
out (:class:`~omnibase_infra.models.runtime.model_runtime_node_selection.ModelRuntimeNodeSelection`).
The runtime refuses a name no discovered contract has, but only when a container
boots, after the deploy agent has already recreated it. This model parses the
comma list a lane overlay sets and checks every name against the contract names
the image would discover, so a typo is refused where the deploy agent reads the
lane env, before any container is recreated, with the unknown name in the message.

The known names are read the way the runtime finds them: each package's
``onex.nodes`` entry points (``pyproject.toml``), each resolving to a node module
whose ``contract.yaml`` carries the contract ``name`` (the entry point name when
it carries none). Nothing is listed by hand, so a node merged tomorrow is known
tomorrow.

The deploy agent loads this file by path from its own checkout, and its
environment need not carry ``omnibase_infra``: this module imports the standard
library and pydantic only.
"""

from __future__ import annotations

import difflib
import re
import tomllib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field

ENV_RUNTIME_NODE_EXCLUSION: Final = "ONEX_RUNTIME_NODE_EXCLUSION"

#: The entry point group a package declares its nodes under.
NODE_ENTRY_POINT_GROUP: Final = "onex.nodes"

# A contract's top-level ``name:`` scalar, bare or quoted.
_NAME_LINE: Final = re.compile(
    r"""^name:[ \t]*(?:"([^"]*)"|'([^']*)'|([^\s#]+))""", re.MULTILINE
)


def parse_contract_name_list(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated name list, dropping blanks and duplicates."""
    return tuple(
        dict.fromkeys(part.strip() for part in (raw or "").split(",") if part.strip())
    )


def _contract_name(contract: Path, entry_point_name: str) -> str | None:
    """The contract's declared name, or None when the node has no contract file."""
    try:
        text = contract.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _NAME_LINE.search(text)
    if match is None:
        return entry_point_name
    return match.group(1) or match.group(2) or match.group(3) or entry_point_name


def known_contract_names(roots: Sequence[Path]) -> frozenset[str]:
    """The contract names the packages checked out under ``roots`` declare.

    Each root is a package checkout holding ``pyproject.toml`` and a ``src``
    layout. A root without a readable ``pyproject.toml`` contributes nothing.
    """
    names: set[str] = set()
    for root in roots:
        try:
            project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        entry_points = (
            project.get("project", {})
            .get("entry-points", {})
            .get(NODE_ENTRY_POINT_GROUP)
        )
        for entry_point_name, target in (entry_points or {}).items():
            module_path = str(target).split(":", 1)[0].replace(".", "/")
            for base in (root / "src", root):
                name = _contract_name(
                    base / module_path / "contract.yaml", entry_point_name
                )
                if name is not None:
                    names.add(name)
                    break
    return frozenset(names)


def contract_roots(
    environ: Mapping[str, str], *, checkout_root: Path
) -> tuple[Path, ...]:
    """The package checkouts the next deploy builds the runtime image from.

    ``DEPLOY_AGENT_REPO_DIR`` is the deploy-source clone the agent builds
    (``checkout_root`` when unset) and ``<OMNI_HOME>/omnimarket`` is the sibling
    the image build also takes nodes from.
    """
    infra = Path(environ.get("DEPLOY_AGENT_REPO_DIR", "").strip() or checkout_root)
    workspace = environ.get("OMNI_HOME", "").strip()
    roots = [infra, *([Path(workspace) / "omnimarket"] if workspace else [])]
    return tuple(root for root in roots if (root / "pyproject.toml").is_file())


class ModelNodeExclusionLaneEnv(BaseModel):
    """The node contract names a lane's overlay turns off; empty turns none off."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    names: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contract names to leave unattached, in declared order, deduplicated."
        ),
    )

    @classmethod
    def parse(cls, raw: str | None) -> Self:
        """Parse the comma list; unset and blank both name nothing."""
        return cls(names=parse_contract_name_list(raw))

    def unknown(self, known: Collection[str]) -> tuple[str, ...]:
        """The names in this list that no known contract carries."""
        return tuple(name for name in self.names if name not in known)

    def require_known(self, known: Collection[str]) -> Self:
        """Return self, or raise ``ValueError`` naming each unknown contract."""
        unknown = self.unknown(known)
        if not unknown:
            return self
        details = []
        for name in unknown:
            close = difflib.get_close_matches(name, sorted(known), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            details.append(f"{name!r}{hint}")
        raise ValueError(
            f"{ENV_RUNTIME_NODE_EXCLUSION} names unknown node contract(s) "
            f"{', '.join(details)}; the runtime would refuse to start on "
            f"{'it' if len(unknown) == 1 else 'them'}. Known contract names "
            f"number {len(known)}."
        )


def validate_node_exclusion(
    raw: str, environ: Mapping[str, str], checkout_root: Path
) -> None:
    """Refuse a node exclusion that names a contract the next build does not declare.

    Raises:
        ValueError: a name is unknown, or no package checkout was found (so
            nothing could be checked, which is not the same as nothing wrong).
    """
    model = ModelNodeExclusionLaneEnv.parse(raw)
    roots = contract_roots(environ, checkout_root=checkout_root)
    known = known_contract_names(roots)
    if model.names and not known:
        raise ValueError(
            f"{ENV_RUNTIME_NODE_EXCLUSION} cannot be checked: no package checkout "
            f"declaring {NODE_ENTRY_POINT_GROUP} entry points was found"
        )
    model.require_known(known)


def node_exclusion_refusal(
    raw: str, environ: Mapping[str, str], agent_file: str
) -> str:
    """Why ``raw`` is refused, or the empty string when it is accepted.

    ``agent_file`` is the deploy agent's ``instance_lanes.py``: the checkout
    root (``<root>/scripts/deploy-agent/deploy_agent/``) is three levels up.
    """
    try:
        validate_node_exclusion(raw, environ, Path(agent_file).resolve().parents[3])
    except ValueError as exc:
        return str(exc)
    return ""


__all__ = [
    "ENV_RUNTIME_NODE_EXCLUSION",
    "NODE_ENTRY_POINT_GROUP",
    "ModelNodeExclusionLaneEnv",
    "known_contract_names",
    "node_exclusion_refusal",
    "parse_contract_name_list",
    "validate_node_exclusion",
]
