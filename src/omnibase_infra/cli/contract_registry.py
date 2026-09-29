# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``onex`` CLI's one read path to the installed contract registry (OMN-19407).

Operator ruling, 2026-09-24: the CLI keeps no list of its own. Every vocabulary
a command offers, validates or classifies against is read from the contract
that owns it, when the command runs, and a registry that cannot be read is a
refusal that names what was missing. There is no fallback list.

WHAT "THE REGISTRY" IS HERE. The installed distributions' entry points:

* ``onex.nodes`` maps a node name to its package, whose ``contract.yaml``
  names the node's ``input_model`` and ``output_model``. A field's closed
  vocabulary is read off those models (a ``Literal`` field's arguments), so a
  flag offering it cannot drift from what the node accepts.
* ``onex.contracts`` maps a config-contract id to a zero-argument loader that
  returns the typed, validated contract (``task_class_authority`` is
  omnimarket's task-class contract).

WHY THIS IS NOT AN IMPORT OF OMNIMARKET. Repo layering forbids omnibase_infra
from importing omnimarket at module scope, and that was the stated reason the
CLI kept hand-copied lists. It was never a reason: resolving an entry point at
run time is how this CLI already found node contracts, and it is not a static
dependency. Nothing here names an omnimarket module.

THE ONE SEAM. :func:`_entry_points` is the only function that reads installed
metadata. Tests replace it with stand-in entries that carry a stand-in
contract and model, then assert that the CLI offers whatever those declare --
a property of the read path, not a pinned list.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import click
import yaml
from pydantic import BaseModel

__all__ = [
    "CONTRACT_RESOURCE_GROUP",
    "NODE_GROUP",
    "ContractChoice",
    "RegistryUnresolvedError",
    "contract_resource",
    "field_default",
    "field_vocabulary",
    "node_contract",
    "node_contract_path",
    "node_input_model",
]

#: Entry-point group mapping a node name to its package (and its contract).
NODE_GROUP = "onex.nodes"
#: Entry-point group mapping a config-contract id to its loader.
CONTRACT_RESOURCE_GROUP = "onex.contracts"


class RegistryUnresolvedError(Exception):
    """A registry entry the CLI needs could not be read.

    The message names the entry-point group, the entry and what was wrong, so
    the remedy (install or repair the distribution that advertises it) is on
    the screen. Never caught and replaced with a default.
    """


def _entry_points(group: str) -> tuple[EntryPoint, ...]:
    """Return every installed entry point in ``group``. The single metadata read."""
    return tuple(entry_points(group=group))


def _single_entry(group: str, name: str) -> EntryPoint:
    matches = [entry for entry in _entry_points(group) if entry.name == name]
    if not matches:
        known = sorted({entry.name for entry in _entry_points(group)})
        raise RegistryUnresolvedError(
            f"no installed distribution advertises {name!r} in the {group!r} "
            f"entry-point group, so the contract that owns it cannot be read. "
            f"Advertised: {', '.join(known) or '(none)'}"
        )
    if len(matches) > 1:
        sources = ", ".join(str(entry.dist) for entry in matches)
        raise RegistryUnresolvedError(
            f"{name!r} is advertised in {group!r} by more than one distribution "
            f"({sources}); uninstall the conflicting one"
        )
    return matches[0]


def node_contract_path(node_name: str) -> Path:
    """Return the packaged ``contract.yaml`` of the node registered as ``node_name``.

    Packaged nodes colocate their contract at ``<module_dir>/contract.yaml``.
    A node that breaks that convention is refused by name rather than guessed.
    """
    entry = _single_entry(NODE_GROUP, node_name)
    module_path = entry.value.split(":", 1)[0].strip()
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise RegistryUnresolvedError(
            f"node {node_name!r} is advertised as module {module_path!r}, which "
            "cannot be found in this environment"
        )
    if spec.submodule_search_locations:
        module_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin is not None:
        module_dir = Path(spec.origin).resolve().parent
    else:
        raise RegistryUnresolvedError(
            f"node {node_name!r} module {module_path!r} has no origin, so its "
            "packaged contract.yaml cannot be located"
        )
    contract = module_dir / "contract.yaml"
    if not contract.is_file():
        raise RegistryUnresolvedError(
            f"node {node_name!r} resolved to {module_dir}, which has no "
            "contract.yaml (the packaging convention colocates it)"
        )
    return contract


def node_contract(node_name: str) -> dict[str, object]:
    """Return the parsed ``contract.yaml`` of the node registered as ``node_name``."""
    path = node_contract_path(node_name)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryUnresolvedError(
            f"contract of node {node_name!r} at {path} could not be read: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RegistryUnresolvedError(
            f"contract of node {node_name!r} at {path} is not a mapping"
        )
    return {str(key): value for key, value in raw.items()}


def node_input_model(node_name: str) -> type[BaseModel]:
    """Return the request model the node's contract declares as ``input_model``.

    Loaded by the module and class the contract names, at run time. That is a
    registry read, not a static import of the package that defines it.
    """
    return _node_model(node_name, "input_model")


def _node_model(node_name: str, role: str) -> type[BaseModel]:
    declared = node_contract(node_name).get(role)
    if not isinstance(declared, dict):
        raise RegistryUnresolvedError(
            f"contract of node {node_name!r} declares no {role}"
        )
    module_name = declared.get("module")
    class_name = declared.get("class") or declared.get("name")
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        raise RegistryUnresolvedError(
            f"contract of node {node_name!r} declares a {role} without a module "
            "and class"
        )
    try:
        model = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise RegistryUnresolvedError(
            f"{role} {module_name}.{class_name} declared by node {node_name!r} "
            f"cannot be loaded: {exc}"
        ) from exc
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise RegistryUnresolvedError(
            f"{role} {module_name}.{class_name} declared by node {node_name!r} "
            "is not a pydantic model"
        )
    return model


def _literal_arguments(annotation: object) -> tuple[object, ...] | None:
    """Return a ``Literal``'s arguments, looking through an alias or ``X | None``."""
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin in (Union, UnionType):
        found = [
            _literal_arguments(member)
            for member in get_args(annotation)
            if member is not type(None)
        ]
        literals = [values for values in found if values is not None]
        if len(literals) == 1:
            return literals[0]
    value = getattr(annotation, "__value__", None)
    if value is not None:
        return _literal_arguments(value)
    return None


def field_vocabulary(model: type[BaseModel], field: str) -> tuple[str, ...]:
    """Return the closed vocabulary ``model`` declares for ``field``, in declared order.

    The field must be typed as a ``Literal`` of strings. Anything else is
    refused by name: a free-text field has no vocabulary for a flag to offer.
    """
    declared = model.model_fields.get(field)
    if declared is None:
        raise RegistryUnresolvedError(
            f"{model.__module__}.{model.__qualname__} declares no field {field!r}"
        )
    values = _literal_arguments(declared.annotation)
    if not values or not all(isinstance(value, str) for value in values):
        raise RegistryUnresolvedError(
            f"{model.__module__}.{model.__qualname__}.{field} is not a closed "
            "vocabulary of strings"
        )
    return tuple(str(value) for value in values)


def field_default(model: type[BaseModel], field: str) -> object:
    """Return the default ``model`` declares for ``field``, or refuse if it has none."""
    declared = model.model_fields.get(field)
    if declared is None or declared.is_required():
        raise RegistryUnresolvedError(
            f"{model.__module__}.{model.__qualname__}.{field} declares no default"
        )
    return declared.get_default(call_default_factory=True)


def contract_resource(resource_id: str) -> object:
    """Load the config contract advertised as ``resource_id`` in ``onex.contracts``.

    The advertised object is a zero-argument loader, called here; its result
    is the typed contract. A loader that raises is reported as unresolved with
    its own message.
    """
    entry = _single_entry(CONTRACT_RESOURCE_GROUP, resource_id)
    try:
        loader = entry.load()
    except (ImportError, AttributeError) as exc:
        raise RegistryUnresolvedError(
            f"{CONTRACT_RESOURCE_GROUP}:{resource_id} ({entry.value}) cannot be "
            f"loaded: {exc}"
        ) from exc
    if not callable(loader):
        raise RegistryUnresolvedError(
            f"{CONTRACT_RESOURCE_GROUP}:{resource_id} ({entry.value}) is not a loader"
        )
    try:
        return loader()
    except (OSError, ValueError) as exc:
        raise RegistryUnresolvedError(
            f"{CONTRACT_RESOURCE_GROUP}:{resource_id} ({entry.value}) could not "
            f"load its contract: {exc}"
        ) from exc


class ContractChoice(click.Choice[str]):
    """``click.Choice`` whose values are read from the registry each time they are used.

    Importing a command reads nothing, so ``onex --help`` never touches the
    registry. Help, completion and validation call ``load``; a registry that
    cannot be read becomes a usage error naming it, never an empty or
    remembered list. Nothing is cached, so no process can hold a stale copy.
    """

    def __init__(self, source: str, load: Callable[[], Sequence[str]]) -> None:
        self._source = source
        self._load = load
        super().__init__((), case_sensitive=True)

    @property  # type: ignore[override]
    def choices(self) -> Sequence[str]:
        """Return the values ``load`` reads from the registry now."""
        try:
            values = tuple(self._load())
        except RegistryUnresolvedError as exc:
            raise click.UsageError(
                f"{self._source} could not be read from the contract registry: {exc}"
            ) from exc
        if not values:
            raise click.UsageError(f"{self._source} declares no values")
        return values

    @choices.setter
    def choices(self, value: Sequence[str]) -> None:
        """Refuse any list but the empty one ``click.Choice.__init__`` assigns.

        The loader is the only source; a literal list handed to this type
        would be the copy it exists to remove.
        """
        if tuple(value):
            raise TypeError(
                f"{type(self).__name__} reads {self._source} from the registry; "
                "it does not accept a literal list"
            )
