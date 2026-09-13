# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-declared task-class selection for ``onex delegate`` (OMN-18305).

WHAT THIS REPLACES. The CLI used to carry ``_CLASSIFICATION_RULES``: an ordered
keyword table lifted verbatim from retired ``delegate/prompt.md`` skill
markdown, matched with a bare ``in`` test over the lowercased prompt, first
rule wins, ``test`` rule first. It was not derived from any contract, and
nothing compared it to one. Two measured consequences on 2026-09-13:

* a 56,593-byte engineering standup classified as ``test``, because the ledger
  rows it summarised contain the word "test" 46 times. ``test`` is not one of
  the nine classes whose quality bar arms ``identifiers_grounded``, so the
  OMN-18297 grounding check and the prose quality band never ran on the
  customer's answer, and the customer was never told a class had been chosen
  for them;
* ``"the latest window"`` classified as ``test``, because "latest" contains
  "test". There was no word-boundary matching anywhere in the table.

WHERE SELECTION LIVES NOW. Each task class declares its own predicate in
omnimarket's ``configs/task_class_contracts.v1.yaml`` (``ModelTaskClassSelection``
in ``omnimarket.inference.task_class_authority``), and this module reads it.

WHY THE EVALUATOR IS HERE AND THE DECLARATION IS THERE. Repo layering runs
compat -> core -> spi -> infra, and omnimarket depends on omnibase_infra, never
the reverse: this package cannot import omnimarket. It already resolves
omnimarket's PACKAGED delegate contract at runtime (``_resolve_packaged_contract``)
and already guards against a drifted omnimarket co-install, so resolving the
packaged task-class contract the same way is the established seam, not a new
dependency. The declaration is the contract's; the evaluation is the caller's.

THE THREE RULES, each a contract field rather than an implementation detail:

* **Presence, never frequency.** A phrase occurs or it does not. Counting is
  what let the bulk of a document outvote its purpose.
* **Word boundaries.** No phrase matches inside a longer word.
* **Shape gates the keyword.** ``min_words`` / ``max_words`` are evaluated
  BEFORE any phrase, so a long prose task is structurally ineligible for the
  short keyword-driven classes whatever words it contains.

Ties between eligible classes are broken by ``priority`` (higher wins) and then
by class name, so resolution is total and deterministic.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

from omnibase_infra.cli.model_selectable_task_class import ModelSelectableTaskClass
from omnibase_infra.cli.model_task_type_resolution import ModelTaskTypeResolution
from omnibase_infra.enums.enum_task_type_resolution import EnumTaskTypeResolution

__all__ = [
    "EnumTaskTypeResolution",
    "ModelSelectableTaskClass",
    "ModelTaskTypeResolution",
    "TaskClassContractError",
    "load_selectable_task_classes",
    "resolve_task_class_contract_path",
    "resolve_task_type",
]

#: Where the task-class contract sits inside the installed omnimarket package.
TASK_CLASS_CONTRACT_RELATIVE_PATH = Path("configs") / "task_class_contracts.v1.yaml"

#: The class a prompt resolves to when no declared predicate claims it. It is
#: deliberately one of the classes whose quality bar arms the prose checks: an
#: unrecognised prompt must not land somewhere the checks are disarmed, which
#: is precisely the failure OMN-18305 was opened for.
DEFAULT_TASK_TYPE = "research"


class TaskClassContractError(Exception):
    """The task-class contract is absent, unreadable, or internally invalid.

    Raised rather than defaulted. A silent default here reintroduces the exact
    class of defect this module replaces — a classification nobody declared.
    """


def resolve_task_class_contract_path() -> Path:
    """Locate the packaged task-class contract inside the installed omnimarket.

    Resolved from the installed package's own location, exactly as omnimarket's
    own loaders do, so it is correct whether omnimarket is a wheel or an
    editable checkout. Raises rather than guessing a path.
    """
    spec = importlib.util.find_spec("omnimarket")
    if spec is None or not spec.origin:
        raise TaskClassContractError(
            "omnimarket is not installed in this environment, so the task-class "
            "contract that declares delegate task classes cannot be resolved. "
            "Pass --task-type explicitly, or repair the co-install."
        )
    path = Path(spec.origin).resolve().parent / TASK_CLASS_CONTRACT_RELATIVE_PATH
    if not path.is_file():
        raise TaskClassContractError(
            f"task-class contract not found at {path}; the installed omnimarket "
            "does not ship it."
        )
    return path


def load_selectable_task_classes(
    contract_path: Path,
) -> tuple[ModelSelectableTaskClass, ...]:
    """Return every gateway-exposed class with its declared selection predicate.

    The CLI's selectable vocabulary IS this projection — there is no second
    list to drift against it. A public class that declares no predicate is a
    contract defect and fails closed here; it is never given a default.
    """
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskClassContractError(
            f"task-class contract at {contract_path} could not be read: {exc}"
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("task_classes"), dict):
        raise TaskClassContractError(
            f"task-class contract at {contract_path} declares no task_classes map"
        )

    selectable: list[ModelSelectableTaskClass] = []
    for name, entry in raw["task_classes"].items():
        if not isinstance(entry, dict) or entry.get("gateway_exposure") != "public":
            continue
        selection = entry.get("selection")
        if not isinstance(selection, dict):
            raise TaskClassContractError(
                f"task class {name!r} is gateway-exposed but declares no "
                "selection predicate; a class the CLI can select must say how "
                "a prompt selects it"
            )
        selectable.append(
            ModelSelectableTaskClass(
                name=str(name),
                priority=int(selection["priority"]),
                phrases=tuple(str(phrase) for phrase in selection.get("phrases") or ()),
                min_words=selection.get("min_words"),
                max_words=selection.get("max_words"),
            )
        )
    if not selectable:
        raise TaskClassContractError(
            f"task-class contract at {contract_path} exposes no public class"
        )
    return tuple(selectable)


def resolve_task_type(
    prompt: str,
    *,
    explicit: str | None,
    classes: tuple[ModelSelectableTaskClass, ...],
) -> ModelTaskTypeResolution:
    """Resolve this run's task class and record how the decision was made.

    An explicit class always wins and is validated against the contract's own
    vocabulary, so a class the contract does not declare is refused by name
    rather than dispatched and rejected downstream.
    """
    vocabulary = sorted(entry.name for entry in classes)
    if explicit is not None:
        if explicit not in set(vocabulary):
            raise TaskClassContractError(
                f"unknown task type {explicit!r}; the task-class contract "
                f"exposes: {', '.join(vocabulary)}"
            )
        return ModelTaskTypeResolution(
            task_type=explicit,
            resolution=EnumTaskTypeResolution.EXPLICIT,
            reason="explicitly selected with --task-type",
        )

    lowered = prompt.lower()
    word_count = len(prompt.split())
    eligible: list[tuple[ModelSelectableTaskClass, str]] = []
    for entry in classes:
        if not entry.shape_admits(word_count):
            continue
        phrase = entry.matching_phrase(lowered)
        if phrase is not None:
            eligible.append((entry, phrase))

    if not eligible:
        return ModelTaskTypeResolution(
            task_type=DEFAULT_TASK_TYPE,
            resolution=EnumTaskTypeResolution.FALLBACK,
            reason=(
                f"no declared selection predicate claimed this "
                f"{word_count}-word prompt; using the declared fallback "
                f"{DEFAULT_TASK_TYPE!r}"
            ),
        )

    # Highest priority wins; an exact tie is broken by class name so the
    # resolution is total and reproducible rather than map-order dependent.
    winner, phrase = min(eligible, key=lambda pair: (-pair[0].priority, pair[0].name))
    return ModelTaskTypeResolution(
        task_type=winner.name,
        resolution=EnumTaskTypeResolution.CONTRACT,
        reason=(
            f"contract predicate for {winner.name!r} (priority {winner.priority}) "
            f"matched the phrase {phrase!r} in a {word_count}-word prompt"
        ),
    )
