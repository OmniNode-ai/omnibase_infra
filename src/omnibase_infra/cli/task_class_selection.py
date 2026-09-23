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

THE RULES, each a contract field rather than an implementation detail:

* **Presence, never frequency.** A phrase occurs or it does not. Counting is
  what let the bulk of a document outvote its purpose.
* **Word boundaries.** No phrase matches inside a longer word.
* **Shape gates the keyword.** ``min_words`` / ``max_words`` are evaluated
  BEFORE any phrase, so a long prose task is structurally ineligible for the
  short keyword-driven classes whatever words it contains.
* **An ambiguous phrase is gated on its object** (OMN-18831). A phrase
  declared under ``qualified_phrases`` claims a prompt only where one of the
  class's declared qualifiers sits within ``within_words`` words of it, so
  "write a parser" is a code request and "write a PR body" is not. The
  vocabulary is the contract's; see ``ModelQualifiedPhrases``.
* **A short prompt is admitted by its opening** (OMN-19140). A class that
  declares ``short_prompt`` is eligible below its ``min_words``, down to the
  block's own floor, only for a prompt that opens with one of its
  ``opening_phrases``. "Summarize in one sentence: ..." reaches
  ``summarization``; a short prompt that merely mentions a summary does not.
  See ``ModelShortPromptSelection``.

Ties between eligible classes are broken by ``priority`` (higher wins) and then
by class name, so resolution is total and deterministic.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from omnibase_infra.cli.model_qualified_phrases import ModelQualifiedPhrases
from omnibase_infra.cli.model_selectable_task_class import ModelSelectableTaskClass
from omnibase_infra.cli.model_short_prompt_selection import (
    ModelShortPromptSelection,
)
from omnibase_infra.cli.model_task_class_execution_budget import (
    ModelTaskClassExecutionBudget,
)
from omnibase_infra.cli.model_task_type_resolution import ModelTaskTypeResolution
from omnibase_infra.enums.enum_task_type_resolution import EnumTaskTypeResolution

__all__ = [
    "EnumTaskTypeResolution",
    "ModelQualifiedPhrases",
    "ModelSelectableTaskClass",
    "ModelShortPromptSelection",
    "ModelTaskClassExecutionBudget",
    "ModelTaskTypeResolution",
    "TaskClassContractError",
    "DEFAULT_EXECUTION_BUDGET",
    "DEFAULT_TASK_TYPE",
    "load_selectable_task_classes",
    "load_selection_fallback",
    "resolve_task_class_execution_budget",
    "resolve_task_class_contract_path",
    "resolve_task_type",
]

logger = logging.getLogger(__name__)

#: The budget an absent ``execution_budgets`` map resolves to (OMN-18924).
#:
#: Not a number chosen here. OMN-15504 introduced the map and declared exactly
#: these values for all eleven public classes in both fixtures it added, so
#: this is the value that change intended, restated as the fallback rather
#: than left implicit in test data. A contract that declares the map overrides
#: it per class, so the day the producer lands this constant stops being read.
DEFAULT_EXECUTION_BUDGET = ModelTaskClassExecutionBudget(
    task_class_timeout_ceiling_seconds=240,
    terminal_delivery_margin_seconds=60,
)

#: Where the task-class contract sits inside the installed omnimarket package.
TASK_CLASS_CONTRACT_RELATIVE_PATH = Path("configs") / "task_class_contracts.v1.yaml"

#: The class a prompt resolves to when no declared predicate claims it, and the
#: contract declares no ``selection_fallback`` of its own.
#:
#: It must satisfy TWO properties, and the first revision of this module only
#: had the second:
#:
#: 1. **Its blocking floors must be shape-agnostic.** A prompt that reached the
#:    fallback is by construction a prompt whose SHAPE no predicate recognised,
#:    so grading it on a shape floor refuses correct answers for not being
#:    something nobody asked them to be. ``research`` — the original choice —
#:    carries ``cites_sources`` and ``methodical_analysis``, two shape floors,
#:    and that is exactly what refused a 170-word drafting prompt on every rung
#:    on 2026-09-15 (ledger ``docs/tracking/ROLLING_WORK_LEDGER.md:8210``).
#:    ``document``'s floors are ``no_refusal``, ``accurate`` and
#:    ``semantic_adequacy``: three qualities any well-formed prose answer can
#:    meet, and no shape at all.
#: 2. **It must still arm the prose checks**, ``identifiers_grounded`` included
#:    (OMN-18297). An unrecognised prompt must not land somewhere ungraded,
#:    which is the failure OMN-18305 was opened for. ``document`` arms them.
#:
#: Both properties are pinned by ``TestTheFallbackIsPermissiveNotStrict``, with
#: a positive control asserting ``research`` and ``planning`` fail the first.
DEFAULT_TASK_TYPE = "document"


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
        try:
            selectable.append(
                ModelSelectableTaskClass(
                    name=str(name),
                    priority=int(selection["priority"]),
                    phrases=tuple(
                        str(phrase) for phrase in selection.get("phrases") or ()
                    ),
                    min_words=selection.get("min_words"),
                    max_words=selection.get("max_words"),
                    qualified_phrases=selection.get("qualified_phrases"),
                    short_prompt=selection.get("short_prompt"),
                )
            )
        except ValidationError as exc:
            raise TaskClassContractError(
                f"task class {name!r} declares an invalid selection predicate in "
                f"{contract_path}: {exc}"
            ) from exc
    if not selectable:
        raise TaskClassContractError(
            f"task-class contract at {contract_path} exposes no public class"
        )
    return tuple(selectable)


def resolve_task_class_execution_budget(
    contract_path: Path,
    *,
    task_type: str,
) -> ModelTaskClassExecutionBudget:
    """Read the selected task class's declared execution budget.

    The CLI already resolves this packaged contract for task-class selection.
    Re-reading that same authority is intentional: the consumer's wall-clock
    deadline must not be a second, stale CLI default.
    """
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskClassContractError(
            f"task-class contract at {contract_path} could not be read: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TaskClassContractError(
            f"task-class contract at {contract_path} is not a mapping"
        )
    if "execution_budgets" not in raw:
        # OMN-18924. ABSENT is not malformed. No merged omnimarket branch has
        # ever declared this map -- its producer is still an open pull request
        # -- so failing closed here refused EVERY delegation on a host whose
        # dispatch venv installs this package by local path, before any
        # dispatch. A new contract field lands consumer-first or is excluded
        # when unset; making an un-defaulted field mandatory inverts that and
        # takes out every caller rather than the one feature it serves.
        #
        # The default is not invented here: it is the value the introducing
        # change's own fixtures declare for all eleven public classes. Logged
        # rather than silent, so a defaulted budget is never read back as a
        # declared one, and so this line disappears from the logs on the day
        # the producer lands.
        logger.info(
            "task-class contract at %s declares no execution_budgets map; "
            "using the default budget (%ss ceiling, %ss delivery margin) "
            "until the contract declares one",
            contract_path,
            DEFAULT_EXECUTION_BUDGET.task_class_timeout_ceiling_seconds,
            DEFAULT_EXECUTION_BUDGET.terminal_delivery_margin_seconds,
        )
        return DEFAULT_EXECUTION_BUDGET
    budgets = raw["execution_budgets"]
    if not isinstance(budgets, dict):
        # PRESENT and malformed. A contract that states a budget and states it
        # wrongly is a defect, and this refusal is kept exactly as OMN-15504
        # wrote it.
        raise TaskClassContractError(
            f"task-class contract at {contract_path} declares no execution_budgets map"
        )
    declared = budgets.get(task_type)
    if not isinstance(declared, dict):
        raise TaskClassContractError(
            f"task class {task_type!r} declares no execution budget in {contract_path}"
        )
    try:
        return ModelTaskClassExecutionBudget.model_validate(declared)
    except ValueError as exc:
        raise TaskClassContractError(
            f"task class {task_type!r} has an invalid execution budget in "
            f"{contract_path}: {exc}"
        ) from exc


def load_selection_fallback(contract_path: Path) -> str:
    """Return the class an unclaimed prompt falls back to, as the contract declares it.

    The fallback is a decision about GRADING, so it belongs beside the grading
    rules rather than in this evaluator: moving it must not require a CLI
    release. A contract that declares ``selection_fallback.task_class`` owns the
    choice; one that does not yields :data:`DEFAULT_TASK_TYPE`, whose docstring
    records the two properties any fallback has to satisfy.

    A declared fallback naming a class the contract does not gateway-expose is a
    contract defect and is refused by name. Silently reverting to the module
    constant there would hide the drift the declaration exists to prevent.
    """
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskClassContractError(
            f"task-class contract at {contract_path} could not be read: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TaskClassContractError(
            f"task-class contract at {contract_path} is not a mapping"
        )

    declared = raw.get("selection_fallback")
    if declared is None:
        return DEFAULT_TASK_TYPE
    if not isinstance(declared, dict) or not declared.get("task_class"):
        raise TaskClassContractError(
            f"task-class contract at {contract_path} declares a "
            "selection_fallback with no task_class"
        )

    name = str(declared["task_class"])
    public = {
        str(key)
        for key, entry in (raw.get("task_classes") or {}).items()
        if isinstance(entry, dict) and entry.get("gateway_exposure") == "public"
    }
    if name not in public:
        raise TaskClassContractError(
            f"selection_fallback names {name!r}, which the contract at "
            f"{contract_path} does not gateway-expose; a prompt can never be "
            f"routed there. Exposed: {', '.join(sorted(public)) or '(none)'}"
        )
    return name


def resolve_task_type(
    prompt: str,
    *,
    explicit: str | None,
    classes: tuple[ModelSelectableTaskClass, ...],
    fallback: str = DEFAULT_TASK_TYPE,
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
    eligible: list[tuple[ModelSelectableTaskClass, str, bool]] = []
    for entry in classes:
        if entry.shape_admits(word_count):
            phrase = entry.matching_phrase(lowered)
            if phrase is not None:
                eligible.append((entry, phrase, False))
            continue
        # OMN-19140: below a class's floor, only a declared opening phrase at
        # the very start of the prompt can admit it.
        opening = entry.short_prompt_phrase(lowered, word_count)
        if opening is not None:
            eligible.append((entry, opening, True))

    if not eligible:
        return ModelTaskTypeResolution(
            task_type=fallback,
            resolution=EnumTaskTypeResolution.FALLBACK,
            reason=(
                f"no declared selection predicate claimed this "
                f"{word_count}-word prompt; using the declared fallback "
                f"{fallback!r}, whose quality floors are shape-agnostic. Pass "
                f"--criteria to state your own acceptance criteria instead"
            ),
        )

    # Highest priority wins; an exact tie is broken by class name so the
    # resolution is total and reproducible rather than map-order dependent.
    winner, phrase, opens = min(
        eligible, key=lambda item: (-item[0].priority, item[0].name)
    )
    how = (
        f"opening phrase {phrase!r} at the start of a {word_count}-word prompt"
        if opens
        else f"phrase {phrase!r} in a {word_count}-word prompt"
    )
    return ModelTaskTypeResolution(
        task_type=winner.name,
        resolution=EnumTaskTypeResolution.CONTRACT,
        reason=(
            f"contract predicate for {winner.name!r} (priority {winner.priority}) "
            f"matched the {how}"
        ),
    )
