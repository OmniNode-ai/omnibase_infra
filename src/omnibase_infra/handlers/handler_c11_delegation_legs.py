# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Grade the C11 delegation legs (OMN-23177, scenario R9).

Four legs are graded over recorded observations against expectations declared in
``config/c11_delegation_legs.yaml``, never read off the observations:

* ``envelope_corpus``: each committed envelope (E1-E6 and E11 of the 2026-10-23
  eval) drew at least one terminal, every terminal carrying the declared refusal
  code and marked non-retryable.
* ``policy_refusal``: a fixed policy refusal on rung 1 makes exactly the declared
  number of attempts and never reaches rung 2 (K9).
* ``rung_503_control``: a 503 on rung 1 still moves; rung 2 is attempted and its
  answer accepted.
* ``canary``: the planted canary appears in no answer, receipt or event.

A leg with no observation is a failure, and a canary scan that saw no surface at
all is a failure, because "found nothing" over nothing proves nothing. The live
driver that records these observations from a lane is a later change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_c11_delegation_legs_request import (
    ModelC11DelegationLegsRequest,
)
from omnibase_infra.models.model_c11_delegation_legs_result import (
    ModelC11DelegationLegsResult,
)

PASS: Final[str] = "PASS"
FAIL: Final[str] = "FAIL"
COULD_NOT_RUN: Final[str] = "COULD_NOT_RUN"

RECORD_VERSION: Final[int] = 1


class ProbeInputError(RuntimeError):
    """The probe could not run: unreadable config, corpus or replay."""


@dataclass(frozen=True)
class LegsConfig:
    """The expectations of the legs, as ``config/c11_delegation_legs.yaml`` declares them."""

    corpus_path: Path
    refusal_failure_code: str
    corpus_ids: tuple[str, ...]
    policy_refusal_attempts: int
    rung_1: str
    rung_2: str
    canary_value: str
    canary_surfaces: tuple[str, ...]


@dataclass
class LegResult:
    name: str
    ok: bool
    reason: str
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "leg": self.name,
            "pass": self.ok,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, dict) else None


def _strings(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProbeInputError(f"{what} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ProbeInputError(f"{what} must hold non-empty strings")
    return tuple(value)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeInputError(f"{what} must be a non-empty string")
    return value


def load_legs_config(path: Path) -> LegsConfig:
    """The legs' declared expectations; unreadable or malformed is an input error."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProbeInputError(
            f"legs config {path} is unreadable: {type(exc).__name__}"
        ) from exc
    config = _mapping(document)
    if config is None:
        raise ProbeInputError("the legs config is not a mapping")
    ladder = _mapping(config.get("ladder"))
    if ladder is None:
        raise ProbeInputError("the legs config declares no ladder")
    attempts = config.get("policy_refusal_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ProbeInputError("policy_refusal_attempts must be a positive integer")
    rung_1 = _text(ladder.get("rung_1"), "ladder.rung_1")
    rung_2 = _text(ladder.get("rung_2"), "ladder.rung_2")
    if rung_1 == rung_2:
        raise ProbeInputError("the two ladder rungs must be distinct backends")
    return LegsConfig(
        corpus_path=Path(_text(config.get("corpus_path"), "corpus_path")),
        refusal_failure_code=_text(
            config.get("refusal_failure_code"), "refusal_failure_code"
        ),
        corpus_ids=_strings(config.get("corpus_ids"), "corpus_ids"),
        policy_refusal_attempts=attempts,
        rung_1=rung_1,
        rung_2=rung_2,
        canary_value=_text(config.get("canary_value"), "canary_value"),
        canary_surfaces=_strings(config.get("canary_surfaces"), "canary_surfaces"),
    )


def load_corpus(path: Path, config: LegsConfig) -> dict[str, Mapping[str, object]]:
    """The committed corpus keyed by envelope id; it must be exactly the declared set."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeInputError(
            f"the committed envelope corpus {path} is unreadable: {type(exc).__name__}"
        ) from exc
    rows = _mapping(document)
    envelopes = rows.get("envelopes") if rows is not None else None
    if not isinstance(envelopes, list):
        raise ProbeInputError("the corpus has no 'envelopes' list")
    corpus: dict[str, Mapping[str, object]] = {}
    for row in envelopes:
        entry = _mapping(row)
        ident = entry.get("id") if entry is not None else None
        envelope = _mapping(entry.get("envelope")) if entry is not None else None
        if not isinstance(ident, str) or not ident:
            raise ProbeInputError("a corpus row has no envelope id")
        if envelope is None:
            raise ProbeInputError(f"corpus envelope {ident} is not a JSON object")
        if ident in corpus:
            raise ProbeInputError(f"corpus envelope {ident} appears twice")
        corpus[ident] = envelope
    if sorted(corpus) != sorted(config.corpus_ids):
        raise ProbeInputError(
            f"the corpus holds {sorted(corpus)}, the grader demands "
            f"{sorted(config.corpus_ids)}"
        )
    return corpus


def grade_envelope_corpus(config: LegsConfig, observed: object) -> LegResult:
    """Every corpus envelope drew terminals, each the declared refusal, none retryable."""
    by_id = _mapping(observed)
    if by_id is None:
        return LegResult(
            "envelope_corpus", False, "no observation recorded for the envelope corpus"
        )
    problems: list[str] = []
    for ident in config.corpus_ids:
        entry = _mapping(by_id.get(ident))
        terminals = entry.get("terminals") if entry is not None else None
        if not isinstance(terminals, list) or not terminals:
            problems.append(f"{ident}: no terminal observed")
            continue
        for raw in terminals:
            terminal = _mapping(raw) or {}
            code = terminal.get("failure_code")
            if code != config.refusal_failure_code:
                problems.append(
                    f"{ident}: terminal code {code!r}, "
                    f"expected {config.refusal_failure_code!r}"
                )
                break
            if terminal.get("retryable") is not False:
                problems.append(f"{ident}: refusal is not marked non-retryable")
                break
    extra = sorted(set(by_id) - set(config.corpus_ids))
    if extra:
        problems.append(f"observations for envelopes outside the corpus: {extra}")
    if problems:
        return LegResult("envelope_corpus", False, "; ".join(problems))
    return LegResult(
        "envelope_corpus",
        True,
        f"{len(config.corpus_ids)} envelopes each refused {config.refusal_failure_code}",
        {"envelopes": list(config.corpus_ids)},
    )


def _attempts_of(observed: object) -> list[Mapping[str, object]] | None:
    entry = _mapping(observed)
    attempts = entry.get("attempts") if entry is not None else None
    if not isinstance(attempts, list) or not attempts:
        return None
    mappings = [_mapping(attempt) for attempt in attempts]
    if any(attempt is None for attempt in mappings):
        return None
    return [attempt for attempt in mappings if attempt is not None]


def grade_policy_refusal(config: LegsConfig, observed: object) -> LegResult:
    """A fixed policy refusal on rung 1 makes the declared attempts and never reaches rung 2."""
    attempts = _attempts_of(observed)
    if attempts is None:
        return LegResult(
            "policy_refusal", False, "no attempts recorded for the policy-refusal run"
        )
    backends = [attempt.get("backend_id") for attempt in attempts]
    if config.rung_2 in backends:
        return LegResult(
            "policy_refusal",
            False,
            f"the run climbed to {config.rung_2!r} after a policy refusal",
            {"backends": backends},
        )
    if len(attempts) != config.policy_refusal_attempts or set(backends) != {
        config.rung_1
    }:
        return LegResult(
            "policy_refusal",
            False,
            f"expected exactly {config.policy_refusal_attempts} attempt on "
            f"{config.rung_1!r}, observed {len(attempts)}: {backends}",
            {"backends": backends},
        )
    return LegResult(
        "policy_refusal",
        True,
        f"{config.policy_refusal_attempts} attempt on {config.rung_1!r} and none after it",
        {"backends": backends},
    )


def grade_rung_503_control(config: LegsConfig, observed: object) -> LegResult:
    """A 503 on rung 1 still moves: rung 2 is attempted and its answer accepted."""
    attempts = _attempts_of(observed)
    if attempts is None:
        return LegResult(
            "rung_503_control", False, "no attempts recorded for the 503 control run"
        )
    backends = [attempt.get("backend_id") for attempt in attempts]
    first, last = attempts[0], attempts[-1]
    if first.get("backend_id") != config.rung_1 or not first.get("failure_class"):
        return LegResult(
            "rung_503_control",
            False,
            f"the first attempt must be {config.rung_1!r} carrying a failure class",
            {"backends": backends},
        )
    if last.get("backend_id") != config.rung_2:
        return LegResult(
            "rung_503_control",
            False,
            f"the run did not move to {config.rung_2!r} after the 503",
            {"backends": backends},
        )
    if last.get("acceptance_decision") != "accept":
        return LegResult(
            "rung_503_control",
            False,
            f"{config.rung_2!r} was attempted and its answer was not accepted",
            {"backends": backends},
        )
    return LegResult(
        "rung_503_control",
        True,
        f"{config.rung_1!r} failed and the run moved to {config.rung_2!r}",
        {"backends": backends},
    )


def _surface_text(value: object) -> str:
    # An empty container is no observation, so it must not count as a surface
    # the scan read.
    if value is None or (isinstance(value, list | dict | str) and not value):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def grade_canary(config: LegsConfig, observed: object) -> LegResult:
    """The planted canary appears in no answer, receipt or event.

    Surfaces are read as text, so a canary nested in a receipt field or inside an
    event list is found the same way. A scan that saw no surface at all fails: the
    control that makes "not found" mean something is that a surface held content.
    """
    surfaces = _mapping(observed)
    if surfaces is None:
        return LegResult("canary", False, "no observation recorded for the canary run")
    texts = {name: _surface_text(surfaces.get(name)) for name in config.canary_surfaces}
    seen = [name for name, text in texts.items() if text.strip()]
    if not seen:
        return LegResult(
            "canary",
            False,
            "no answer, receipt or event was observed, so the scan proves nothing",
        )
    leaked = [name for name, text in texts.items() if config.canary_value in text]
    if leaked:
        return LegResult(
            "canary",
            False,
            f"the planted canary appears in: {', '.join(leaked)}",
            {"surfaces_with_canary": leaked, "surfaces_observed": seen},
        )
    return LegResult(
        "canary",
        True,
        "the planted canary appears in no answer, receipt or event",
        {"surfaces_observed": seen},
    )


def plant_canary_in_answer(
    config: LegsConfig, observations: Mapping[str, object]
) -> dict[str, object]:
    """A copy of ``observations`` whose canary leg answer carries the canary.

    The self-test: it proves the scan can fail, which is what gives its "not
    found" any meaning. The input is never mutated.
    """
    planted = dict(observations)
    leg = dict(_mapping(planted.get("canary")) or {})
    leg["answer"] = f"{_surface_text(leg.get('answer'))} {config.canary_value}".strip()
    planted["canary"] = leg
    return planted


def grade_legs(
    config: LegsConfig, observations: Mapping[str, object]
) -> list[LegResult]:
    """Grade the four delegation legs against the declared expectations."""
    return [
        grade_envelope_corpus(config, observations.get("envelope_corpus")),
        grade_policy_refusal(config, observations.get("policy_refusal")),
        grade_rung_503_control(config, observations.get("rung_503_control")),
        grade_canary(config, observations.get("canary")),
    ]


class HandlerC11DelegationLegs:
    """Grade the C11 delegation legs over a recorded observations file."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelC11DelegationLegsRequest
    ) -> ModelC11DelegationLegsResult:
        try:
            config = load_legs_config(Path(request.config_path))
            corpus = load_corpus(config.corpus_path, config)
            payload = _mapping(
                json.loads(Path(request.replay_path).read_text(encoding="utf-8"))
            )
            observations = _mapping(payload.get("observations") if payload else None)
            if observations is None:
                raise ProbeInputError("the replay file holds no 'observations' mapping")
        except (ProbeInputError, OSError, ValueError) as exc:
            message = f"C11 delegation legs could not run: {exc}"
            return ModelC11DelegationLegsResult(
                success=False, outcome=COULD_NOT_RUN, error_message=message
            )
        if request.plant_canary_in_answer:
            observations = plant_canary_in_answer(config, observations)
        legs = grade_legs(config, observations)
        failures = tuple(f"{leg.name}: {leg.reason}" for leg in legs if not leg.ok)
        outcome = FAIL if failures else PASS
        evidence: dict[str, object] = {
            "legs": [leg.to_dict() for leg in legs],
            "corpus_envelopes": sorted(corpus),
            "planted_canary": request.plant_canary_in_answer,
        }
        if request.record_path:
            Path(request.record_path).write_text(
                json.dumps(
                    {
                        "version": RECORD_VERSION,
                        "criterion": "C11",
                        "scenario": "R9",
                        "ticket": "OMN-23177",
                        "verdict": outcome.lower(),
                        "planted_canary": request.plant_canary_in_answer,
                        "failures": list(failures),
                        "legs": evidence["legs"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return ModelC11DelegationLegsResult(
            success=outcome == PASS,
            outcome=outcome,
            reasons=failures,
            evidence=evidence,
            error_message="; ".join(failures) if failures else None,
        )
