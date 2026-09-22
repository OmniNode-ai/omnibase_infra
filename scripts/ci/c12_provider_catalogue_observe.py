# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19195 -- the C12 OBSERVER. Runs INSIDE the deployed onex-api container.

WHAT THIS IS
    The half of the C12 producer that executes in the deployed interpreter,
    against the deployed ``omnimarket`` package and the deployed ``onex-api``
    router. ``c12_provider_catalogue_probe.py`` pipes this file to
    ``docker exec -i -u appuser <container> python -`` and grades what it
    prints. This file GRADES NOTHING: it reports what the deployed code
    returned, and the verdict is decided on the runner, in a grader that is
    tested offline against recorded observations. A program that both runs the
    subject and decides whether it passed is a grader grading itself.

WHY INSIDE THE CONTAINER AND NOT OVER HTTP
    No customer-visible HTTP rendering of the catalogue exists: the intake
    route answers 401 before its body is validated, and its OpenAPI schema
    carries only a charset regex. The 2026-09-19 hand readback recorded the
    same limit. The strongest reading available is therefore the deployed
    package through the deployed interpreter, plus the request model the
    deployed route actually binds -- which is what this reads.

IT WRITES NOTHING
    Every injection below is an in-memory copy of a shipped row handed to a
    PURE deployed function (``catalogue_parity_gap``,
    ``find_house_keyed_catalogue_entries``, the request model's validator). No
    file is written, no catalogue is reloaded from a rewritten path, no route
    is called and no event is published. Constructing the request model runs
    only its field validators; ``register_inference_credential`` is never
    reached.

OUTPUT
    Exactly one JSON object on stdout, last line. An import or read failure is
    reported as ``{"error": ...}`` rather than raised, so the runner can tell
    "the subject could not be read" (exit 2) from "the subject is wrong"
    (exit 1).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

# The customer route this criterion's catalogue is bound to. Declared here and
# echoed back so the grader checks the observer looked at the right one.
ROUTE_PREFIX = "/v1/tenants/me/inference-credentials"

# A provider id no catalogue offers and no platform rung backs. Used as the
# unbacked injection and as a refused-at-intake case. OMN-17373 records it as
# deliberately absent.
UNBACKED_PROVIDER = "openai"

# Claude ids the intake model must refuse. Two spellings, because the deployed
# pattern is a case-insensitive substring match and a regression to an exact
# match would pass the first and not the second.
CLAUDE_PROVIDERS = ("anthropic", "Claude-3")

# The synthetic provider used for the un-registerable-key injection. It is
# synthetic ON PURPOSE: a real one exists today (vertex), but the probe must
# still be able to exercise the conjunct on the day vertex is removed, and an
# injection that stops being constructible is a negative test that silently
# stops running.
SYNTHETIC_TOKEN_PROVIDER = "c12probe"
SYNTHETIC_TOKEN_RUNG = {
    "backend_id": "c12-probe-token-rung",
    "secret_ref": f"llm.{SYNTHETIC_TOKEN_PROVIDER}.access_token",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exc(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:600]}


def _string_values(row: Any) -> list[str]:
    """Every string VALUE anywhere in a row, keys excluded."""
    out: list[str] = []
    if isinstance(row, dict):
        for value in row.values():
            out.extend(_string_values(value))
    elif isinstance(row, list):
        for value in row:
            out.extend(_string_values(value))
    elif isinstance(row, str):
        out.append(row)
    return out


def _claude_hits(pattern: Any, rows: list[dict[str, Any]]) -> list[str]:
    """The Claude scan. The SAME function runs on the shipped rows and on the
    injected row, so the injection is the scan's positive control."""
    hits: list[str] = []
    for row in rows:
        for value in _string_values(row):
            if pattern.search(value):
                hits.append(value)
    return hits


def _findings(fn: Any, rows: list[dict[str, Any]], rungs: list[dict[str, Any]]) -> Any:
    try:
        found = fn(rows, rungs)
    except Exception as exc:  # noqa: BLE001 - the exception IS the observation
        return {"raised": _exc(exc)}
    return {
        "findings": [
            {"provider": f.provider, "finding_class": f.finding_class} for f in found
        ]
    }


def _intake(model: Any, provider: str) -> dict[str, Any]:
    try:
        model(name="c12-probe", provider=provider, key_value="c12-probe-not-a-key")
    except Exception as exc:  # noqa: BLE001 - the refusal IS the observation
        errors: list[dict[str, Any]] = []
        raw_errors = getattr(exc, "errors", None)
        if callable(raw_errors):
            for err in raw_errors():
                errors.append(
                    {
                        "loc": [str(p) for p in err.get("loc", ())],
                        "type": str(err.get("type", "")),
                        "msg": str(err.get("msg", ""))[:400],
                    }
                )
        return {"accepted": False, "exception": type(exc).__name__, "errors": errors}
    return {"accepted": True}


def observe(app_root: str) -> dict[str, Any]:
    if app_root and app_root not in sys.path:
        sys.path.insert(0, app_root)

    from omnimarket.projection.credential_publisher import (
        ModelInferenceCredentialCreateRequest,
    )
    from omnimarket.routing import byok_provider_backends as byok
    from omnimarket.validators import byok_catalogue_no_house_entry as house

    obs: dict[str, Any] = {
        "omnimarket_version": importlib.metadata.version("omnimarket"),
        "files": {
            "catalogue": _sha256(Path(byok.CATALOG_PATH)),
            "platform_contract": _sha256(Path(house.BIFROST_CONTRACT_PATH)),
            "catalogue_module": _sha256(Path(byok.__file__)),
            "house_validator": _sha256(Path(house.__file__)),
        },
        "forbidden_pattern": byok.FORBIDDEN_PROVIDER_PATTERN.pattern,
    }

    # ---- the shipped reading ------------------------------------------------
    offered = list(byok.customer_provider_catalogue())
    not_offered = sorted(byok.load_byok_not_offered_providers())
    rows = house.read_catalogue_rows()
    rungs = house.read_platform_rungs()
    house_slugs = sorted(byok.house_keyed_provider_slugs(rungs))
    gap = byok.catalogue_parity_gap(
        house_slugs, offered=offered, not_offered=not_offered
    )
    obs["shipped"] = {
        "offered": offered,
        "not_offered": not_offered,
        "rung_count": len(rungs),
        "house_keyed_slugs": house_slugs,
        "parity_gap": {
            "missing_from_catalogue": list(gap.missing_from_catalogue),
            "unbacked_in_catalogue": list(gap.unbacked_in_catalogue),
        },
        "claude_hits": _claude_hits(
            byok.FORBIDDEN_PROVIDER_PATTERN,
            rows + [{"provider": p} for p in not_offered],
        ),
        "house_entry": _findings(house.find_house_keyed_catalogue_entries, rows, rungs),
    }

    # ---- the customer surface ----------------------------------------------
    from routers import inference_credentials as route_mod

    body_models: list[str] = []
    bound = False
    for route in route_mod.router.routes:
        methods = getattr(route, "methods", set()) or set()
        if getattr(route, "path", None) != ROUTE_PREFIX or "POST" not in methods:
            continue
        # The RESOLVED annotation FastAPI validates the body against. The
        # endpoint's own ``__annotations__`` are strings under postponed
        # evaluation and would compare unequal to the class they name.
        dependant = getattr(route, "dependant", None)
        for param in getattr(dependant, "body_params", None) or ():
            candidate = getattr(getattr(param, "field_info", None), "annotation", None)
            if candidate is None:
                continue
            body_models.append(
                f"{getattr(candidate, '__module__', '?')}."
                f"{getattr(candidate, '__qualname__', str(candidate))}"
            )
            if candidate is ModelInferenceCredentialCreateRequest:
                bound = True
    obs["route"] = {
        "prefix": ROUTE_PREFIX,
        "body_models": body_models,
        "bound_to_catalogue_model": bound,
    }
    obs["intake"] = {
        p: _intake(ModelInferenceCredentialCreateRequest, p)
        for p in [*offered, *not_offered, UNBACKED_PROVIDER, *CLAUDE_PROVIDERS]
    }

    # ---- the negative test: each clause shown to bite, same process ---------
    neg: dict[str, Any] = {}

    g = byok.catalogue_parity_gap(
        house_slugs, offered=[*offered, UNBACKED_PROVIDER], not_offered=not_offered
    )
    neg["parity_unbacked"] = {
        "injected": UNBACKED_PROVIDER,
        "unbacked_in_catalogue": list(g.unbacked_in_catalogue),
        "missing_from_catalogue": list(g.missing_from_catalogue),
    }

    declared = sorted({*offered, *not_offered})
    dropped = declared[0] if declared else None
    g = byok.catalogue_parity_gap(
        house_slugs,
        offered=[p for p in offered if p != dropped],
        not_offered=[p for p in not_offered if p != dropped],
    )
    neg["parity_missing"] = {
        "dropped": dropped,
        "unbacked_in_catalogue": list(g.unbacked_in_catalogue),
        "missing_from_catalogue": list(g.missing_from_catalogue),
    }

    claude_row = copy.deepcopy(rows[0]) if rows else {}
    claude_row["provider"] = CLAUDE_PROVIDERS[0]
    neg["claude_row"] = {
        "injected": CLAUDE_PROVIDERS[0],
        "claude_hits": _claude_hits(
            byok.FORBIDDEN_PROVIDER_PATTERN, [*rows, claude_row]
        ),
    }

    # House-entry injections are built from REAL shipped values: the offered
    # row and a house rung that shares its slug.
    target_row = None
    target_rung = None
    for row in rows:
        for rung in rungs:
            ref = rung.get("secret_ref")
            if (
                isinstance(ref, str)
                and ref.startswith(f"llm.{row.get('provider')}.")
                and isinstance(rung.get("backend_id"), str)
            ):
                target_row, target_rung = row, rung
                break
        if target_row is not None:
            break

    if target_row is not None and target_rung is not None:
        laundered = copy.deepcopy(target_row)
        laundered["model_name"] = target_rung["secret_ref"]
        neg["house_declared_ref"] = {
            "provider": target_row["provider"],
            "value": target_rung["secret_ref"],
            **_findings(
                house.find_house_keyed_catalogue_entries,
                [laundered if r is target_row else r for r in rows],
                rungs,
            ),
        }
        collided = copy.deepcopy(target_row)
        collided["backend_id"] = target_rung["backend_id"]
        neg["house_rung_collision"] = {
            "provider": target_row["provider"],
            "value": target_rung["backend_id"],
            **_findings(
                house.find_house_keyed_catalogue_entries,
                [collided if r is target_row else r for r in rows],
                rungs,
            ),
        }
    else:
        neg["house_declared_ref"] = {"not_constructible": True}
        neg["house_rung_collision"] = {"not_constructible": True}

    token_row = copy.deepcopy(rows[0]) if rows else {}
    token_row["provider"] = SYNTHETIC_TOKEN_PROVIDER
    token_row["backend_id"] = "byok-c12probe"
    neg["no_registerable_key"] = {
        "provider": SYNTHETIC_TOKEN_PROVIDER,
        **_findings(
            house.find_house_keyed_catalogue_entries,
            [*rows, token_row],
            [*rungs, dict(SYNTHETIC_TOKEN_RUNG)],
        ),
    }

    neg["empty_rungs_fail_closed"] = _findings(
        house.find_house_keyed_catalogue_entries, rows, []
    )

    obs["negative"] = neg
    obs["finding_classes"] = {
        "declared_house_ref": house.DECLARED_HOUSE_REF,
        "house_rung_identity_collision": house.HOUSE_RUNG_IDENTITY_COLLISION,
        "no_customer_registerable_key": house.NO_CUSTOMER_REGISTERABLE_KEY,
    }
    return obs


def main() -> int:
    app_root = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        payload = observe(app_root)
    except Exception as exc:  # noqa: BLE001 - reported, graded as could-not-run
        payload = {"error": _exc(exc)}
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
