# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19195 -- the C12 OBSERVER. Runs INSIDE the deployed runtime container.

WHAT THIS IS
    The half of the C12 producer that executes in the deployed interpreter,
    against the deployed ``omnimarket`` package and the runtime's contract-owned
    intake mount (``node_inference_credentials``; the route left onex-api).
    ``c12_provider_catalogue_probe.py`` pipes this file to
    ``docker exec -i -u omniinfra <container> python -`` and grades what it
    prints. This file GRADES NOTHING: a program that both runs the subject and
    decides whether it passed is a grader grading itself.

WHY INSIDE THE CONTAINER AND NOT OVER HTTP
    The intake route answers 401 before its body is validated, so nothing is
    readable over HTTP; the deployed package, the mount's declared create route
    and the model its create handler validates with are the strongest reading.

IT WRITES NOTHING
    Every injection is an in-memory copy of a shipped row handed to a PURE
    deployed function: no file written, no route called, no event published.

OUTPUT: one JSON object on stdout, last line; a read failure is ``{"error": ...}``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

# The customer route this criterion's catalogue is bound to (the grader pins it).
ROUTE_PREFIX = "/v1/tenants/me/inference-credentials"

# A provider id no catalogue offers and no platform rung backs: the unbacked
# injection and a refused-at-intake case (``openai`` until OMN-17373 offered it).
UNBACKED_PROVIDER = "mistral"

# The un-registerable-key injection's provider, synthetic ON PURPOSE: a real one
# exists today (vertex), but the injection must stay constructible the day it is
# removed, or the negative test silently stops running.
SYNTHETIC_TOKEN_PROVIDER = "c12probe"
SYNTHETIC_TOKEN_RUNG = {
    "backend_id": "c12-probe-token-rung",
    "secret_ref": f"llm.{SYNTHETIC_TOKEN_PROVIDER}.access_token",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exc(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:600]}


def _customer_only(rows: list[dict[str, Any]]) -> list[str]:
    """Providers every one of whose rows declares ``mirrors_house_rung: false``.

    Reports which providers the catalogue exempts, from the RAW rows; whether the
    exemption is legitimate is the grader's call against its own declared list.
    """
    mirroring: dict[str, bool] = {}
    for row in rows:
        name = str(row.get("provider", "")).strip()
        mirroring[name] = mirroring.get(name, False) or (
            row.get("mirrors_house_rung", True) is not False
        )
    return sorted(name for name, any_mirror in mirroring.items() if not any_mirror)


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


def _create_route_and_models(pkg_dir: Path) -> tuple[str | None, list[Any]]:
    """The mount contract's POST create path and the classes its handler validates with."""
    import yaml

    node_dir = pkg_dir / "nodes" / "node_inference_credentials"
    contract = yaml.safe_load((node_dir / "contract.yaml").read_text(encoding="utf-8"))
    routes = {(r["method"], r["operation"]): r["path"] for r in contract["http_routes"]}
    path = routes.get(("POST", "create"))
    handler = node_dir / "handlers" / "handler_inference_credentials.py"
    tree = ast.parse(handler.read_text(encoding="utf-8"))
    imported = {
        (a.asname or a.name): (n.module, a.name)
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
        for a in n.names
    }
    models: list[Any] = []
    for n in ast.walk(tree):
        f = n.func if isinstance(n, ast.Call) else None
        if (
            isinstance(f, ast.Attribute)
            and f.attr in {"model_validate", "model_validate_json"}
            and isinstance(f.value, ast.Name)
            and f.value.id in imported
        ):
            module, name = imported[f.value.id]
            models.append(getattr(importlib.import_module(module), name))
    return path, models


def observe(app_root: str) -> dict[str, Any]:
    if app_root and app_root not in sys.path:
        sys.path.insert(0, app_root)

    from omnimarket.handlers.routing import byok_provider_backends as byok
    from omnimarket.projection.credential_publisher import (
        ModelInferenceCredentialCreateRequest,
    )
    from omnimarket.validators import byok_catalogue_no_house_entry as house

    obs: dict[str, Any] = {
        "omnimarket_version": importlib.metadata.version("omnimarket"),
        "files": {
            "catalogue": _sha256(Path(byok.CATALOG_PATH)),
            "platform_contract": _sha256(Path(house.BIFROST_CONTRACT_PATH)),
            "catalogue_module": _sha256(Path(byok.__file__)),
            "house_validator": _sha256(Path(house.__file__)),
        },
    }

    # ---- the shipped reading ------------------------------------------------
    offered = list(byok.customer_provider_catalogue())
    not_offered = sorted(byok.load_byok_not_offered_providers())
    rows = house.read_catalogue_rows()
    rungs = house.read_platform_rungs()
    house_slugs = sorted(byok.house_keyed_provider_slugs(rungs))
    customer_only = _customer_only(rows)
    house_offered = [p for p in offered if p not in customer_only]
    gap = byok.catalogue_parity_gap(
        house_slugs, offered=house_offered, not_offered=not_offered
    )
    obs["shipped"] = {
        "offered": offered,
        "customer_only": customer_only,
        "not_offered": not_offered,
        "rung_count": len(rungs),
        "house_keyed_slugs": house_slugs,
        "parity_gap": {
            "missing_from_catalogue": list(gap.missing_from_catalogue),
            "unbacked_in_catalogue": list(gap.unbacked_in_catalogue),
        },
        "house_entry": _findings(house.find_house_keyed_catalogue_entries, rows, rungs),
    }

    # ---- the customer surface ----------------------------------------------
    route_path, candidates = _create_route_and_models(
        Path(byok.__file__).resolve().parents[2]
    )
    obs["route"] = {
        "prefix": route_path,
        "body_models": [f"{m.__module__}.{m.__qualname__}" for m in candidates],
        "bound_to_catalogue_model": any(
            m is ModelInferenceCredentialCreateRequest for m in candidates
        ),
    }
    obs["intake"] = {
        p: _intake(ModelInferenceCredentialCreateRequest, p)
        for p in [*offered, *not_offered, UNBACKED_PROVIDER]
    }

    # ---- the negative test: each clause shown to bite, same process ---------
    neg: dict[str, Any] = {}

    g = byok.catalogue_parity_gap(
        house_slugs,
        offered=[*house_offered, UNBACKED_PROVIDER],
        not_offered=not_offered,
    )
    neg["parity_unbacked"] = {
        "injected": UNBACKED_PROVIDER,
        "unbacked_in_catalogue": list(g.unbacked_in_catalogue),
        "missing_from_catalogue": list(g.missing_from_catalogue),
    }

    declared = sorted({*house_offered, *not_offered})
    dropped = declared[0] if declared else None
    g = byok.catalogue_parity_gap(
        house_slugs,
        offered=[p for p in house_offered if p != dropped],
        not_offered=[p for p in not_offered if p != dropped],
    )
    neg["parity_missing"] = {
        "dropped": dropped,
        "unbacked_in_catalogue": list(g.unbacked_in_catalogue),
        "missing_from_catalogue": list(g.missing_from_catalogue),
    }

    exempt_row = copy.deepcopy(rows[0]) if rows else {}
    exempt_row["provider"] = UNBACKED_PROVIDER
    exempt_row["mirrors_house_rung"] = False
    neg["customer_only_self_exemption"] = {
        "injected": UNBACKED_PROVIDER,
        "customer_only": _customer_only([*rows, exempt_row]),
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
