#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-20217/20218/20219 walk oracle. Committed verbatim from appendix A.0.2 of
# knowledge-base-internal beta/plans/2026-09-30-delegation-sprint-test-matrix.md,
# the delegation sprint test matrix, where it previously existed only as an
# inline code block. AC3 of each walk ticket requires this checker's output
# attached to every draw; a checker each walker transcribes separately is not
# the same oracle, so the three walks could not be compared. The only edit is
# splitting one multi-import line; no behaviour is changed.
"""Usage: python check_receipt.py RECEIPT.json --route L|C --pin BACKEND_ID
            [--lane LANE] [--secret-ref REF] [--tenant TENANT]

Reads one run's receipt.json, unwraps its delegation terminal the way `onex delegate` does, and
checks the route evidence. Prints every field it read and ends with VALID or INVALID_ROUTE.

Where the terminal is (cli_delegate._delegation_result, delegate_terminal_resolver):
  * result_model names a ModelDelegateSkill* class   -> receipt.result IS the terminal
  * otherwise receipt.result is a runtime summary and the terminal is in `terminal_payload`
    (then `handler_result`): bare on an in-process run, and on a deployed-lane run an event
    envelope (`envelope_id`) with the terminal one level down under `payload`.
Top-level keys of receipt.json (backend_id, model, routing_tier, bus, locus, lane,
requested_backend_id, backend_pin_honoured) are written by the CLI from the same terminal.
"""
import argparse
import json
import re
import sys


def terminal(doc: dict) -> dict:
    env = doc["receipt"]
    res = env["result"]
    if "ModelDelegateSkill" in str(env.get("result_model") or ""):
        return res
    for field in ("terminal_payload", "handler_result"):
        carrier = res.get(field)
        if not isinstance(carrier, dict):
            continue
        if "envelope_id" in carrier and isinstance(carrier.get("payload"), dict) and "attempts" in carrier["payload"]:
            return carrier["payload"]
        if "attempts" in carrier:
            return carrier
    raise LookupError("no resolvable delegation terminal in this receipt (a REFUSED or dropped-reply run)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt")
    ap.add_argument("--route", choices=["L", "C"], required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--lane")
    ap.add_argument("--secret-ref", help="the reference `onex secret set` printed when the key was registered")
    ap.add_argument("--tenant", help="the walker's tenant id, if recorded at setup")
    ap.add_argument("--model", help="override the expected model name")
    a = ap.parse_args(argv)

    doc = json.load(open(a.receipt))
    bad, seen = [], []

    def expect(name, got, want):
        seen.append(f"{name} = {got!r}")
        if got != want:
            bad.append(f"{name}: read {got!r}, expected {want!r}")

    try:
        t = terminal(doc)
    except LookupError as exc:
        print(f"INVALID_ROUTE\n  {exc}")
        return 1
    accepted = [x for x in t.get("attempts", []) if x.get("acceptance_decision") == "accept"]
    acc = accepted[0] if len(accepted) == 1 else {}
    if len(accepted) != 1:
        bad.append(f"expected exactly one accepted attempt, found {len(accepted)}")
    m = t.get("metrics") or {}

    expect("requested_backend_id", doc.get("requested_backend_id"), a.pin)
    expect("backend_pin_honoured", doc.get("backend_pin_honoured"), True)
    if a.route == "L":
        expect("bus", doc.get("bus"), "kafka")
        expect("locus", doc.get("locus"), "deployed-lane")
        if a.lane:
            expect("lane", doc.get("lane"), a.lane)
        expect("backend_id", doc.get("backend_id"), a.pin)
        expect("model", t.get("model_name"), a.model or "Qwen3.8-27B")
        expect("routing_tier", doc.get("routing_tier"), "local")
        expect("accepted attempt backend_id", acc.get("backend_id"), a.pin)
        expect("secret_source", t.get("secret_source"), None)
        expect("secret_ref", t.get("secret_ref"), None)
        expect("metrics.cost_usd (calculated)", m.get("cost_usd"), 0.0)
    else:
        expect("bus", doc.get("bus"), "inmemory")
        expect("locus", doc.get("locus"), "in-process")
        expect("lane", doc.get("lane"), None)
        expect("backend_id", doc.get("backend_id"), "byok-gemini")
        expect("model", t.get("model_name"), a.model or "gemini-2.5-flash-lite")
        expect("routing_tier", doc.get("routing_tier"), "cheap_cloud")
        expect("accepted attempt substituted_from_backend_id", acc.get("substituted_from_backend_id"), a.pin)
        expect("secret_source", t.get("secret_source"), "store")
        ref = t.get("secret_ref") or ""
        seen.append(f"secret_ref = {ref!r}")
        if not a.secret_ref:
            bad.append("--secret-ref is required for route C: the reference printed at registration")
        elif ref != a.secret_ref:
            bad.append(f"secret_ref: read {ref!r}, the registered reference is {a.secret_ref!r}")
        if ref.startswith("llm."):
            bad.append("secret_ref is a house reference (llm.*): a house credential answered customer work")
        if not re.fullmatch(r"cred_localinstall_gemini_[0-9a-f]{32}", ref):
            bad.append("secret_ref is not shaped cred_localinstall_gemini_<32 hex>")
    tenant = t.get("tenant_id")
    seen.append(f"tenant_id = {tenant!r}")
    if a.route == "C":
        if not tenant or tenant == "omninode":
            bad.append(f"tenant_id: read {tenant!r}, expected the walker's own tenant, not the house default")
        if a.tenant and tenant != a.tenant:
            bad.append(f"tenant_id: read {tenant!r}, the walker's tenant is {a.tenant!r}")
    else:
        if tenant == "omninode":
            bad.append("tenant_id is the house default 'omninode'")
        if a.tenant and tenant not in (None, a.tenant):
            bad.append(f"tenant_id: read {tenant!r}, the walker's tenant is {a.tenant!r}")
        if tenant is None:
            seen.append("(tenant_id is null on this deployed-lane terminal: read it from the projection row)")
    seen.append(f"tokens measured: input {m.get('input_tokens')}, output {m.get('output_tokens')}; "
                f"cost_usd {m.get('cost_usd')} is CALCULATED from tier pricing, not a provider charge")
    print("\n".join("  " + s for s in seen))
    print("VALID" if not bad else "INVALID_ROUTE\n  " + "\n  ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
