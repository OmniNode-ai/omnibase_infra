# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19180 -- the C14 producer: routing proven on a no-checkout customer install.

WHAT THIS IS
    The producer for beta criterion C14, "Routing proven on the customer-local
    path; no customer-visible task class has an unbindable tier", whose proof is
    "rows 1, 2 and 4 probes on a no-checkout customer machine". Its EXIT CODE is
    the criterion's verdict -- the shape C11 and C15 already set -- and it writes
    a JSON record so a reader can see each row without reading a log.

    ``0``  all three rows held on this machine, and every control held
    ``1``  at least one row did not; the record names the row and the finding
    ``2``  the probe could not run: the machine is not a no-checkout install, the
           installed product could not be introspected, or a control proved an
           instrument blind. Distinct from ``1`` for the reason C11 gives: "I
           could not run" is not "the product routed wrongly".

THE THREE ROWS (knowledge-base-internal ``beta/GOAL.md``; beta PRD R-DELEG-5/7)
    row 1  A good local answer ends the chain, and the decision is in the log.
           One real ``onex delegate`` run on the installed CLI. Graded from the
           attempt ladder in that run's own ``receipt.json``: the first rung
           tried is local, the accepted rung is local and is the LAST rung, the
           accept is a typed decision, nothing escalated, nothing cost money.
           The ladder is read from the receipt, not the local SQLite evidence
           row, which drops it (OMN-18889) -- the receipt carries it intact.
    row 2  No routing tier names an unbindable backend. Every customer-visible
           task class -- the ``public`` Gateway projection of the SHIPPED
           task-class authority -- is walked tier by tier through the installed
           routing reducer's own helpers, and every backend each tier could hand
           that class is classified on THIS machine: declared at all, carrying a
           complete endpoint after the machine's overlay, and, for a local
           endpoint, actually serving the model it names.
    row 4  The reviewer leg is unmetered. The quality-gate judge -- the second
           opinion that arbitrates a disputed answer -- is resolved through the
           judge adapter's own resolver, and must land on a private-network
           endpoint with no credential, serving its model, and the run's
           recorded egress must carry zero calls to a non-private host. No
           disputed answer is forced: the arbitration ROUTE is graded, and the
           run's egress bounds what any leg of it actually called.

"NO CHECKOUT" IS ASSERTED, NOT ASSUMED
    Every installed package the product is made of must import from a
    ``site-packages`` directory with no ``.git`` above it, the interpreter runs
    isolated (``-I``, so neither ``PYTHONPATH`` nor the script's own directory
    reaches ``sys.path``), and every delegation path binding must point inside
    the installed package or at the machine's own overlay -- never into a
    repository. A machine that fails any of those is exit 2, because a verdict
    taken there would describe a developer's workstation, not a customer's.

    The routing-tiers binding must also be BYTE-IDENTICAL to the file the
    installed package ships, so the walk grades the shipped contract rather
    than a local edit of it.

A ZERO HAS TO BE PROVEN
    Row 1's "nothing later" and row 4's "unmetered" are both absence claims.
    The delegation runs behind a recording forward proxy set through the
    environment, which every process of the run inherits whether the product
    forks or spawns its inference child, and which both of its transports
    (httpx and curl) honour. The recording must contain the accepted rung's own
    inference POST before an empty remainder is read as zero -- a recording
    with no POST at all is an instrument that was not in the path, and grades
    RED, not green.

THE CONTROLS ARE GRADED
    Two negative controls run in the SAME invocation, through the SAME
    classifier the verdict uses: a backend id that is declared nowhere must
    classify ``deleted_backend``, and an endpoint on a closed port must classify
    ``unreachable``. If either comes back bindable the classifier cannot fail,
    and the run is exit 2 -- a probe that cannot fail is not evidence.

WHY THE COLLECTOR RUNS UNDER THE CUSTOMER'S INTERPRETER
    The host process is stdlib-only and never imports the product. The routing
    walk runs as ``<customer python> -I <this file> --collect`` and the
    delegation is the installed ``onex`` console script itself, so both see
    exactly the install a customer has. ``--replay`` grades a recorded
    ``{collection, run}`` offline, which is what makes the grader falsifiable in
    the ordinary test suite.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2

#: The distributions a customer install is made of. Each must import from a
#: site-packages directory with no ``.git`` above it.
PRODUCT_MODULES: Final[tuple[str, ...]] = (
    "omnimarket",
    "omnibase_infra",
    "omnibase_core",
)

#: The prompt the shared customer machine (C13, OMN-19179) measured answering
#: on its pinned local model, on an explicitly named class. The carrier's own
#: one-word recipe (OMN-16932, "Reply with exactly the word: alive" on
#: research) was measured on that model on 2026-09-22 and refused three times
#: as an empty extraction: a 3B model does not emit the answer marker the
#: extraction needs. That is the model's limit, not routing's, and a prompt the
#: machine's model cannot answer would turn row 1 red for a reason that is not
#: C14's.
DEFAULT_PROMPT: Final[str] = "explain what a calendar app needs"
DEFAULT_TASK_TYPE: Final[str] = "document"

LOCAL_TIER: Final[str] = "local"

#: Declared nowhere, on purpose. The negative control that proves the
#: classifier can say ``deleted_backend``.
CONTROL_DELETED_BACKEND_ID: Final[str] = "c14-control-backend-declared-nowhere"
#: Port 9 (discard) on loopback: closed on every runner this has met. The
#: negative control that proves the classifier can say ``unreachable``.
CONTROL_UNREACHABLE_ENDPOINT: Final[str] = (
    "http://127.0.0.1:9/v1/chat/completions"  # url-authority-ok: negative control, a closed port that must classify unreachable
)

#: Hostnames that resolve to the machine's own gateway rather than the public
#: internet. Resolution still happens; this only names what "private" allows.
_PRIVATE_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {"localhost", "host.docker.internal"}
)

# Finding classes, row 2. The remedy differs per class, so they stay distinct.
DELETED_BACKEND: Final[str] = "deleted_backend"
NO_ENDPOINT: Final[str] = "no_endpoint_on_this_machine"
UNREACHABLE: Final[str] = "unreachable"
MODEL_NOT_SERVED: Final[str] = "model_not_served"
TIER_UNDECLARED: Final[str] = "tier_undeclared"
TIER_SERVES_NOTHING: Final[str] = "tier_serves_class_with_no_backend"
NO_LOCAL_FIRST_RUNG: Final[str] = "no_local_first_rung"


#: The ONLY environment a customer command sees. Everything else the host
#: process carries -- PYTHONPATH, OMNI_HOME, a developer's own bindings -- is
#: dropped rather than filtered, so a new leak cannot arrive by being unlisted.
CUSTOMER_ENV_ALLOWLIST: Final[tuple[str, ...]] = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "DELEGATION_ROUTING_TIERS_PATH",
    "BIFROST_CONTRACT_PATH",
    "BIFROST_OVERLAY_PATH",
)


class ProbeInputError(RuntimeError):
    """The probe could not run. Exit 2, never exit 1."""


def customer_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The allowlisted environment every customer command runs under."""
    return {key: environ[key] for key in CUSTOMER_ENV_ALLOWLIST if environ.get(key)}


# ---------------------------------------------------------------------------
# Host-side helpers (stdlib only)
# ---------------------------------------------------------------------------


def endpoint_is_private(endpoint_url: str) -> bool:
    """Whether an endpoint's host resolves ONLY to private or loopback addresses.

    A host that resolves to nothing is not private: it is unresolvable, and the
    caller treats it as a non-private endpoint rather than giving it the benefit
    of the doubt.
    """
    host = urlsplit(endpoint_url).hostname or ""
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        addresses = {info[4][0] for info in infos}
        if not addresses:
            return False
        return all(_address_is_private(a) for a in addresses) or (
            host in _PRIVATE_HOSTNAMES and bool(addresses)
        )
    return _address_is_private(str(address))


def _private_address(endpoint_url: str) -> str | None:
    """The private IP to connect to for ``endpoint_url``, or None if it is not private.

    The recorder connects to the ADDRESS this returns, never to the hostname a
    request named, so a destination outside the private network is unreachable
    through it by construction rather than by a check a later edit could skip.
    """
    if not endpoint_is_private(endpoint_url):
        return None
    host = urlsplit(endpoint_url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return None
    for info in infos:
        candidate = str(info[4][0])
        if _address_is_private(candidate):
            return ipaddress.ip_address(candidate.split("%", 1)[0]).compressed
    return None


def _address_is_private(raw: str) -> bool:
    try:
        address = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return False
    return address.is_private or address.is_loopback


# ---------------------------------------------------------------------------
# Grading (pure; the offline half of the proof)
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    name: str
    title: str
    ok: bool
    findings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "ok": self.ok,
            "findings": list(self.findings),
            "evidence": self.evidence,
        }


@dataclass
class Record:
    machine: dict[str, Any]
    rows: list[RowResult]
    controls: list[dict[str, Any]]
    input_errors: list[str]

    @property
    def verdict(self) -> str:
        if self.input_errors:
            return "COULD_NOT_RUN"
        return "PASS" if all(row.ok for row in self.rows) else "FAIL"

    @property
    def exit_code(self) -> int:
        return {"PASS": EXIT_OK, "FAIL": EXIT_FINDINGS}.get(self.verdict, EXIT_INPUT)

    @property
    def detail(self) -> str:
        if self.input_errors:
            return "; ".join(self.input_errors)
        red = [row.name for row in self.rows if not row.ok]
        if not red:
            return "rows 1, 2 and 4 held on a no-checkout install"
        return "RED rows: " + ", ".join(red)

    def to_dict(self, *, as_of: str) -> dict[str, Any]:
        return {
            "record_version": RECORD_VERSION,
            "criterion": "C14",
            "ticket": "OMN-19180",
            "as_of": as_of,
            "verdict": self.verdict,
            "detail": self.detail,
            "machine": self.machine,
            "controls": self.controls,
            "rows": [row.to_dict() for row in self.rows],
            "input_errors": list(self.input_errors),
        }


def grade_machine(collection: Mapping[str, Any]) -> list[str]:
    """Return the reasons this machine is not a no-checkout customer install."""
    errors: list[str] = []
    if collection.get("collect_error"):
        errors.append(f"collector failed: {collection['collect_error']}")
        return errors
    modules = collection.get("modules")
    if not isinstance(modules, dict) or not modules:
        return ["collector reported no installed product modules"]
    for name in PRODUCT_MODULES:
        entry = modules.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{name} is not importable on this machine")
            continue
        if not entry.get("in_site_packages"):
            errors.append(
                f"{name} imports from {entry.get('path')!r}, not site-packages"
            )
        if entry.get("git_ancestor"):
            errors.append(
                f"{name} imports from inside a git working tree "
                f"({entry.get('git_ancestor')!r}) -- this is a checkout"
            )
    if not collection.get("isolated_interpreter"):
        errors.append("the collector did not run under an isolated (-I) interpreter")
    for key, binding in (collection.get("bindings") or {}).items():
        if isinstance(binding, dict) and binding.get("git_ancestor"):
            errors.append(
                f"{key} is bound into a git working tree ({binding.get('path')!r})"
            )
    tiers = (collection.get("bindings") or {}).get("DELEGATION_ROUTING_TIERS_PATH")
    if (
        isinstance(tiers, dict)
        and tiers.get("path")
        and not tiers.get("matches_shipped")
    ):
        errors.append(
            "DELEGATION_ROUTING_TIERS_PATH is bound to a file that is not "
            "byte-identical to the installed package's shipped routing_tiers.yaml, "
            "so the walk would grade a local edit rather than the shipped contract"
        )
    return errors


def grade_controls(
    collection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Both negative controls must come back UNbindable, or the classifier is blind."""
    raw = collection.get("controls")
    controls = list(raw) if isinstance(raw, list) else []
    errors: list[str] = []
    expected = {
        "deleted_backend_control": DELETED_BACKEND,
        "unreachable_endpoint_control": UNREACHABLE,
    }
    by_name = {c.get("name"): c for c in controls if isinstance(c, dict)}
    for name, want in expected.items():
        got = by_name.get(name, {}).get("classification")
        if got != want:
            errors.append(
                f"{name} classified {got!r}, expected {want!r}: the bindability "
                "classifier cannot produce a RED, so its greens mean nothing"
            )
    return controls, errors


def grade_row2(collection: Mapping[str, Any]) -> RowResult:
    """No customer-visible task class has a tier naming an unbindable backend."""
    row = RowResult(
        name="row2_no_unbindable_tier",
        title="no routing tier names an unbindable backend",
        ok=True,
    )
    authority = collection.get("public_task_classes")
    contract_public = collection.get("public_task_classes_from_contract_yaml")
    walk = collection.get("walk")
    if not isinstance(authority, list) or not authority:
        row.ok = False
        row.findings.append(
            "the task-class authority yielded ZERO customer-visible classes -- an "
            "empty enumeration is a failure, never a vacuous pass"
        )
        return row
    if not isinstance(contract_public, list) or sorted(contract_public) != sorted(
        authority
    ):
        row.ok = False
        row.findings.append(
            "the authority's public projection "
            f"{sorted(authority)} disagrees with the shipped contract's own "
            f"gateway_exposure: public entries {contract_public!r} -- the "
            "enumeration shrank or grew somewhere between file and loader"
        )
    if not isinstance(walk, dict):
        row.ok = False
        row.findings.append("the collector returned no routing walk")
        return row
    missing = sorted(set(authority) - set(walk))
    if missing:
        row.ok = False
        row.findings.append(f"classes absent from the walk (not a skip): {missing}")

    backends = collection.get("backends") or {}
    for task_class in sorted(walk):
        tiers = walk[task_class]
        if not isinstance(tiers, list) or not tiers:
            row.ok = False
            row.findings.append(f"{task_class}: resolved an empty tier order")
            continue
        if tiers[0].get("tier") != LOCAL_TIER:
            # Row 1's "a good local answer ends the chain" cannot hold for a
            # class whose ladder does not START on a free local rung: the
            # first answer it gets is a metered one.
            row.ok = False
            row.findings.append(
                f"{task_class}: [{NO_LOCAL_FIRST_RUNG}] its tier order starts "
                f"at {tiers[0].get('tier')!r}, so no local answer can end the chain"
            )
        for tier in tiers:
            tier_name = tier.get("tier")
            if not tier.get("declared", True):
                row.ok = False
                row.findings.append(
                    f"{task_class}/{tier_name}: [{TIER_UNDECLARED}] the class's "
                    "tier_order names a tier routing_tiers.yaml does not declare"
                )
                continue
            candidates = tier.get("candidates") or []
            if not candidates:
                row.ok = False
                row.findings.append(
                    f"{task_class}/{tier_name}: [{TIER_SERVES_NOTHING}] the tier "
                    "is in the class's closed tier_order but no model in it "
                    "serves the class"
                )
                continue
            for backend_id in candidates:
                classification = (backends.get(backend_id) or {}).get(
                    "classification", DELETED_BACKEND
                )
                if classification != "bindable":
                    row.ok = False
                    row.findings.append(
                        f"{task_class}/{tier_name}: backend {backend_id!r} is "
                        f"[{classification}]"
                    )
    row.evidence = {
        "public_task_classes": sorted(authority),
        "classes_walked": len(walk),
        "backends": backends,
    }
    return row


def _egress(run: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Every outbound request the recording proxy saw during the delegation."""
    raw = run.get("egress") if isinstance(run, Mapping) else None
    return (
        [entry for entry in raw if isinstance(entry, dict)]
        if isinstance(raw, list)
        else []
    )


def grade_row1(run: Mapping[str, Any] | None) -> RowResult:
    """A good local answer ends the chain, and the decision is in the log."""
    row = RowResult(
        name="row1_local_answer_ends_chain",
        title="a good local answer ends the chain; the decision is in the log",
        ok=True,
    )
    if not isinstance(run, Mapping) or not run:
        row.ok = False
        row.findings.append("no delegation run was recorded")
        return row
    if run.get("error"):
        row.ok = False
        row.findings.append(f"the delegation could not be run: {run['error']}")
        return row
    receipt = run.get("receipt_result")
    if not isinstance(receipt, Mapping):
        row.ok = False
        row.findings.append(
            f"no receipt result was read (cli exit {run.get('exit_code')}); "
            f"terminal: {str(run.get('terminal_error') or '')[:400]}"
        )
        return row

    status = receipt.get("status")
    attempts = (
        receipt.get("attempts") if isinstance(receipt.get("attempts"), list) else []
    )
    row.evidence = {
        "correlation_id": receipt.get("correlation_id"),
        "cli_exit_code": run.get("exit_code"),
        "status": status,
        "task_type": receipt.get("task_type"),
        "model_name": receipt.get("model_name"),
        "provider": receipt.get("provider"),
        "attempts_count": receipt.get("attempts_count"),
        "escalation_count": receipt.get("escalation_count"),
        "attempts": attempts,
    }
    if run.get("exit_code") != 0 or status != "completed":
        row.ok = False
        row.findings.append(
            f"the run did not complete (exit {run.get('exit_code')}, status "
            f"{status!r}): {str(receipt.get('error_message') or '')[:400]}"
        )
    if not attempts:
        row.ok = False
        row.findings.append("the receipt carries an EMPTY attempt ladder")
        return row
    if receipt.get("attempts_count") != len(attempts):
        row.ok = False
        row.findings.append(
            f"attempts_count {receipt.get('attempts_count')} disagrees with the "
            f"{len(attempts)} attempt record(s) on the receipt"
        )
    if attempts[0].get("tier") != LOCAL_TIER:
        row.ok = False
        row.findings.append(
            f"the first rung tried was {attempts[0].get('tier')!r}, not local"
        )
    non_local = [a.get("tier") for a in attempts if a.get("tier") != LOCAL_TIER]
    if non_local:
        row.ok = False
        row.findings.append(f"the ladder left the local tier: {non_local}")
    accepted = [
        index
        for index, attempt in enumerate(attempts)
        if attempt.get("acceptance_decision") == "accept"
    ]
    if not accepted:
        row.ok = False
        row.findings.append("no rung carries a typed accept decision")
    else:
        first = accepted[0]
        winner = attempts[first]
        if first != len(attempts) - 1:
            row.ok = False
            row.findings.append(
                f"{len(attempts) - 1 - first} rung(s) ran AFTER an accepted "
                "local answer -- the chain did not end"
            )
        if winner.get("tier") != LOCAL_TIER or not winner.get("quality_gate_passed"):
            row.ok = False
            row.findings.append(
                "the accepted rung is not a local rung that passed its gate: "
                f"tier={winner.get('tier')!r} passed={winner.get('quality_gate_passed')!r}"
            )
        if not winner.get("acceptance_reason"):
            row.ok = False
            row.findings.append("the accept decision carries no typed reason")
    if receipt.get("escalation_count") != 0:
        row.ok = False
        row.findings.append(
            f"escalation_count is {receipt.get('escalation_count')!r}, not 0"
        )
    cost = (receipt.get("metrics") or {}).get("cost_usd")
    if cost not in (0, 0.0):
        row.ok = False
        row.findings.append(f"the run cost {cost!r} USD; a local answer costs nothing")

    egress = _egress(run)
    posts = [entry for entry in egress if entry.get("method") == "POST"]
    row.evidence["egress"] = egress
    if not posts:
        row.ok = False
        row.findings.append(
            "the egress recorder saw NO inference POST at all, so its silence "
            "about later calls is an unproven zero, not a zero"
        )
    else:
        non_private = [
            entry.get("target") for entry in egress if not entry.get("private")
        ]
        if non_private:
            row.ok = False
            row.findings.append(
                f"the run reached beyond the private network: {non_private}"
            )
        if len(posts) > len(attempts):
            row.ok = False
            row.findings.append(
                f"{len(posts)} inference POSTs for {len(attempts)} recorded "
                "rung(s) -- a call happened that the ladder does not account for"
            )
    return row


def grade_row4(
    collection: Mapping[str, Any], run: Mapping[str, Any] | None
) -> RowResult:
    """The reviewer (judge) leg is unmetered on this machine."""
    row = RowResult(
        name="row4_reviewer_leg_unmetered",
        title="the reviewer leg resolves to an unmetered local model",
        ok=True,
    )
    judge = collection.get("judge")
    if not isinstance(judge, Mapping) or judge.get("error"):
        row.ok = False
        row.findings.append(
            "the reviewer leg could not be resolved: "
            f"{(judge or {}).get('error') if isinstance(judge, Mapping) else judge!r}"
        )
        return row
    row.evidence = dict(judge)
    endpoint = str(judge.get("endpoint") or "")
    if not judge.get("endpoint_private"):
        row.ok = False
        row.findings.append(
            f"the reviewer leg ({judge.get('backend_id')!r}) resolves to "
            f"{endpoint!r} (provider {judge.get('provider')!r}), which is not a "
            "private-network endpoint -- a disputed answer is arbitrated by a "
            "metered provider"
        )
    if judge.get("requires_credential"):
        row.ok = False
        row.findings.append(
            "the reviewer leg requires a provider credential; "
            "an unmetered local reviewer needs none"
        )
    if judge.get("endpoint_private") and judge.get("classification") != "bindable":
        row.ok = False
        row.findings.append(
            f"the reviewer leg's local endpoint is [{judge.get('classification')}]"
        )

    egress = _egress(run)
    metered = [entry.get("target") for entry in egress if not entry.get("private")]
    row.evidence["egress"] = egress
    if not egress:
        row.ok = False
        row.findings.append(
            "the egress recorder saw nothing during the run, so 'zero metered "
            "calls' is unproven"
        )
    elif metered:
        row.ok = False
        row.findings.append(f"the run made metered (non-private) calls: {metered}")
    return row


def grade(collection: Mapping[str, Any], run: Mapping[str, Any] | None) -> Record:
    machine_errors = grade_machine(collection)
    controls, control_errors = grade_controls(collection)
    input_errors = machine_errors + control_errors
    rows = [
        grade_row1(run),
        grade_row2(collection),
        grade_row4(collection, run),
    ]
    machine = {
        "modules": collection.get("modules"),
        "bindings": collection.get("bindings"),
        "isolated_interpreter": collection.get("isolated_interpreter"),
        "python": collection.get("python"),
    }
    return Record(
        machine=machine, rows=rows, controls=controls, input_errors=input_errors
    )


# ---------------------------------------------------------------------------
# Collector -- runs UNDER THE CUSTOMER'S INTERPRETER (-I), imports the product
# ---------------------------------------------------------------------------


def _git_ancestor(path: Path) -> str | None:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return str(parent)
    return None


def _module_facts(name: str) -> dict[str, Any]:
    import importlib

    module = importlib.import_module(name)
    path = Path(str(module.__file__)).resolve()
    version = ""
    try:
        from importlib.metadata import version as dist_version

        version = dist_version(name.replace("_", "-"))
    except Exception:  # noqa: BLE001 - a missing version is recorded, not fatal
        version = ""
    return {
        "path": str(path),
        "version": version,
        "in_site_packages": any(
            part in ("site-packages", "dist-packages") for part in path.parts
        ),
        "git_ancestor": _git_ancestor(path.parent),
    }


def _classify_backend(
    backend_id: str,
    *,
    declarations: Mapping[str, Any],
    loaded: Mapping[str, Any],
    served_cache: dict[str, frozenset[str] | None],
    probe_served_models: Any,
) -> dict[str, Any]:
    declaration = declarations.get(backend_id)
    if declaration is None:
        return {"classification": DELETED_BACKEND}
    tier = getattr(declaration, "tier", None)
    facts: dict[str, Any] = {"declared_tier": str(tier) if tier is not None else None}
    ref = loaded.get(backend_id)
    if ref is None:
        facts["classification"] = NO_ENDPOINT
        return facts
    endpoint = str(ref.endpoint_url)
    facts.update(
        {
            "endpoint": endpoint,
            "model_name": ref.model_name,
            "provider": ref.provider,
            # Whether a credential is needed, never its reference: the record is
            # an uploaded artifact and carries no credential material at all.
            "requires_credential": bool(ref.api_key_ref),
            "endpoint_private": endpoint_is_private(endpoint),
        }
    )
    if not facts["endpoint_private"]:
        # A cloud backend with a complete endpoint is bindable: the customer
        # brings the key. Whether a key is present here is recorded, not graded.
        facts["classification"] = "bindable"
        return facts
    return _classify_private_endpoint(
        facts, endpoint, ref.model_name, served_cache, probe_served_models
    )


def _classify_private_endpoint(
    facts: dict[str, Any],
    endpoint: str,
    model_name: str,
    served_cache: dict[str, frozenset[str] | None],
    probe_served_models: Any,
) -> dict[str, Any]:
    if endpoint not in served_cache:
        served_cache[endpoint] = probe_served_models(endpoint, timeout_seconds=10.0)
    served = served_cache[endpoint]
    facts["served_models"] = sorted(served) if served else None
    if not served:
        facts["classification"] = UNREACHABLE
    elif model_name not in served and model_name.lower() not in {
        s.lower() for s in served
    }:
        facts["classification"] = MODEL_NOT_SERVED
    else:
        facts["classification"] = "bindable"
    return facts


def collect() -> dict[str, Any]:  # pragma: no cover - exercised on the machine
    """Introspect the installed product. Runs only under the customer's python."""
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "isolated_interpreter": bool(sys.flags.isolated),
    }
    out["modules"] = {name: _module_facts(name) for name in PRODUCT_MODULES}

    import yaml
    from omnimarket.adapters.llm.bifrost.config_loader_bifrost_delegation import (
        load_bifrost_delegation_config,
    )
    from omnimarket.inference.delegation_config_provenance import (
        resolve_bifrost_path_binding,
    )
    from omnimarket.inference.task_class_authority import load_task_class_authority
    from omnimarket.nodes.node_delegation_quality_gate_reducer.judge import (
        adapter_routing_resolved_judge as judge_module,
    )
    from omnimarket.nodes.node_delegation_routing_reducer.handlers import (
        handler_delegation_routing as reducer,
    )
    from omnimarket.nodes.node_llm_delegation_call_effect.handlers.transport import (
        probe_served_models,
    )
    from omnimarket.routing.routing_tiers_path import (
        ROUTING_TIERS_PACKAGED_DEFAULT_PATH,
        ROUTING_TIERS_PATH_ENV_KEY,
    )

    shipped = ROUTING_TIERS_PACKAGED_DEFAULT_PATH.resolve()
    bindings: dict[str, Any] = {}
    for key in (
        ROUTING_TIERS_PATH_ENV_KEY,
        "BIFROST_CONTRACT_PATH",
        "BIFROST_OVERLAY_PATH",
    ):
        raw = os.environ.get(key, "").strip()
        entry: dict[str, Any] = {"path": raw or None}
        if raw:
            bound = Path(raw).resolve()
            entry["git_ancestor"] = _git_ancestor(bound.parent)
            if key == ROUTING_TIERS_PATH_ENV_KEY:
                entry["matches_shipped"] = (
                    bound.is_file() and bound.read_bytes() == shipped.read_bytes()
                )
        bindings[key] = entry
    out["bindings"] = bindings
    if not bindings[ROUTING_TIERS_PATH_ENV_KEY]["path"]:
        # The reducer refuses an unbound key (OMN-15628). The walk grades the
        # SHIPPED file regardless, so bind it for THIS process only and say so.
        os.environ[ROUTING_TIERS_PATH_ENV_KEY] = str(shipped)
        bindings[ROUTING_TIERS_PATH_ENV_KEY]["collector_bound_to_shipped"] = True

    authority = load_task_class_authority()
    public = sorted(authority.public_task_classes)
    out["public_task_classes"] = public
    shipped_contract = yaml.safe_load(
        (
            ROUTING_TIERS_PACKAGED_DEFAULT_PATH.parent / "task_class_contracts.v1.yaml"
        ).read_text(encoding="utf-8")
    )
    raw_classes = (shipped_contract or {}).get("task_classes") or {}
    out["public_task_classes_from_contract_yaml"] = sorted(
        name
        for name, entry in raw_classes.items()
        if isinstance(entry, dict) and entry.get("gateway_exposure") == "public"
    )

    binding = resolve_bifrost_path_binding()
    merged = load_bifrost_delegation_config(
        config_path=binding.contract_path, overlay_path=binding.overlay_path
    )
    declarations = {b.backend_id: b for b in merged.backends if b.backend_id}
    loaded = reducer._load_bifrost_endpoints()
    config = reducer._get_config()
    contract = reducer._get_task_class_contract()
    tier_by_name = {tier.name: tier for tier in config.tiers}
    served_cache: dict[str, frozenset[str] | None] = {}

    walk: dict[str, list[dict[str, Any]]] = {}
    referenced: set[str] = set()
    for task_class in public:
        entry = reducer._task_class_entry(contract, task_class)
        declared_order = (
            ((entry or {}).get("escalation_policy") or {}).get("tier_order")
            if isinstance((entry or {}).get("escalation_policy"), dict)
            else None
        )
        names = (
            list(declared_order)
            if isinstance(declared_order, list) and declared_order
            else [t.name for t in reducer._tier_order_from_contract(config, entry)]
        )
        override = reducer._get_contract_model_ref(task_class, contract=contract)
        explicit = reducer._is_explicit_task_model_override(task_class, contract)
        tiers_out: list[dict[str, Any]] = []
        for name in names:
            tier = tier_by_name.get(name)
            if tier is None:
                tiers_out.append({"tier": name, "declared": False, "candidates": []})
                continue
            candidates = [m.backend_ref for m in tier.models if task_class in m.use_for]
            if explicit and override is not None:
                candidates += [m.backend_ref for m in tier.models if m.id == override]
            unique = sorted(set(candidates))
            referenced.update(unique)
            try:
                selected = reducer.backend_id_for_tier(
                    name, task_class, require_credential=False
                )
            except Exception as exc:  # noqa: BLE001 - recorded, graded by candidates
                selected = f"<error {type(exc).__name__}>"
            tiers_out.append(
                {
                    "tier": name,
                    "declared": True,
                    "candidates": unique,
                    "selected_without_credential": selected,
                    "paid": tier.cost_per_1k_tokens > 0,
                }
            )
        walk[task_class] = tiers_out
    out["walk"] = walk
    out["backends"] = {
        backend_id: _classify_backend(
            backend_id,
            declarations=declarations,
            loaded=loaded,
            served_cache=served_cache,
            probe_served_models=probe_served_models,
        )
        for backend_id in sorted(referenced)
    }

    out["controls"] = [
        {
            "name": "deleted_backend_control",
            "input": CONTROL_DELETED_BACKEND_ID,
            **_classify_backend(
                CONTROL_DELETED_BACKEND_ID,
                declarations=declarations,
                loaded=loaded,
                served_cache=served_cache,
                probe_served_models=probe_served_models,
            ),
        },
        {
            "name": "unreachable_endpoint_control",
            "input": CONTROL_UNREACHABLE_ENDPOINT,
            **_classify_private_endpoint(
                {"endpoint_private": endpoint_is_private(CONTROL_UNREACHABLE_ENDPOINT)},
                CONTROL_UNREACHABLE_ENDPOINT,
                "c14-control-model",
                served_cache,
                probe_served_models,
            ),
        },
    ]

    try:
        judge_backend_id = judge_module._DEFAULT_JUDGE_BACKEND_ID
        resolved = (
            judge_module.RoutingResolvedJudgeInferenceAdapter()._resolve_backend()
        )
        endpoint = str(resolved.endpoint_ref)
        judge: dict[str, Any] = {
            "backend_id": judge_backend_id,
            "endpoint": endpoint,
            "model_id": resolved.model_id,
            "requires_credential": bool(getattr(resolved, "secret_ref", None)),
            "provider": getattr(declarations.get(judge_backend_id), "provider", None),
            "endpoint_private": endpoint_is_private(endpoint),
        }
        if judge["endpoint_private"]:
            _classify_private_endpoint(
                judge,
                endpoint,
                str(resolved.model_id),
                served_cache,
                probe_served_models,
            )
        out["judge"] = judge
    except Exception as exc:  # noqa: BLE001 - an unresolvable reviewer is a finding
        out["judge"] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    return out


# ---------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------


def _run_collector(
    customer_python: str, env: Mapping[str, str], timeout: float
) -> dict[str, Any]:
    proc = subprocess.run(
        [customer_python, "-I", str(Path(__file__).resolve()), "--collect"],
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {"collect_error": (proc.stderr or proc.stdout).strip()[-1500:]}
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"collect_error": "collector printed no JSON: " + proc.stdout[-800:]}
    return (
        payload if isinstance(payload, dict) else {"collect_error": "non-object JSON"}
    )


def receipt_result_of(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """The delegation terminal a ``receipt.json`` carries, in either shape.

    A completed run's receipt holds the terminal directly under
    ``receipt.result``; a failed run's holds the workflow envelope there, with
    the terminal one level down under ``terminal_payload``. Reading only the
    first shape turns every failed run into "no attempts at all", which hides
    the ladder that explains the failure.
    """
    result = (receipt.get("receipt") or {}).get("result")
    if not isinstance(result, dict):
        return None
    terminal = result.get("terminal_payload")
    return terminal if isinstance(terminal, dict) else result


class _EgressRecorder:
    """A recording forward proxy: the instrument behind every absence claim.

    The delegation's inference call runs in a CHILD process the product starts
    itself (spawn on macOS, fork elsewhere), so an in-process hook in the parent
    sees nothing on one platform and something on the other. Proxy settings are
    environment, which every child inherits on every platform, and both of the
    product's transports (httpx and curl) honour them. So this sees every HTTP
    request any process of the run makes, plain requests by URL and TLS
    requests by their ``CONNECT host:port``.

    It is in the path, not beside it, and that is deliberate: a recorder the
    traffic can route around is how a zero goes unproven.
    """

    def __init__(self) -> None:
        import http.server
        import threading

        self.entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        recorder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: object) -> None:
                return

            def _record(self, method: str, target: str, host_url: str) -> None:
                with recorder._lock:
                    recorder.entries.append(
                        {
                            "method": method,
                            "target": target,
                            "private": endpoint_is_private(host_url),
                        }
                    )

            def do_CONNECT(self) -> None:
                host, _, port = self.path.partition(":")
                target = f"https://{host}:{port or 443}"
                self._record("CONNECT", self.path, target)
                address = _private_address(target)
                if address is None:
                    # Recorded, then refused: a customer-local probe never
                    # spends a metered call, and the record already grades it.
                    self.send_error(403, "c14 probe: non-private egress refused")
                    return
                try:
                    upstream = socket.create_connection(
                        (address, int(port or 443)), timeout=30
                    )
                except OSError:
                    self.send_error(502)
                    return
                self.send_response(200, "Connection Established")
                self.end_headers()
                _pipe(self.connection, upstream)

            def _forward(self) -> None:
                import http.client

                self._record(self.command, self.path, self.path)
                parts = urlsplit(self.path)
                address = _private_address(self.path)
                if address is None:
                    self.send_error(403, "c14 probe: non-private egress refused")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in {"proxy-connection", "connection", "keep-alive"}
                }
                path = parts.path + (f"?{parts.query}" if parts.query else "")
                try:
                    conn = http.client.HTTPConnection(
                        address, parts.port or 80, timeout=600
                    )
                    headers["Host"] = parts.netloc
                    conn.request(self.command, path or "/", body=body, headers=headers)
                    response = conn.getresponse()
                    payload = response.read()
                except OSError:
                    self.send_error(502)
                    return
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in {
                        "transfer-encoding",
                        "connection",
                        "content-length",
                    }:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

            do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _forward  # noqa: N815

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> _EgressRecorder:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _pipe(client: socket.socket, upstream: socket.socket) -> None:
    import selectors

    selector = selectors.DefaultSelector()
    selector.register(client, selectors.EVENT_READ, upstream)
    selector.register(upstream, selectors.EVENT_READ, client)
    try:
        while True:
            events = selector.select(timeout=600)
            if not events:
                return
            for key, _ in events:
                data = key.fileobj.recv(65536)  # type: ignore[union-attr]
                if not data:
                    return
                key.data.sendall(data)
    except OSError:
        return
    finally:
        selector.close()
        upstream.close()


def _run_delegation(
    onex: str,
    env: Mapping[str, str],
    work_dir: Path,
    *,
    prompt: str,
    task_type: str,
    timeout: float,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    with _EgressRecorder() as recorder:
        run_env = dict(env)
        for key in ("NO_PROXY", "no_proxy"):
            run_env.pop(key, None)
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            run_env[key] = recorder.url
        try:
            proc = subprocess.run(
                [onex, "delegate", prompt, "--task-type", task_type],
                env=run_env,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"onex delegate did not finish within {timeout:.0f}s"}
        egress = list(recorder.entries)
    run: dict[str, Any] = {
        "exit_code": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
        "egress": egress,
    }
    try:
        skill_result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        run["terminal_error"] = "stdout carried no JSON: " + proc.stdout[-600:]
        skill_result = {}
    run_id = skill_result.get("run_id") if isinstance(skill_result, dict) else None
    receipt_path = work_dir / ".onex_state" / "runs" / str(run_id) / "receipt.json"
    if run_id and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        run["receipt_result"] = receipt_result_of(receipt)
        run["receipt_path"] = str(receipt_path)
    elif isinstance(skill_result, dict):
        terminal = (skill_result.get("result") or {}).get("terminal_payload") or {}
        run["terminal_error"] = terminal.get("error_message") or run.get(
            "terminal_error"
        )
    return run


def _render_summary(record: Record, *, as_of: str) -> str:
    lines = [
        "## C14 -- routing on a no-checkout customer install (OMN-19180)",
        "",
        f"**Verdict: `{record.verdict}`** at `{as_of}`.",
        "",
        record.detail,
        "",
        "| Row | Holds | Findings |",
        "| --- | --- | --- |",
    ]
    for row in record.rows:
        findings = "<br>".join(row.findings) if row.findings else "—"
        lines.append(f"| {row.title} | {'yes' if row.ok else '**no**'} | {findings} |")
    lines += ["", "Controls:", ""]
    for control in record.controls:
        lines.append(f"- `{control.get('name')}` -> `{control.get('classification')}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--customer-python",
        default="",
        help="Interpreter of the no-checkout customer install. Required unless --replay.",
    )
    parser.add_argument("--work-dir", default="", help="Where the delegation runs.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--task-type", default=DEFAULT_TASK_TYPE)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument(
        "--replay",
        default="",
        help="Grade a recorded {collection, run} JSON instead of probing a machine.",
    )
    parser.add_argument("--record", default="", help="Write the JSON record here.")
    parser.add_argument("--summary", default="", help="Append a markdown summary here.")
    args = parser.parse_args(argv)

    if args.collect:  # pragma: no cover - customer-interpreter half
        print(json.dumps(collect(), sort_keys=True, default=str))
        return EXIT_OK

    as_of = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if args.replay:
            payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            collection = payload.get("collection")
            run = payload.get("run")
            if not isinstance(collection, dict):
                raise ProbeInputError("replay payload has no 'collection' object")
        else:
            if not args.customer_python:
                raise ProbeInputError("--customer-python is required unless --replay")
            if not Path(args.customer_python).is_file():
                raise ProbeInputError(f"no interpreter at {args.customer_python!r}")
            env = customer_environment(os.environ)
            work_dir = Path(args.work_dir or Path.cwd() / "c14-work").resolve()
            collection = _run_collector(args.customer_python, env, timeout=args.timeout)
            onex = Path(args.customer_python).parent / "onex"
            if not onex.is_file():
                raise ProbeInputError(f"no installed onex console script at {onex}")
            run = _run_delegation(
                str(onex),
                env,
                work_dir,
                prompt=args.prompt,
                task_type=args.task_type,
                timeout=args.timeout,
            )
            payload = {"collection": collection, "run": run}
            (work_dir / "c14-observation.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        record = grade(collection, run if isinstance(run, dict) else None)
    except ProbeInputError as exc:
        print(f"::error::C14 probe could not run: {exc}", file=sys.stderr)
        return EXIT_INPUT

    out = record.to_dict(as_of=as_of)
    if args.record:
        Path(args.record).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_render_summary(record, as_of=as_of))

    for row in record.rows:
        print(f"{row.name:<32} holds={row.ok}")
        for finding in row.findings:
            print(f"    - {finding}")
    for error in record.input_errors:
        print(f"::error::C14 could not run: {error}", file=sys.stderr)
    if record.verdict == "FAIL":
        print(f"::error::C14 RED -- {record.detail}", file=sys.stderr)
    elif record.verdict == "PASS":
        print(f"C14 GREEN -- {record.detail}")
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
