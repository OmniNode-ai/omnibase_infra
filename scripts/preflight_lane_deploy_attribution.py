#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane deploy ATTRIBUTION recorder + live-grant INTERLOCK (OMN-15218).

Twice in two days the ``.201`` stability-test lane was rebuilt/restarted with
**no attributable trigger** while live, unconsumed prod-promotion grants were
pinned to the digests that rebuild replaced:

  * 2026-07-26T21:45:15Z — lane rebuilt to a local-workspace image while grant
    ``grant-6dbeae94`` was live and pinned to the PREVIOUS digests.
  * 2026-07-27T10:05:43-10:09:07Z — stability containers restarted by an
    unknown actor while the three ``batch-b551aa00`` grants were live.

Neither event named an actor, a reason, or a ticket, and neither was blocked or
even warned about. The prod-promotion gate (OMN-13418) resolves
"stability-proven" from the LIVE stability container at grant-issuance time, so
an unattributed stability rebuild between issuance and consumption silently
erodes the premise every live grant rests on. Two occurrences make this a
mechanism gap, not an incident — a rule is not a mechanism.

This module is that mechanism. It is a **pre-mutation preflight** invoked from
the sanctioned deploy path (``scripts/deploy-runtime.sh`` and the lane refresh
tooling in ``scripts/runtime_build/``) BEFORE anything is tagged, built,
recreated, or restarted. It enforces two rules:

1. **ATTRIBUTION (fail-fast).** Every deploy/restart of a governed lane
   (``stability-test``, ``prod``, ``judge``) must declare *why* it is happening:
   ``ONEX_DEPLOY_REASON`` is mandatory and must be a real sentence, not a
   placeholder. Actor identity (user, uid, host, ssh peer, parent command),
   the invoking command, the resolved ticket, and the grant verdict are written
   to a durable JSONL deploy log plus a per-run attribution record, and are
   folded into ``registry.json`` by the caller. An unattributed rebuild becomes
   impossible on this path and traceable after the fact everywhere else.

2. **GRANT INTERLOCK (fail-closed, refuse-by-default).** When the target lane is
   ``stability-test`` and ``onex_change_control`` carries unconsumed, unexpired
   prod-promotion grants at ``@main``, the deploy REFUSES and names every live
   grant. The only override is an explicit acknowledgement
   (``ONEX_DEPLOY_GRANT_ACK``) that must name **each** live ``grant_id``; the
   acknowledgement itself is recorded in the attribution record, so overriding
   is attributable rather than silent. A blanket ``true`` does not work — a
   stale acknowledgement left in an environment cannot pre-authorize a grant
   that did not exist when it was set.

   Grant state is resolved from ``onex_change_control@main`` (never a PR
   branch), exactly like the OMN-13418 resolver's I/O boundary. If that state
   cannot be established — no clone, fetch failure, unparseable YAML, malformed
   entry — the verdict is ``UNREADABLE`` and the deploy REFUSES. Indeterminate
   grant state is not a pass; it needs the ``unreadable-grant-state``
   acknowledgement token to proceed.

Layering note: ``omnibase_infra`` must not import ``omnimarket`` (where the
OMN-13439 grant resolver EFFECT lives). The grant FILE is the contract surface
— this module parses that YAML directly and imports nothing from the governance
or market repos, the same way the resolver imports nothing from
``onex_change_control``.

Test seam: every I/O boundary is injectable. ``--grants-file`` substitutes a
local file for the ``@main`` fetch and ``--now`` pins evaluation time, so the
whole verdict table is exercised hermetically — no lane contact, no network, no
real ``@main`` dependency (faithful dependency substitution, not mocks).

Usage::

    # From deploy-runtime.sh / refresh_stability_lane.sh, before any mutation:
    ONEX_DEPLOY_REASON="OMN-15181 prod bootstrap rehearsal" \\
      uv run python scripts/preflight_lane_deploy_attribution.py \\
        --lane stability-test \\
        --compose-project omnibase-infra-stability-test \\
        --source deploy-runtime.sh \\
        --invoking-command "deploy-runtime.sh --execute --force --restart"

    # Evaluate without writing the durable record (dry-run/preview):
    ... --check-only

Exit codes:
    0 — allowed; attribution recorded (JSON record on stdout)
    1 — REFUSED (missing attribution, live grants, or unreadable grant state)
    2 — usage / environment error
"""

from __future__ import annotations

import argparse
import enum
import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# --- policy constants --------------------------------------------------------

#: Lanes whose deploys/restarts require durable attribution. ``dev`` is
#: deliberately excluded: it is the fully-mutable test platform (CLAUDE.md lane
#: table) and gating it would push operators off the sanctioned path.
GOVERNED_LANES: frozenset[str] = frozenset({"stability-test", "prod", "judge"})

#: Lanes the live-grant interlock applies to. Prod promotion is already gated by
#: the OMN-13418 grant gate itself; the hole this closes is the PRE-prod lane
#: whose rebuild invalidates the "stability-proven" premise of a live grant.
GRANT_INTERLOCK_LANES: frozenset[str] = frozenset({"stability-test"})

#: The grant registry path inside the ``onex_change_control`` repo.
GRANTS_REPO_RELPATH = "grants/prod_promotion_grants.yaml"

#: The ref the grant registry is resolved from. Never a PR branch (OMN-13418).
GRANTS_REF = "main"

#: Env var carrying the mandatory human reason for the deploy.
ENV_REASON = "ONEX_DEPLOY_REASON"

#: Env var carrying an explicit ticket id (otherwise extracted from the reason).
ENV_TICKET = "ONEX_DEPLOY_TICKET"

#: Env var carrying the live-grant acknowledgement (must name each grant_id).
ENV_GRANT_ACK = "ONEX_DEPLOY_GRANT_ACK"

#: Optional actor override for automation that knows its own identity better
#: than ``getpass.getuser()`` does (e.g. a CI runner naming its workflow run).
ENV_ACTOR = "ONEX_DEPLOY_ACTOR"

#: Acknowledgement token required to proceed past an UNREADABLE grant state.
UNREADABLE_ACK_TOKEN = "unreadable-grant-state"

#: A reason must clear this length AND not be one of the placeholders below.
MIN_REASON_LENGTH = 12

#: Placeholder reasons that carry no information. Rejected so the mandatory
#: field cannot be satisfied with a keystroke (the OMN-15218 failure mode is
#: *unattributed* rebuilds; "x" is unattributed with extra steps).
PLACEHOLDER_REASONS: frozenset[str] = frozenset(
    {
        "",
        "-",
        ".",
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "todo",
        "test",
        "testing",
        "deploy",
        "redeploy",
        "restart",
        "rebuild",
        "update",
        "wip",
        "x",
        "xxx",
        "asdf",
        "reason",
    }
)

#: Required fields on a grant entry (OMN-13437 schema). A missing field makes
#: the registry unreadable rather than "no live grants".
REQUIRED_GRANT_FIELDS: frozenset[str] = frozenset(
    {
        "grant_id",
        "runtime_lane",
        "image_digest",
        "promotion_batch_id",
        "approved_by",
        "expires_at",
        "created_at",
        "reason",
    }
)

#: Bound on the git calls used to resolve grant state. A hung fetch must fail
#: closed on a timer, not block a deploy forever.
GIT_TIMEOUT_SECONDS = 60

SCHEMA_VERSION = "1.0.0"


class EnumGrantVerdict(enum.Enum):
    """Outcome of resolving live grant state for the target lane."""

    #: Lane is outside :data:`GRANT_INTERLOCK_LANES`; interlock did not run.
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: Grant state read successfully; no unconsumed, unexpired grants.
    CLEAR = "CLEAR"
    #: Grant state read successfully; live grants exist and pin this lane's proof.
    LIVE_GRANTS = "LIVE_GRANTS"
    #: Grant state could NOT be established. Fails closed.
    UNREADABLE = "UNREADABLE"


class PreflightError(RuntimeError):
    """Usage / environment error (exit 2). Never a policy refusal."""


# --- pure helpers ------------------------------------------------------------


def lane_from_compose_project(compose_project: str) -> str:
    """Derive the lane name from a compose project name.

    ``omnibase-infra`` -> ``dev``; ``omnibase-infra-<lane>`` -> ``<lane>``.
    Mirrors the derivation deploy-runtime.sh uses for its overlay and hot-patch
    gates so one deploy cannot be attributed to two different lane names.
    """
    lane = compose_project.removeprefix("omnibase-infra").removeprefix("-")
    return lane or "dev"


def normalize_reason(raw: str | None) -> str:
    """Collapse whitespace on a reason string (``None`` -> empty)."""
    return " ".join((raw or "").split())


def reason_is_meaningful(reason: str) -> bool:
    """Whether a reason is a real justification rather than a placeholder."""
    collapsed = normalize_reason(reason)
    if collapsed.strip(".- ").lower() in PLACEHOLDER_REASONS:
        return False
    return len(collapsed) >= MIN_REASON_LENGTH


def extract_ticket(*candidates: str | None) -> str:
    """Return the first ``OMN-<digits>`` token found in the candidates."""
    for candidate in candidates:
        if not candidate:
            continue
        for token in candidate.replace(",", " ").replace("/", " ").split():
            stripped = token.strip("()[]{}:;.,").upper()
            if stripped.startswith("OMN-") and stripped[4:].isdigit():
                return stripped
    return ""


def parse_ack_tokens(raw: str | None) -> tuple[str, ...]:
    """Split an acknowledgement env value into normalized tokens."""
    if not raw:
        return ()
    for separator in (",", ";"):
        raw = raw.replace(separator, " ")
    return tuple(token.strip().lower() for token in raw.split() if token.strip())


def _coerce_datetime(value: Any) -> datetime:
    """Coerce a grant timestamp (ISO-8601, ``Z`` allowed) to an aware datetime."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(
            f"grant timestamp must be a datetime or ISO-8601 string; got {value!r}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def evaluate_grant_state(raw: bytes, *, now: datetime) -> dict[str, Any]:
    """Evaluate raw grant-registry bytes into a verdict block. Pure — no I/O.

    Returns a dict with ``verdict``, ``live_grants``, ``grants_sha256`` and
    ``errors``. Any structural surprise (non-mapping root, missing ``entries``
    list, entry missing a required field, unparseable timestamp) yields
    ``UNREADABLE`` — an unparseable registry must never read as "no grants".
    """
    block: dict[str, Any] = {
        "verdict": EnumGrantVerdict.UNREADABLE.value,
        "grants_sha256": hashlib.sha256(raw).hexdigest(),
        "live_grants": [],
        "errors": [],
    }

    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        block["errors"].append(f"grant registry does not parse as YAML: {exc}")
        return block

    if document is None:
        block["errors"].append("grant registry is empty (expected an 'entries' key)")
        return block
    if not isinstance(document, Mapping):
        block["errors"].append(
            f"grant registry root must be a mapping; got {type(document).__name__}"
        )
        return block

    entries = document.get("entries")
    if entries is None:
        block["errors"].append("grant registry has no 'entries' key")
        return block
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        block["errors"].append(
            f"grant registry 'entries' must be a list; got {type(entries).__name__}"
        )
        return block

    live: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            block["errors"].append(f"entry[{index}] is not a mapping")
            return block
        missing = sorted(REQUIRED_GRANT_FIELDS - set(entry))
        if missing:
            block["errors"].append(
                f"entry[{index}] is missing required field(s): {', '.join(missing)}"
            )
            return block
        try:
            expires_at = _coerce_datetime(entry["expires_at"])
        except ValueError as exc:
            block["errors"].append(
                f"entry[{index}] ({entry['grant_id']}) has an unparseable expires_at: {exc}"
            )
            return block

        # A consumed grant is spent (OMN-13424) and no longer pins this lane.
        if entry.get("consumed", False) is True:
            continue
        if expires_at <= now:
            continue

        live.append(
            {
                "grant_id": str(entry["grant_id"]),
                "runtime_lane": str(entry["runtime_lane"]),
                "image_digest": str(entry["image_digest"]),
                "promotion_batch_id": str(entry["promotion_batch_id"]),
                "approved_by": str(entry["approved_by"]),
                "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
            }
        )

    block["live_grants"] = live
    block["verdict"] = (
        EnumGrantVerdict.LIVE_GRANTS if live else EnumGrantVerdict.CLEAR
    ).value
    return block


def apply_acknowledgement(
    grant_block: Mapping[str, Any], ack_tokens: Sequence[str]
) -> dict[str, Any]:
    """Decide whether an acknowledgement clears the grant verdict.

    ``LIVE_GRANTS`` clears only when every live ``grant_id`` is named. That is
    what makes the override attributable rather than a blanket switch: an
    acknowledgement set before a grant existed cannot name it, so it cannot
    silently authorize the next rebuild.

    ``UNREADABLE`` clears only with the explicit
    :data:`UNREADABLE_ACK_TOKEN` sentinel.
    """
    verdict = str(grant_block.get("verdict"))
    lowered = {token.lower() for token in ack_tokens}
    result: dict[str, Any] = {
        "acknowledged": False,
        "acknowledgement_tokens": list(ack_tokens),
        "unacknowledged_grant_ids": [],
    }

    if verdict == EnumGrantVerdict.LIVE_GRANTS.value:
        live_ids = [
            str(grant["grant_id"]) for grant in grant_block.get("live_grants", [])
        ]
        missing = [grant_id for grant_id in live_ids if grant_id.lower() not in lowered]
        result["unacknowledged_grant_ids"] = missing
        result["acknowledged"] = not missing
    elif verdict == EnumGrantVerdict.UNREADABLE.value:
        result["acknowledged"] = UNREADABLE_ACK_TOKEN in lowered

    return result


def collect_actor(env: Mapping[str, str]) -> dict[str, Any]:
    """Collect the durable actor identity for this invocation.

    Records everything cheaply available that distinguishes "who did this":
    login user, uid, host, the ssh peer (which is what an unattributed remote
    rebuild would otherwise hide), and the parent process command line.
    """
    try:
        user = getpass.getuser()
    except (KeyError, OSError):  # pragma: no cover - only fails on exotic hosts
        user = env.get("USER", "unknown")

    actor: dict[str, Any] = {
        "user": env.get("SUDO_USER") or user,
        "effective_user": user,
        "uid": os.getuid(),
        "host": socket.gethostname(),
        "ssh_connection": env.get("SSH_CONNECTION", ""),
        "ssh_client": env.get("SSH_CLIENT", ""),
        "declared_actor": env.get(ENV_ACTOR, ""),
        "parent_command": _parent_command(),
        "ci": env.get("GITHUB_RUN_ID", ""),
    }
    actor["identity"] = actor["declared_actor"] or f"{actor['user']}@{actor['host']}"
    return actor


def _parent_command() -> str:
    """Best-effort parent-process command line (empty string when unavailable)."""
    try:
        completed = subprocess.run(
            ["ps", "-o", "args=", "-p", str(os.getppid())],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):  # pragma: no cover - platform dependent
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


# --- I/O boundary ------------------------------------------------------------


def fetch_grant_bytes_from_main(grants_repo: Path) -> tuple[bytes, str]:
    """Read the grant registry from ``onex_change_control@main``.

    Returns ``(raw_bytes, resolved_commit_sha)``. Raises :class:`OSError` /
    :class:`subprocess.SubprocessError` derivatives on any failure so the caller
    can fail closed. Never falls back to the working tree or a PR branch — the
    ``@main`` anchor is the anti-self-issue property of the whole grant scheme.
    """
    if not (grants_repo / ".git").exists():
        raise FileNotFoundError(f"not a git clone: {grants_repo}")

    def _git(*args: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={grants_repo}",
                "-C",
                str(grants_repo),
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            raise ChildProcessError(
                f"git {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout

    # Refresh first: a stale local origin/main would hide a grant that landed
    # minutes ago, which is exactly the window this interlock exists to cover.
    _git("fetch", "origin", GRANTS_REF, "--quiet")
    commit = _git("rev-parse", f"origin/{GRANTS_REF}").strip()
    raw = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={grants_repo}",
            "-C",
            str(grants_repo),
            "show",
            f"origin/{GRANTS_REF}:{GRANTS_REPO_RELPATH}",
        ],
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if raw.returncode != 0:
        raise ChildProcessError(
            f"git show origin/{GRANTS_REF}:{GRANTS_REPO_RELPATH} failed: {raw.stderr.decode(errors='replace').strip()}"
        )
    return raw.stdout, commit


def resolve_grant_block(
    *,
    lane: str,
    now: datetime,
    grants_file: Path | None,
    grants_repo: Path | None,
) -> dict[str, Any]:
    """Resolve the grant verdict block for ``lane``, failing closed on any error."""
    if lane not in GRANT_INTERLOCK_LANES:
        return {
            "verdict": EnumGrantVerdict.NOT_APPLICABLE.value,
            "source": "",
            "grants_commit": "",
            "grants_sha256": "",
            "live_grants": [],
            "errors": [],
        }

    if grants_file is not None:
        # Offline / test substitution: the exact same evaluation over a local
        # file. Used by the unit suite and by an operator diagnosing a verdict.
        try:
            raw = grants_file.read_bytes()
        except OSError as exc:
            return {
                "verdict": EnumGrantVerdict.UNREADABLE.value,
                "source": str(grants_file),
                "grants_commit": "",
                "grants_sha256": "",
                "live_grants": [],
                "errors": [f"could not read grant registry file: {exc}"],
            }
        block = evaluate_grant_state(raw, now=now)
        block["source"] = str(grants_file)
        block["grants_commit"] = ""
        return block

    if grants_repo is None:
        return {
            "verdict": EnumGrantVerdict.UNREADABLE.value,
            "source": "",
            "grants_commit": "",
            "grants_sha256": "",
            "live_grants": [],
            "errors": [
                "no onex_change_control clone resolved (set OMNI_HOME or pass "
                "--grants-repo) — cannot establish live grant state at "
                f"onex_change_control@{GRANTS_REF}"
            ],
        }

    try:
        raw, commit = fetch_grant_bytes_from_main(grants_repo)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Fail closed on ANY resolution failure: missing clone (FileNotFoundError),
        # failed fetch/show (ChildProcessError, an OSError), timeout
        # (subprocess.TimeoutExpired), or a decode/format surprise.
        return {
            "verdict": EnumGrantVerdict.UNREADABLE.value,
            "source": f"{grants_repo}@origin/{GRANTS_REF}",
            "grants_commit": "",
            "grants_sha256": "",
            "live_grants": [],
            "errors": [
                f"could not resolve grant registry at origin/{GRANTS_REF}: {exc}"
            ],
        }

    block = evaluate_grant_state(raw, now=now)
    block["source"] = f"{grants_repo}@origin/{GRANTS_REF}:{GRANTS_REPO_RELPATH}"
    block["grants_commit"] = commit
    return block


# --- record assembly ---------------------------------------------------------


def build_record(
    *,
    lane: str,
    compose_project: str,
    source: str,
    invoking_command: str,
    mode: str,
    env: Mapping[str, str],
    now: datetime,
    grant_block: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the attribution record and its ALLOW/REFUSE verdict. Pure."""
    reason = normalize_reason(env.get(ENV_REASON))
    ack_tokens = parse_ack_tokens(env.get(ENV_GRANT_ACK))
    ack = apply_acknowledgement(grant_block, ack_tokens)

    refusals: list[str] = []
    attribution_required = lane in GOVERNED_LANES

    if attribution_required and not reason:
        refusals.append(
            f"{ENV_REASON} is not set. Every {lane} deploy/restart must declare why it is "
            "happening (OMN-15218: two unattributed stability rebuilds in two days)."
        )
    elif attribution_required and not reason_is_meaningful(reason):
        refusals.append(
            f"{ENV_REASON}={reason!r} is a placeholder. Give a real justification "
            f"(>= {MIN_REASON_LENGTH} chars, ideally naming a ticket)."
        )

    verdict = str(grant_block.get("verdict"))
    live_grants = list(grant_block.get("live_grants", []))
    if verdict == EnumGrantVerdict.LIVE_GRANTS.value and not ack["acknowledged"]:
        named = ", ".join(
            f"{grant['grant_id']} (lane={grant['runtime_lane']}, digest={grant['image_digest'][:19]}…, "
            f"batch={grant['promotion_batch_id']}, expires={grant['expires_at']})"
            for grant in live_grants
        )
        refusals.append(
            f"{len(live_grants)} live prod-promotion grant(s) pin the current {lane} proof: {named}. "
            f"Refreshing {lane} now invalidates the stability-proven premise those grants rest on "
            "(OMN-15218 / OMN-13418). To proceed anyway, acknowledge every grant explicitly: "
            f'{ENV_GRANT_ACK}="{",".join(str(g["grant_id"]) for g in live_grants)}" — the '
            "acknowledgement is recorded in the attribution record."
        )
    elif verdict == EnumGrantVerdict.UNREADABLE.value and not ack["acknowledged"]:
        refusals.append(
            "live prod-promotion grant state is UNREADABLE, so this deploy cannot be proven safe: "
            + "; ".join(str(err) for err in grant_block.get("errors", []))
            + f". Fail-closed (OMN-15218). Fix the grant source, or set {ENV_GRANT_ACK}="
            f"{UNREADABLE_ACK_TOKEN} to proceed on the record."
        )

    ticket = extract_ticket(env.get(ENV_TICKET), reason)

    return {
        "schema_version": SCHEMA_VERSION,
        "mechanism_ticket": "OMN-15218",
        "ts_utc": now.isoformat().replace("+00:00", "Z"),
        "lane": lane,
        "compose_project": compose_project,
        "source": source,
        "mode": mode,
        "invoking_command": invoking_command,
        "reason": reason,
        "ticket": ticket,
        "attribution_required": attribution_required,
        "actor": collect_actor(env),
        "grant_guard": {
            "applies": lane in GRANT_INTERLOCK_LANES,
            "verdict": verdict,
            "grants_ref": f"onex_change_control@{GRANTS_REF}",
            "source": grant_block.get("source", ""),
            "grants_commit": grant_block.get("grants_commit", ""),
            "grants_sha256": grant_block.get("grants_sha256", ""),
            "live_grants": live_grants,
            "errors": list(grant_block.get("errors", [])),
            **ack,
        },
        "result": "REFUSE" if refusals else "ALLOW",
        "refusal_reasons": refusals,
    }


def write_record(record: Mapping[str, Any], record_dir: Path) -> dict[str, str]:
    """Persist the record: one JSONL deploy-log line + one per-run JSON file.

    The JSONL log is append-only and is the surface a future session reads to
    answer "who rebuilt this lane, when, and why" without re-deriving a forensic
    chain by hand. Both REFUSE and ALLOW records are written — a refused deploy
    attempt is itself attribution-worthy.
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    attribution_dir = record_dir / "deploy-attribution"
    attribution_dir.mkdir(parents=True, exist_ok=True)

    stamp = str(record["ts_utc"]).replace(":", "").replace("-", "")
    record_path = (
        attribution_dir / f"{stamp}-{record['lane']}-{record['result'].lower()}.json"
    )
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log_path = record_dir / "deploy-log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {"record_path": str(record_path), "log_path": str(log_path)}


# --- CLI ---------------------------------------------------------------------


def default_grants_repo(env: Mapping[str, str]) -> Path | None:
    """Resolve the ``onex_change_control`` clone from ``OMNI_HOME`` (may be None)."""
    omni_home = env.get("OMNI_HOME", "").strip()
    if not omni_home:
        return None
    return Path(omni_home) / "onex_change_control"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record lane-deploy attribution and enforce the live-grant interlock (OMN-15218).",
    )
    parser.add_argument(
        "--lane",
        default="",
        help="Target lane (derived from --compose-project when omitted).",
    )
    parser.add_argument(
        "--compose-project", default="", help="Compose project of the target lane."
    )
    parser.add_argument(
        "--source", default="unknown", help="Which entrypoint invoked this preflight."
    )
    parser.add_argument(
        "--invoking-command", default="", help="The command line being guarded."
    )
    parser.add_argument(
        "--record-dir",
        default="",
        help="Directory for deploy-log.jsonl (default: ~/.omnibase/infra).",
    )
    parser.add_argument(
        "--grants-repo",
        default="",
        help="onex_change_control clone (default: $OMNI_HOME/onex_change_control).",
    )
    parser.add_argument(
        "--grants-file",
        default="",
        help="Read grant state from this file instead of @main (offline/test).",
    )
    parser.add_argument(
        "--now", default="", help="ISO-8601 evaluation time (default: now, UTC)."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Evaluate and print; do not write the durable record.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the JSON record on stdout (no human summary).",
    )
    return parser


def _print_human_summary(record: Mapping[str, Any], stream: Any) -> None:
    grant_guard = record["grant_guard"]
    print(
        f"[lane-deploy-attribution] lane            : {record['lane']} ({record['compose_project']})",
        file=stream,
    )
    print(
        f"[lane-deploy-attribution] actor           : {record['actor']['identity']} (uid={record['actor']['uid']})",
        file=stream,
    )
    if record["actor"]["ssh_connection"]:
        print(
            f"[lane-deploy-attribution] ssh             : {record['actor']['ssh_connection']}",
            file=stream,
        )
    print(
        f"[lane-deploy-attribution] reason          : {record['reason'] or '<MISSING>'}",
        file=stream,
    )
    print(
        f"[lane-deploy-attribution] ticket          : {record['ticket'] or '<none>'}",
        file=stream,
    )
    print(
        f"[lane-deploy-attribution] invoked by      : {record['source']}", file=stream
    )
    print(
        f"[lane-deploy-attribution] grant verdict   : {grant_guard['verdict']} ({len(grant_guard['live_grants'])} live)",
        file=stream,
    )
    for grant in grant_guard["live_grants"]:
        print(
            f"[lane-deploy-attribution]   - {grant['grant_id']} lane={grant['runtime_lane']} "
            f"digest={grant['image_digest']} batch={grant['promotion_batch_id']} expires={grant['expires_at']}",
            file=stream,
        )
    if grant_guard["acknowledged"]:
        print(
            f"[lane-deploy-attribution] acknowledged    : {', '.join(grant_guard['acknowledgement_tokens'])}",
            file=stream,
        )
    print(
        f"[lane-deploy-attribution] result          : {record['result']}", file=stream
    )
    for reason in record["refusal_reasons"]:
        print(f"[lane-deploy-attribution] REFUSED: {reason}", file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ

    compose_project = args.compose_project.strip()
    lane = args.lane.strip()
    if not lane and not compose_project:
        print("ERROR: one of --lane / --compose-project is required.", file=sys.stderr)
        return 2
    if not lane:
        lane = lane_from_compose_project(compose_project)
    if not compose_project:
        compose_project = (
            "omnibase-infra" if lane == "dev" else f"omnibase-infra-{lane}"
        )

    if args.now:
        try:
            now = _coerce_datetime(args.now)
        except ValueError as exc:
            print(f"ERROR: --now is not an ISO-8601 datetime: {exc}", file=sys.stderr)
            return 2
    else:
        now = datetime.now(UTC)

    grants_file = Path(args.grants_file).expanduser() if args.grants_file else None
    grants_repo = (
        Path(args.grants_repo).expanduser()
        if args.grants_repo
        else default_grants_repo(env)
    )

    grant_block = resolve_grant_block(
        lane=lane,
        now=now,
        grants_file=grants_file,
        grants_repo=grants_repo,
    )

    record = build_record(
        lane=lane,
        compose_project=compose_project,
        source=args.source,
        invoking_command=args.invoking_command,
        mode="check-only" if args.check_only else "execute",
        env=env,
        now=now,
        grant_block=grant_block,
    )

    if not args.check_only:
        record_dir = (
            Path(args.record_dir).expanduser()
            if args.record_dir
            else Path.home() / ".omnibase" / "infra"
        )
        try:
            record["written"] = write_record(record, record_dir)
        except OSError as exc:
            # A record we cannot persist is not attribution. Fail, do not warn.
            print(
                f"ERROR: could not write the attribution record under {record_dir}: {exc}",
                file=sys.stderr,
            )
            return 2

    if not args.json:
        _print_human_summary(record, sys.stderr)
    print(json.dumps(record, sort_keys=True))

    return 1 if record["result"] == "REFUSE" else 0


if __name__ == "__main__":
    sys.exit(main())
