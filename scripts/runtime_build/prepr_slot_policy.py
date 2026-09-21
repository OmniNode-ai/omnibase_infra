#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-PR verify pool's slot policy table (OMN-18893, epic OMN-18888).

Why this is a module and not a case statement in the entrypoint
---------------------------------------------------------------
Task 4's acceptance criterion is that the entrypoint's refusal of every
declared lane is "asserted by a test that reads the policy table rather than
matching text". A ``case`` arm inside a 700-line shell script is only testable
by grepping the script, and a grep passes just as happily against a comment
that names the lane as against a branch that refuses it. So the table lives
here, the shell reads it through ``--json``, and the tests read the same
dictionaries the shell does. One declaration, two readers, no transcription.

What a slot is
--------------
A slot is an ephemeral compose project that runs ONE branch's runtime family
against the dev lane's shared Postgres, Redpanda, Valkey and Keycloak,
isolated on every axis by namespace: a topic prefix, a database-name suffix
with its own principals, a consumer-group environment token, a Valkey logical
index and its own tenant. It is destroyed when its verification ends.

A slot is a premise for NOTHING. It is deliberately outside ``GOVERNED_LANES``
and ``GRANT_INTERLOCK_LANES`` in ``scripts/preflight_lane_deploy_attribution.py``,
sources no ``stability-proven`` digest, and must never appear in a promotion's
evidence chain. Rule 24(e) in ``omni_home``'s CLAUDE.md sanctions building one
only through ``scripts/runtime_build/prepr_verify_lane.sh``, which is the sole
consumer of this table.

The refusal set is spelled by COMPOSE PROJECT, not by lane word
---------------------------------------------------------------
``resolve_lane_name`` in ``compose_files.sh`` derives a lane from a compose
project by stripping the ``omnibase-infra`` prefix, so "the dev lane" is the
EMPTY suffix. A refusal table keyed on lane words would therefore have to
refuse the empty string, which is indistinguishable from an unset argument.
Keying on the full compose project makes the dev lane an ordinary entry and
removes that special case entirely.

The table is CLOSED in both directions. A compose project that is neither a
declared lane nor a pool slot is refused as unknown rather than built: a typo
in a project name must not mint a thirteenth lane on the lab host.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Final, NamedTuple

#: Exit codes. Named, stable, and asserted by
#: ``tests/unit/scripts/test_prepr_verify_lane_entrypoint_omn18893.py`` so a
#: caller can branch on WHICH refusal fired rather than on a message string.
EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 2
EXIT_REFUSED_DECLARED_LANE: Final[int] = 3
EXIT_REFUSED_UNKNOWN_PROJECT: Final[int] = 4
EXIT_REFUSED_DIRTY_WORKTREE: Final[int] = 5
EXIT_REFUSED_ATTRIBUTION: Final[int] = 6
EXIT_BUILD_LOCK_CONTENDED: Final[int] = 7
EXIT_SLOT_LOCK_CONTENDED: Final[int] = 8
EXIT_PROVISION_FAILED: Final[int] = 9
EXIT_BUILD_FAILED: Final[int] = 10
EXIT_BOOT_FAILED: Final[int] = 11
EXIT_PROVENANCE_MISMATCH: Final[int] = 12

#: Every compose project this entrypoint refuses BY NAME, with the reason it
#: refuses it. Rule 24(e): the new entrypoint "refuses every declared lane by
#: name" and "accepts no lane argument at all".
#:
#: ``omnibase-infra-prod`` is here although OMN-18320 shut that lane down, for
#: the same reason the raw-bypass matcher still names it: the refusal must hold
#: whether or not a lane by that name is currently running.
DECLARED_LANE_PROJECTS: Final[dict[str, str]] = {
    "omnibase-infra": (
        "the dev lane. It is the shared dependency server this pool BORROWS "
        "(Postgres, Redpanda, Valkey, Keycloak by service hostname on its "
        "network); a slot never mutates its containers."
    ),
    "omnibase-infra-stability-test": (
        "the stability lane. Its readiness projection is the 'stability-proven' "
        "premise of a live compose-path prod grant (rule 12), and a slot must "
        "never be able to touch it."
    ),
    "omnibase-infra-judge": "the judge lane, declared NOT authorized for mutation.",
    "omnibase-infra-prod": (
        "the production compose project. Retired under OMN-18320 and refused "
        "anyway, because the refusal must not depend on the lane running."
    ),
    "omnibase-infra-lakshman": (
        "the collaborator lane, owned by and mutable only by its owner."
    ),
    "omnibase-infra-dogfood": "the dogfood lane.",
    "omninode-ci-bus": "the fleet CI bus, a broker and not a runtime lane.",
}


class SlotPolicy(NamedTuple):
    """Everything that differs between one pool slot and another.

    Every field here is a value the entrypoint injects into the compose
    overlay. Nothing in ``docker/docker-compose.prepr.yml`` names a slot, so
    the overlay is shared and this table is the only place a slot number
    turns into a port, a prefix or a database suffix.
    """

    slot: int
    compose_project: str
    #: The slot token ``provision_db_slot.sh`` derives its database and role
    #: names from: ``^[a-z][a-z0-9]{0,11}$``, so ``prepr1`` and not ``prepr-1``.
    db_slot: str
    #: The topic prefix applied at the resolver seam and at the gateway API
    #: (OMN-18891). Same token as the database suffix on purpose: one slot
    #: identity read off every isolation axis is one thing to check.
    topic_namespace: str
    #: ``KAFKA_ENVIRONMENT``. ``compute_consumer_group_id`` composes
    #: ``{env}.{service}.{node}.{purpose}.{version}``, so a distinct env token
    #: yields a disjoint consumer group for every node with no code change.
    kafka_environment: str
    #: Valkey logical database index. Shared server, per-slot index.
    valkey_db_index: int
    runtime_main_port: int
    runtime_effects_port: int
    gateway_port: int
    projection_api_port: int


#: The pool. Two slots, per the plan's section 3.2 sizing: memory is not the
#: binding constraint, the ~20-25 minute image build is, and a third slot buys
#: queueing relief only in the phase that is already serialised by the
#: pool-wide build lock.
#:
#: The port blocks are the ones OMN-18890 reserved in
#: ``deploy/lane-census/lane-manifest.yaml``, enumerated free on the lab host
#: on 2026-09-20. Slot 1's runtime pair is the retired prod lane's old block,
#: which is why it is free (OMN-18320).
SLOTS: Final[dict[int, SlotPolicy]] = {
    1: SlotPolicy(
        slot=1,
        compose_project="omnibase-infra-prepr-1",
        db_slot="prepr1",
        topic_namespace="prepr1",
        kafka_environment="prepr1",
        valkey_db_index=1,
        runtime_main_port=28085,
        runtime_effects_port=28086,
        gateway_port=28090,
        projection_api_port=23002,
    ),
    2: SlotPolicy(
        slot=2,
        compose_project="omnibase-infra-prepr-2",
        db_slot="prepr2",
        topic_namespace="prepr2",
        kafka_environment="prepr2",
        valkey_db_index=2,
        runtime_main_port=38085,
        runtime_effects_port=38086,
        gateway_port=38090,
        projection_api_port=33002,
    ),
}

#: The compose project the pool-wide BUILD lock is taken on. It is not a slot
#: and never a compose project anything is deployed to; ``lane_lock.py`` keys
#: its lock file on a string, and this is the string the pool uses so two
#: slots cannot build at once on a host already at load average 7 with 66 GiB
#: swapped out. Kept out of ``SLOTS`` deliberately: a lock name that appeared
#: in the slot table could be passed as a deploy target.
POOL_BUILD_LOCK_PROJECT: Final[str] = "omnibase-infra-prepr-pool-build"

#: The lane words the attribution preflight must require a reason for. Imported
#: by ``scripts/preflight_lane_deploy_attribution.py`` so the two cannot drift.
POOL_LANE_NAMES: Final[frozenset[str]] = frozenset(
    policy.compose_project.removeprefix("omnibase-infra-") for policy in SLOTS.values()
)


class RefusalError(Exception):
    """A policy refusal carrying the exit code the entrypoint must return."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_slot(slot: int) -> SlotPolicy:
    """Return the policy for a slot number, or refuse.

    Refusing an out-of-pool slot number is not pedantry: a slot number picked
    outside this table would derive a compose project the lane manifest does
    not declare, so the census could not see it and nothing would ever report
    it as drift.
    """
    policy = SLOTS.get(slot)
    if policy is None:
        raise RefusalError(
            EXIT_USAGE,
            f"slot {slot} is not in the pool. Declared slots: "
            f"{sorted(SLOTS)}. The pool size is a measured sizing decision "
            f"(plan section 3.2), not a default to widen at a call site.",
        )
    return policy


def assert_target_is_a_pool_slot(compose_project: str) -> SlotPolicy:
    """Refuse any compose project that is not a pool slot.

    This is the function rule 24(e)'s sanction rests on. It is deliberately
    total: a declared lane is refused by name with its own exit code, and
    anything else is refused as unknown. There is no fall-through that builds.
    """
    reason = DECLARED_LANE_PROJECTS.get(compose_project)
    if reason is not None:
        raise RefusalError(
            EXIT_REFUSED_DECLARED_LANE,
            f"'{compose_project}' is {reason}\n"
            f"This entrypoint builds pre-PR verify SLOTS only and accepts no "
            f"lane argument at all. It cannot be pointed at a declared lane, "
            f"which is the entire reason it exists as a separate entrypoint "
            f"rather than a flag on scripts/deploy-runtime.sh.",
        )
    for policy in SLOTS.values():
        if policy.compose_project == compose_project:
            return policy
    raise RefusalError(
        EXIT_REFUSED_UNKNOWN_PROJECT,
        f"'{compose_project}' is neither a declared lane nor a pool slot. "
        f"Refusing rather than creating it: an unrecognised compose project on "
        f"this host is a lane nothing declares, so the census cannot see it and "
        f"nothing will ever report it as drift. Pool slots: "
        f"{sorted(p.compose_project for p in SLOTS.values())}.",
    )


def slot_environment(policy: SlotPolicy) -> dict[str, str]:
    """The env the compose overlay resolves, fail-closed, for this slot.

    ``docker/docker-compose.prepr.yml`` spells every one of these with ``:?``
    so a missing value aborts the compose invocation rather than defaulting a
    slot onto the dev lane's namespace. That is the whole isolation story in
    one property: the overlay cannot run un-namespaced.
    """
    return {
        "ONEX_PREPR_SLOT": str(policy.slot),
        "ONEX_DB_SLOT": policy.db_slot,
        "KAFKA_TOPIC_NAMESPACE": policy.topic_namespace,
        "KAFKA_ENVIRONMENT": policy.kafka_environment,
        "PREPR_VALKEY_DB_INDEX": str(policy.valkey_db_index),
        "PREPR_RUNTIME_MAIN_PORT": str(policy.runtime_main_port),
        "PREPR_RUNTIME_EFFECTS_PORT": str(policy.runtime_effects_port),
        "PREPR_GATEWAY_PORT": str(policy.gateway_port),
        "PREPR_PROJECTION_API_PORT": str(policy.projection_api_port),
    }


def _emit(payload: object) -> int:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepr_slot_policy.py",
        description=(
            "Read-only policy table for the pre-PR verify pool. This module "
            "mutates nothing and opens no connection."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--slot",
        type=int,
        help="print one slot's policy as JSON.",
    )
    mode.add_argument(
        "--slot-env",
        type=int,
        metavar="SLOT",
        help="print one slot's compose environment as KEY=VALUE lines.",
    )
    mode.add_argument(
        "--list-slots",
        action="store_true",
        help="print every declared slot as JSON.",
    )
    mode.add_argument(
        "--assert-pool-slot",
        metavar="COMPOSE_PROJECT",
        help=(
            "exit 0 if the compose project is a pool slot; exit "
            f"{EXIT_REFUSED_DECLARED_LANE} if it is a declared lane, "
            f"{EXIT_REFUSED_UNKNOWN_PROJECT} if it is unknown."
        ),
    )
    mode.add_argument(
        "--print-refusals",
        action="store_true",
        help="print the declared-lane refusal table as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_slots:
            return _emit({str(k): v._asdict() for k, v in sorted(SLOTS.items())})
        if args.print_refusals:
            return _emit(DECLARED_LANE_PROJECTS)
        if args.assert_pool_slot is not None:
            policy = assert_target_is_a_pool_slot(args.assert_pool_slot)
            return _emit(policy._asdict())
        if args.slot_env is not None:
            policy = resolve_slot(args.slot_env)
            for key, value in sorted(slot_environment(policy).items()):
                sys.stdout.write(f"{key}={value}\n")
            return EXIT_OK
        return _emit(resolve_slot(args.slot)._asdict())
    except RefusalError as exc:
        sys.stderr.write(f"[prepr-slot-policy] REFUSED: {exc}\n")
        return exc.code


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
