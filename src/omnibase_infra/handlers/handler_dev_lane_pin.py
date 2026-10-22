# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read or set one dev-lane image pin in the operator env store, with a ledger row (OMN-18475).

WHY. The compose dev lane takes two images from the operator env store instead
of building them: ``onex-api`` from ``ONEX_API_IMAGE`` and the cloud corpus
migrations from ``ONEX_CLOUD_MIGRATE_IMAGE``. ``ONEX_API_IMAGE`` had a governed
writer (``scripts/runtime_build/repoint_dev_lane_onex_api.py``, which resolves the
image from the lane's own builds); ``ONEX_CLOUD_MIGRATE_IMAGE`` had none. On
2026-10-21 at 22:48Z a lane changed it by editing the store with a one-off
script, against the operator's ask that agents get deterministic tools.

WHAT. ``read`` returns a pin and whether its image is resident. ``set`` changes
exactly one line, keeps a backup, reads the file back, appends a STATUS row to
the work ledger, and restores the backup if the row is refused, so no pin
changes without a row naming it. The dev deploy agent reads the store again at
its next compose command (``deploy_agent.executor.operator_store_overlay``), so
no restart follows.

REFUSES PROD. The lane must be ``dev``: a prod image is promoted through the
gated node path with a fresh approval (OMN-13418) and is never written into a
file by a command. Only the keys in :class:`EnumDevLanePinKey` can be set, and
``test_handler_dev_lane_pin_omn18475`` fails if any of them is rendered by a
compose file of a governed lane, which is what keeps the store, shared with the
prod agent unit, from carrying a dev pin into a prod container. A ``PROD_*`` key
is refused by name.

NO DEFAULTS (rule 8). The store, the ledger and the ledger writer have none, so
the handler never writes a file or a ledger the caller did not name. Values are
image references, never secrets, and a value that is not one is refused.

Command line::

    python -m omnibase_infra.handlers.handler_dev_lane_pin read \\
        --env-file <store> --key ONEX_CLOUD_MIGRATE_IMAGE
    python -m omnibase_infra.handlers.handler_dev_lane_pin set \\
        --env-file <store> --key ONEX_CLOUD_MIGRATE_IMAGE --value <image:tag> \\
        --lane dev --ledger <ledger> --ledger-command '<onex-ledger argv>' \\
        --ledger-lane <lane> --ticket OMN-<n> [--execute]
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_dev_lane_pin_key import EnumDevLanePinKey
from omnibase_infra.enums.enum_dev_lane_pin_outcome import EnumDevLanePinOutcome
from omnibase_infra.models.model_dev_lane_pin_read_request import (
    ModelDevLanePinReadRequest,
)
from omnibase_infra.models.model_dev_lane_pin_result import ModelDevLanePinResult
from omnibase_infra.models.model_dev_lane_pin_set_request import (
    ModelDevLanePinSetRequest,
)

#: The one lane a pin is set on.
DEV_LANE = "dev"

EXIT_OK = 0
EXIT_REFUSED = 3

#: An image reference: a name, then a tag, a digest or both.
_IMAGE_REFERENCE = re.compile(
    r"^(?P<name>[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*)"
    r"(?::(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}))?"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}))?$"
)
_TICKET = re.compile(r"^OMN-\d+$")
_LANE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LEDGER_TIMEOUT_SECONDS = 330
_DOCKER_TIMEOUT_SECONDS = 60
_ACTOR = "script:dev-lane-pin"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class DevLanePinRefusedError(Exception):
    """A condition under which nothing is written; the message names it."""


class HandlerDevLanePin:
    """Fail closed: every unproven condition is a refusal, never a write."""

    def __init__(
        self,
        run: Runner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._run: Runner = run or subprocess.run
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def read(self, request: ModelDevLanePinReadRequest) -> ModelDevLanePinResult:
        """The pin's value and whether its image is resident on this daemon."""
        env_file = Path(request.env_file)
        try:
            key = _parse_key(request.key)
            _, _, value = _read_pin(env_file, key)
        except DevLanePinRefusedError as refusal:
            return _refused(request.key, env_file, refusal)
        try:
            image_id: str | None = self._inspect(value, docker=request.docker_cmd)
        except DevLanePinRefusedError:
            image_id = None
        return ModelDevLanePinResult(
            outcome=EnumDevLanePinOutcome.READ,
            key=key.value,
            env_file=str(env_file),
            value=value,
            resident=image_id is not None,
            image_id=image_id,
        )

    def set(self, request: ModelDevLanePinSetRequest) -> ModelDevLanePinResult:
        """Change one pin and append its ledger row, or refuse and write nothing."""
        env_file = Path(request.env_file)
        try:
            return self._set(request, env_file)
        except DevLanePinRefusedError as refusal:
            return _refused(request.key, env_file, refusal)

    def _set(
        self, request: ModelDevLanePinSetRequest, env_file: Path
    ) -> ModelDevLanePinResult:
        if request.lane != DEV_LANE:
            raise DevLanePinRefusedError(
                f"lane {request.lane!r} refused: this handler sets dev-lane pins "
                "only. A prod image moves through the gated promotion path "
                "(OMN-13418), never through a file edit; nothing was written."
            )
        key = _parse_key(request.key)
        _check_reference(request.value)
        if not _TICKET.match(request.ticket):
            raise DevLanePinRefusedError(
                f"ticket {request.ticket!r} is not OMN-<n>; nothing was written."
            )
        if not _LANE_TOKEN.match(request.ledger_lane):
            raise DevLanePinRefusedError(
                f"ledger lane {request.ledger_lane!r} is not one token; nothing was "
                "written."
            )
        if not request.ledger_command:
            raise DevLanePinRefusedError(
                "the ledger command is empty; nothing was written."
            )
        image_id = self._inspect(request.value, docker=request.docker_cmd)
        lines, index, before = _read_pin(env_file, key)
        common = {
            "key": key.value,
            "env_file": str(env_file),
            "pin_before": before,
            "resident": True,
            "image_id": image_id,
        }
        if before == request.value:
            return ModelDevLanePinResult(
                outcome=EnumDevLanePinOutcome.UNCHANGED,
                value=before,
                pin_after=before,
                **common,
            )
        if not request.execute:
            return ModelDevLanePinResult(
                outcome=EnumDevLanePinOutcome.PLANNED,
                value=request.value,
                pin_after=before,
                reason="execute not requested; nothing was written",
                **common,
            )
        now = self._clock()
        backup = _write_pin(
            env_file,
            lines=lines,
            index=index,
            key=key,
            value=request.value,
            stamp=now.strftime("%Y%m%dT%H%M%SZ"),
        )
        row = _ledger_row(
            stamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            request=request,
            key=key,
            before=before,
            env_file=env_file,
            backup=backup,
            image_id=image_id,
        )
        try:
            self._append(request, row)
        except DevLanePinRefusedError as refusal:
            _replace(env_file, backup.read_text(encoding="utf-8"))
            raise DevLanePinRefusedError(
                f"{refusal}; a pin does not change without its ledger row, so "
                f"{env_file} was restored from {backup}"
            ) from refusal
        return ModelDevLanePinResult(
            outcome=EnumDevLanePinOutcome.WRITTEN,
            value=request.value,
            pin_after=request.value,
            backup_path=str(backup),
            ledger_row=row,
            **common,
        )

    def _inspect(self, reference: str, *, docker: str) -> str:
        """The image id of a resident reference, or a refusal.

        A pin naming an image that is not resident renders a compose service
        that cannot start, because the dev lane's one-shots run with
        ``--pull never``.
        """
        argv = [docker, "image", "inspect", "--format", "{{.Id}}", reference]
        try:
            completed = self._run(
                argv,
                capture_output=True,
                text=True,
                timeout=_DOCKER_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DevLanePinRefusedError(
                f"`{docker} image inspect {reference}` did not run "
                f"({type(exc).__name__}); residency is unproven, so nothing was "
                "written."
            ) from exc
        if completed.returncode != 0:
            raise DevLanePinRefusedError(
                f"{reference} is not resident on this host (`{docker} image inspect` "
                f"exited {completed.returncode}: "
                f"{(completed.stderr or '').strip() or '<no stderr>'}). A pin naming "
                "it would render a service that cannot start; nothing was written."
            )
        return str(completed.stdout).strip()

    def _append(self, request: ModelDevLanePinSetRequest, row: str) -> None:
        argv = [*request.ledger_command, request.ledger_path, "--append", row]
        try:
            completed = self._run(
                argv,
                capture_output=True,
                text=True,
                timeout=_LEDGER_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DevLanePinRefusedError(
                f"the ledger append did not run ({type(exc).__name__})"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:400]
            raise DevLanePinRefusedError(
                f"the ledger append exited {completed.returncode}: {detail}"
            )


def _refused(
    key: str, env_file: Path, refusal: DevLanePinRefusedError
) -> ModelDevLanePinResult:
    return ModelDevLanePinResult(
        outcome=EnumDevLanePinOutcome.REFUSED,
        key=key,
        env_file=str(env_file),
        reason=str(refusal),
    )


def _parse_key(raw: str) -> EnumDevLanePinKey:
    if raw.startswith("PROD_"):
        raise DevLanePinRefusedError(
            f"{raw} is a prod image key. A prod image is promoted through the gated "
            "node path with a fresh approval (OMN-13418), never written by this "
            "handler; nothing was written."
        )
    try:
        return EnumDevLanePinKey(raw)
    except ValueError:
        allowed = ", ".join(key.value for key in EnumDevLanePinKey)
        raise DevLanePinRefusedError(
            f"{raw} is not a dev-lane pin this handler sets (allowed: {allowed}); "
            "nothing was written."
        ) from None


def _check_reference(value: str) -> None:
    match = _IMAGE_REFERENCE.match(value)
    if match is None:
        raise DevLanePinRefusedError(
            f"{value!r} is not an image reference; a pin is an image name with a "
            "tag or digest, never a secret or free text. Nothing was written."
        )
    if not match.group("digest") and match.group("tag") in (None, "latest"):
        raise DevLanePinRefusedError(
            f"{value} names no fixed tag; a pin names a tag other than latest, or a "
            "digest. Nothing was written."
        )


def _read_pin(env_file: Path, key: EnumDevLanePinKey) -> tuple[list[str], int, str]:
    """The store's lines, the index of the key's one line, and its value.

    A missing key and a duplicated key are both refusals: a sourcing shell takes
    the LAST assignment, so rewriting one of two lines would change the file and
    not the lane.
    """
    if not env_file.is_file():
        raise DevLanePinRefusedError(
            f"env file {env_file} does not exist; nothing was written."
        )
    lines = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    prefixes = (f"{key.value}=", f"export {key.value}=")
    hits = [i for i, line in enumerate(lines) if line.startswith(prefixes)]
    if not hits:
        raise DevLanePinRefusedError(
            f"{env_file} carries no {key.value}= line. This handler changes an "
            "existing pin; it does not decide that a lane should have one. Nothing "
            "was written."
        )
    if len(hits) > 1:
        numbers = ", ".join(str(i + 1) for i in hits)
        raise DevLanePinRefusedError(
            f"{env_file} carries {len(hits)} {key.value}= lines (lines {numbers}); a "
            "sourcing shell takes the last, so rewriting one would change the file "
            "and not the lane. Nothing was written."
        )
    index = hits[0]
    raw = lines[index].split("=", 1)[1].strip()
    try:
        tokens = shlex.split(raw) if raw else [""]
    except ValueError:
        tokens = [raw]
    return lines, index, tokens[0] if len(tokens) == 1 else raw


def _write_pin(
    env_file: Path,
    *,
    lines: list[str],
    index: int,
    key: EnumDevLanePinKey,
    value: str,
    stamp: str,
) -> Path:
    """Replace exactly one line atomically, keeping a backup, and prove it.

    The file is replaced by a rename, so the deploy agent's re-read never sees a
    half-written store, and it keeps its mode. The readback asserts the line
    count is unchanged, exactly the key's line differs, and that line now holds
    the value; otherwise the backup is restored and the write refused.
    """
    backup = env_file.with_name(f"{env_file.name}.bak.omn18475-{stamp}")
    shutil.copy2(env_file, backup)
    updated = list(lines)
    trailing = "\n" if lines[index].endswith("\n") else ""
    prefix = "export " if lines[index].startswith("export ") else ""
    updated[index] = f"{prefix}{key.value}={value}{trailing}"
    _replace(env_file, "".join(updated))

    after = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    differing = [
        i
        for i in range(max(len(before), len(after)))
        if before[i : i + 1] != after[i : i + 1]
    ]
    try:
        _, _, readback = _read_pin(env_file, key)
    except DevLanePinRefusedError:
        readback = None
    if len(after) != len(before) or differing != [index] or readback != value:
        _replace(env_file, "".join(before))
        raise DevLanePinRefusedError(
            f"readback refused the write to {env_file}: {len(before)} line(s) "
            f"before, {len(after)} after, {len(differing)} differ, read back "
            f"{readback!r}. The file was restored from {backup}."
        )
    return backup


def _replace(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp = Path(handle.name)
        handle.write(text)
    try:
        temp.chmod(mode)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _ledger_row(
    *,
    stamp: str,
    request: ModelDevLanePinSetRequest,
    key: EnumDevLanePinKey,
    before: str,
    env_file: Path,
    backup: Path,
    image_id: str,
) -> str:
    """The STATUS row a write appends. Every cell is a value this handler checked."""
    return (
        f"{stamp} | STATUS | lane={request.ledger_lane} | ticket={request.ticket} | "
        f"actor={_ACTOR} | model=none | action=dev-pin-set | key={key.value} | "
        f"pin-before={before} | pin-after={request.value} | dev lane pin "
        f"{key.value} set from {before} to {request.value} in the operator env "
        f"store {env_file.name} (backup {backup.name}, image {image_id[:19]} "
        "resident); the dev deploy agent reads the store again at its next "
        "compose command, no restart (OMN-18475)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnibase_infra.handlers.handler_dev_lane_pin",
        description="Read or set one dev-lane image pin in the operator env store.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("read", "set"):
        sub = commands.add_parser(name)
        sub.add_argument("--env-file", required=True, help="operator env store")
        sub.add_argument("--key", required=True, help="the pin key")
        sub.add_argument("--docker", default="docker", help="docker executable")
    write = commands.choices["set"]
    write.add_argument("--value", required=True, help="the image reference to pin")
    write.add_argument("--lane", required=True, help="must be dev")
    write.add_argument("--ledger", required=True, help="the rolling work ledger")
    write.add_argument(
        "--ledger-command",
        required=True,
        help="argv prefix of the ledger writer, shell-quoted",
    )
    write.add_argument("--ledger-lane", required=True, help="the row's lane")
    write.add_argument("--ticket", required=True, help="the row's ticket, OMN-<n>")
    write.add_argument("--execute", action="store_true", help="write the pin")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = HandlerDevLanePin()
    if args.command == "read":
        result = handler.read(
            ModelDevLanePinReadRequest(
                env_file=args.env_file, key=args.key, docker_cmd=args.docker
            )
        )
    else:
        result = handler.set(
            ModelDevLanePinSetRequest(
                env_file=args.env_file,
                key=args.key,
                value=args.value,
                lane=args.lane,
                ledger_path=args.ledger,
                ledger_command=tuple(shlex.split(args.ledger_command)),
                ledger_lane=args.ledger_lane,
                ticket=args.ticket,
                execute=args.execute,
                docker_cmd=args.docker,
            )
        )
    sys.stdout.write(result.model_dump_json(indent=2, exclude_none=True) + "\n")
    return EXIT_REFUSED if result.outcome is EnumDevLanePinOutcome.REFUSED else EXIT_OK


__all__ = ["DevLanePinRefusedError", "HandlerDevLanePin", "main"]


if __name__ == "__main__":
    sys.exit(main())
