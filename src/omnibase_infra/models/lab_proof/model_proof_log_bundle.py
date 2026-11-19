# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The build logs of one pool proof, kept beside its receipt (OMN-19998).

A pool proof builds on a lab host in a work directory that its own teardown
removes. Until now the proof printed the last four lines of ``build.log`` and
the rest went with the directory, so a build that failed (omnibase_infra#5702
on the pre-PR slot with ``build rc=10``, omnibase_infra#5994 on the isolated
member after three minutes) left no cause to read.

The driver reads the logs back over its own transport before the teardown
phase runs, as one gzip archive, and this model keeps them: a directory next
to the receipts, one file per log, named in the receipt's ``stack_built``
evidence and in the proof output together with the failing build step.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
import tarfile
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

#: the archive the driver stores under this output key before it tears down
ARCHIVE_KEY = "proof-logs-archive"
#: each log is cut to its last bytes before it is archived, so a runaway build
#: cannot make the read-back larger than a transport can carry
TAIL_BYTES = 8 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.log$")
_STEP_HEADER = re.compile(r"^#(\d+) (\[.*\].*)$")
_STEP_ERROR = re.compile(r"^#(\d+) ERROR: (.*)$")
_FAIL_LINE = re.compile(r"^\s*(?:FAIL|ERROR|fatal)\b.*$", re.IGNORECASE)
_LANE_LINE = re.compile(r"^\[prepr-verify-lane\] REFUSED/FAILED.*$")
_STEP_LIMIT = 240
#: the build-output lines the receipt's ``stack_built`` evidence reads
STACK_BUILT_LINES = (
    r"^(?:build|up) rc=\d+.*$|^slot-migration FAILED$"
    r"|^proof-logs kept=.*$|^build-failing-step .*$"
)
_BEGIN = "PROOF-LOGS-BEGIN"
_END = "PROOF-LOGS-END"


class ModelProofLogBundle(BaseModel):
    """The ``*.log`` files of one proof work directory, by file name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def fetch_command(work: str) -> str:
        """The shell command that prints the work directory's logs as base64 gzip.

        The archive sits between two marker lines, because the transport returns
        stdout and stderr together and a warning must not corrupt it. Prints
        nothing and exits non-zero when the directory is gone. ``work`` is the
        driver's own ``$HOME/...`` path, never user input.
        """
        tail = f'tail -c {TAIL_BYTES} "$f" > "$d/$f"'
        return (
            f"cd {work} && d=$(mktemp -d) && "
            f'{{ for f in *.log; do [ -f "$f" ] && {tail}; done; '
            f"echo {_BEGIN}; "
            f"tar -czf - -C \"$d\" . | base64 | tr -d '\\n'; "
            f"rc=$?; echo; echo {_END}; "
            f'rm -rf "$d"; exit $rc; }}'
        )

    @classmethod
    def from_archive(cls, output: str) -> ModelProofLogBundle:
        """Read the command's output back; an unreadable archive reads as no logs."""
        match = re.search(rf"(?ms)^{_BEGIN}\n(.*?)\n?^{_END}$", output)
        if match is None:
            return cls()
        try:
            raw = base64.b64decode("".join(match.group(1).split()), validate=True)
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                found: dict[str, str] = {}
                for member in tar.getmembers():
                    name = Path(member.name).name
                    if not member.isfile() or not _SAFE_NAME.match(name):
                        continue
                    handle = tar.extractfile(member)
                    if handle is not None:
                        found[name] = handle.read().decode("utf-8", "replace")
        except (binascii.Error, tarfile.TarError, OSError, EOFError):
            return cls()
        return cls(files=tuple(sorted(found.items())))

    def failing_step(self) -> str:
        """The step of ``build.log`` that failed, or the line that says why.

        BuildKit's plain progress names a failed step as ``#N ERROR: <why>``
        and its command on the ``#N [stage i/j] RUN ...`` line; the image audit
        and the slot entrypoint each print a line that starts FAIL or
        ``[prepr-verify-lane]``. The first form found wins, in that order.
        """
        text = dict(self.files).get("build.log", "")
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        for line in lines:
            error = _STEP_ERROR.match(line)
            if error:
                header = next(
                    (
                        m.group(2)
                        for m in map(_STEP_HEADER.match, lines)
                        if m and m.group(1) == error.group(1)
                    ),
                    "",
                )
                step = f"step #{error.group(1)} {header} ERROR: {error.group(2)}"
                return " ".join(step.split())[:_STEP_LIMIT]
        for pattern in (_FAIL_LINE, _LANE_LINE):
            hit = next((line for line in lines if pattern.match(line)), "")
            if hit:
                return " ".join(hit.split())[:_STEP_LIMIT]
        return " ".join(lines[-1].split())[:_STEP_LIMIT] if lines else ""

    def write(self, dest: Path) -> Path:
        """Write each log under ``dest`` (owner-only) and return the directory."""
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, text in self.files:
            path = dest / name
            path.write_text(text, encoding="utf-8")
            path.chmod(0o600)
        return dest

    @classmethod
    def keep(cls, outputs: Mapping[str, str], dest: Path) -> tuple[str, str]:
        """Keep the logs read before teardown; return (proof text, build output).

        The build output gains up to two lines after its own, which the receipt's
        ``stack_built`` evidence reads: where the logs were kept, and the failing
        step when ``build rc`` is not 0. The proof text names both for a reader.
        """
        build = outputs.get("build", "")
        bundle = cls.from_archive(outputs.get(ARCHIVE_KEY, ""))
        if not bundle.files:
            note = "proof-logs not kept: the work directory held no readable log"
            return f"  {note}", f"{note}\n{build}"
        where = bundle.write(dest)
        names = ",".join(name for name, _ in bundle.files)
        lines = [f"proof-logs kept={where} files={names}"]
        if not re.search(r"build rc=0\b", build):
            step = bundle.failing_step() or "no failing step found in build.log"
            lines.append(f"build-failing-step {step}")
        return "\n".join(f"  {line}" for line in lines), "\n".join([build, *lines])
