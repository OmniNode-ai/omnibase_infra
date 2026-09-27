# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run several ``omnibase_infra.validators`` modules in one interpreter (OMN-19613).

Why this exists
---------------
The ``Lint`` job in ``.github/workflows/ci.yml`` is a ``needs:`` of the test
matrix, so every second it spends sits on every PR's critical path. It used to
run fourteen validator modules as fourteen steps, each ``uv run python -m
omnibase_infra.validators.<name> ...``. Each step cost 12 to 15 s on the CI
runners (run 35616533588), including the step that reads one YAML file, so
most of that time was interpreter start-up and importing ``omnibase_infra``
and its dependencies, paid fourteen times. This runner pays it once.

Contract
--------
* The validator list is read from stdin, one per line, as
  ``<title> | <module> <arg> <arg> ...``. Blank lines and lines starting with
  ``#`` are ignored. The list lives in ``ci.yml`` itself, as a heredoc, so the
  workflow still names every module it runs and every argument it passes.
* Only modules under ``omnibase_infra.validators.`` are accepted, and each must
  expose ``main(argv) -> int``. That is the shape all of them share, and their
  ``if __name__ == "__main__": raise SystemExit(main())`` block is exactly what
  ``python -m`` ran before. Calling ``main(args)`` here is the same call.
* EVERY validator runs, even after one fails, so one red validator never hides
  another. The runner exits 1 if any validator failed, and 0 otherwise.
* Exit semantics per validator match ``python -m``: the returned int, or the
  code carried by ``SystemExit`` (``None`` is 0; a non-int payload is printed
  to stderr and counts as 1), and an uncaught exception prints its traceback
  and counts as 1.
* Each failure is reported separately, by title, as its own GitHub error
  annotation, and the per-validator results (exit code and seconds) go to
  stdout and, when ``GITHUB_STEP_SUMMARY`` is set, to the job summary. The
  validators' own output is left untouched inside a collapsible log group.
* A malformed line, an unknown module or a module without ``main`` is itself a
  failure of that entry, reported the same way. It never passes silently.
"""

from __future__ import annotations

import importlib
import os
import shlex
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TextIO

VALIDATOR_PACKAGE = "omnibase_infra.validators."


@dataclass(frozen=True)
class ValidatorSpec:
    """One validator invocation: a human title, a module and its argv."""

    title: str
    module: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ValidatorResult:
    """The outcome of one validator invocation."""

    spec: ValidatorSpec
    exit_code: int
    seconds: float


class SpecError(ValueError):
    """A validator line that cannot be turned into a runnable spec."""


def parse_specs(text: str) -> list[ValidatorSpec]:
    """Parse ``<title> | <module> <args...>`` lines into specs.

    Raises SpecError naming the offending line, so a typo in the workflow
    fails the step instead of silently dropping a validator.
    """
    specs: list[ValidatorSpec] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        title, sep, command = line.partition("|")
        title = title.strip()
        if not sep or not title:
            raise SpecError(
                f"line {number}: expected '<title> | <module> <args>': {raw!r}"
            )
        tokens = shlex.split(command)
        if not tokens:
            raise SpecError(f"line {number}: no module after '|': {raw!r}")
        module, *args = tokens
        if not module.startswith(VALIDATOR_PACKAGE):
            raise SpecError(
                f"line {number}: {module!r} is not under {VALIDATOR_PACKAGE!r}; "
                "this runner only runs omnibase_infra validator modules"
            )
        specs.append(ValidatorSpec(title=title, module=module, args=tuple(args)))
    if not specs:
        raise SpecError("no validators given on stdin")
    return specs


def _exit_code_from_system_exit(exc: SystemExit, stderr: TextIO) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # python -m prints a non-int SystemExit payload to stderr and exits 1.
    print(code, file=stderr)
    return 1


def run_one(
    spec: ValidatorSpec,
    *,
    importer: Callable[[str], object] | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one validator's ``main(args)`` and return its exit code."""
    err = stderr if stderr is not None else sys.stderr
    load = importer if importer is not None else importlib.import_module
    try:
        module = load(spec.module)
    except Exception:  # noqa: BLE001 - python -m reports any import failure as exit 1
        traceback.print_exc(file=err)
        return 1
    main = getattr(module, "main", None)
    if not callable(main):
        print(f"{spec.module} has no callable main(argv)", file=err)
        return 1
    saved_argv = sys.argv
    sys.argv = [spec.module, *spec.args]
    try:
        result = main(list(spec.args))
    except SystemExit as exc:
        return _exit_code_from_system_exit(exc, err)
    except Exception:  # noqa: BLE001 - an uncaught error in main is exit 1 under python -m
        traceback.print_exc(file=err)
        return 1
    finally:
        sys.argv = saved_argv
    if result is None:
        return 0
    if isinstance(result, bool) or not isinstance(result, int):
        print(f"{spec.module}.main returned {result!r}, not an int exit code", file=err)
        return 1
    return int(result)


def _escape_annotation(text: str) -> str:
    # GitHub workflow-command escaping for the message and property values.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_annotation(text).replace(":", "%3A").replace(",", "%2C")


def run_all(
    specs: Sequence[ValidatorSpec],
    *,
    importer: Callable[[str], object] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[ValidatorResult]:
    """Run every spec in order, never stopping early, and return the results."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    results: list[ValidatorResult] = []
    for spec in specs:
        print(f"::group::{spec.title}", file=out)
        print(f"$ python -m {spec.module} {shlex.join(spec.args)}".rstrip(), file=out)
        out.flush()
        started = clock()
        code = run_one(spec, importer=importer, stderr=err)
        seconds = clock() - started
        out.flush()
        err.flush()
        print("::endgroup::", file=out)
        results.append(ValidatorResult(spec=spec, exit_code=code, seconds=seconds))
        if code != 0:
            print(
                f"::error title={_escape_property(spec.title)}::"
                + _escape_annotation(
                    f"{spec.module} exited with code {code} "
                    f"(args: {shlex.join(spec.args) or 'none'})"
                ),
                file=out,
            )
    return results


def summarize(results: Sequence[ValidatorResult]) -> str:
    """A Markdown table of every result, failures first."""
    ordered = sorted(results, key=lambda r: (r.exit_code == 0, r.spec.title))
    lines = [
        "| Validator | Module | Exit | Seconds |",
        "| --- | --- | --- | --- |",
    ]
    for r in ordered:
        verdict = "0" if r.exit_code == 0 else f"**{r.exit_code}**"
        lines.append(
            f"| {r.spec.title} | `{r.spec.module}` | {verdict} | {r.seconds:.1f} |"
        )
    failed = sum(1 for r in results if r.exit_code != 0)
    total = sum(r.seconds for r in results)
    lines.append("")
    lines.append(
        f"{len(results)} validators, {failed} failed, {total:.1f} s in one process."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print(
            "usage: run_validators_in_process.py < '<title> | <module> <args>' lines",
            file=sys.stderr,
        )
        return 2
    try:
        specs = parse_specs(sys.stdin.read())
    except SpecError as exc:
        print(f"::error title=Validator list::{_escape_annotation(str(exc))}")
        return 1
    results = run_all(specs)
    table = summarize(results)
    print(table)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("### omnibase_infra validators (one process)\n\n")
            handle.write(table + "\n")
    return 1 if any(r.exit_code != 0 for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
