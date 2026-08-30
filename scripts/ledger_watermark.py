#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve a reader's position in an append-only ledger section by ROW IDENTITY.

Why this exists (OMN-17023). `morning-friction-sweep` stored "how far have I
read" as a LINE NUMBER into a ledger that is not time-ordered and that gets
rewritten whenever its oldest rows are rolled into an archive. On 2026-08-27 a
split moved 19,744 lines out; the stored mark (21,618) then addressed entirely
different content, and the sweep skipped every row between silently. A line
number cannot detect that the content at that line changed meaning -- there is
nothing to compare it against. So the failure is not "the mark drifted", it is
"the mark cannot know it drifted", and no amount of care from the caller fixes
that.

The mark this tool keeps is the identity of the last row read: its heading line
plus a 12-hex digest over its normalized body. Both halves are load-bearing.
The heading alone repeats (lanes reuse titles); the digest alone cannot be
located in an archive by eye. Together they survive a roll -- the row is found
in the archive file it was moved to, and everything now live is correctly
unread -- and they fail CLOSED when the row was edited or is gone, because a
reader that cannot prove where it stopped must not guess.

Exit codes:
  0  resolved (JSON result on stdout)
  2  usage error
  3  UNRESOLVED -- the anchor row is not on disk anywhere, or its digest no
     longer matches. Nothing is advanced; the caller must re-anchor
     deliberately rather than resume from an unknown position.
  4  SCHEMA -- the state file's watermark schema is not the one this tool
     speaks (including the pre-OMN-17023 line-number schema, which is
     refused rather than reinterpreted; run --migrate to convert it).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

WATERMARK_SCHEMA_VERSION = 2
WATERMARK_RESULT_SCHEMA = "ledger-watermark/2"
SCHEMA_VERSION_KEY = "watermark_schema_version"

EXIT_UNRESOLVED = 3
EXIT_SCHEMA = 4

_HERE = Path(__file__).resolve().parent


def _load_ledger_lock() -> Any:
    """Load the section parser from its single implementation.

    Section/row parsing lives in ledger_lock.py because that is the tool that
    WRITES the section; a second copy here would be free to drift from the
    writer's idea of where a row starts, which is precisely the class of bug
    this ticket is closing.
    """
    script = _HERE / "ledger_lock.py"
    spec = importlib.util.spec_from_file_location("ledger_lock", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK = _load_ledger_lock()


class UnresolvedAnchorError(RuntimeError):
    """The anchor row cannot be located; the caller must not advance."""


class SchemaMismatchError(RuntimeError):
    """The state file does not speak this tool's watermark schema."""


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SchemaMismatchError(
            f"state file not found: {path}. Create it with "
            f"{{'{SCHEMA_VERSION_KEY}': {WATERMARK_SCHEMA_VERSION}, 'watermarks': {{...}}}}"
        )
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SchemaMismatchError(f"state file {path} is not a JSON object")
    return parsed


def require_current_schema(state: dict[str, Any]) -> None:
    version = state.get(SCHEMA_VERSION_KEY)
    if version == WATERMARK_SCHEMA_VERSION:
        return
    if version is None:
        raise SchemaMismatchError(
            f"state file carries no {SCHEMA_VERSION_KEY}: this is the pre-OMN-17023 "
            "line-number watermark, which cannot be reinterpreted as a row anchor "
            "without silently skipping rows. Convert it once with --migrate."
        )
    raise SchemaMismatchError(
        f"state file declares {SCHEMA_VERSION_KEY}={version}, this tool speaks "
        f"{WATERMARK_SCHEMA_VERSION}. Refusing to read a schema it does not know."
    )


def source_entry(state: dict[str, Any], source: str) -> dict[str, Any]:
    watermarks = state.get("watermarks")
    if not isinstance(watermarks, dict) or source not in watermarks:
        raise SchemaMismatchError(f"state file has no watermark for source {source!r}")
    entry = watermarks[source]
    if not isinstance(entry, dict):
        raise SchemaMismatchError(f"watermark for {source!r} is not an object")
    return entry


def require_section_heading(entry: dict[str, Any], source: str) -> str:
    heading = entry.get("section_heading")
    if not isinstance(heading, str) or not heading.strip():
        raise SchemaMismatchError(
            f"watermark {source!r} declares no section_heading -- the capped, append-only "
            "section a row anchor is relative to must be named explicitly"
        )
    return heading


def archive_files(entry: dict[str, Any]) -> list[Path]:
    raw = entry.get("archive_dir")
    if not raw:
        return []
    directory = Path(raw)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.md") if p.is_file())


def find_in_archive(
    files: list[Path], anchor_heading: str, anchor_digest: str | None
) -> tuple[str, list[Any]] | None:
    """Where the anchor row was archived, and the rows archived AFTER it.

    Returns `(archive path, rows still unread in the archives)` or None.

    The second half is the part that is easy to get wrong and expensive to get
    wrong. A roll moves the OLDEST rows out and does not ask the reader whether
    it had read them yet -- so a reader whose anchor lands in an archive can
    have unread rows on BOTH sides of the split. Reporting only the live file's
    rows would silently drop the archived ones: the same class of silent skip
    the line-number watermark produced, reached by a different route. Rows are
    ordered by roll order within a file and by filename (the roll stamps the
    date) across files.

    Matching is by the same (heading, digest) identity the live search uses; a
    heading found in an archive whose digest disagrees is NOT a match, because
    that is a different row that happens to share a title.
    """
    target = anchor_heading.strip()
    for position, path in enumerate(files):
        text = path.read_text(encoding="utf-8")
        if target not in text:
            continue
        archived = _archive_entries(text)
        index = None
        for offset, candidate in enumerate(archived):
            if candidate.heading.strip() != target:
                continue
            if anchor_digest is None or candidate.digest() == anchor_digest:
                index = offset
        if index is None:
            continue
        unread: list[Any] = list(archived[index + 1 :])
        for later in files[position + 1 :]:
            unread.extend(_archive_entries(later.read_text(encoding="utf-8")))
        return str(path), unread
    return None


def _archive_entries(text: str) -> list[Any]:
    lines = text.splitlines(keepends=True)
    starts = [
        i for i, line in enumerate(lines) if LOCK.ENTRY_HEADING_PATTERN.match(line)
    ]
    entries = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        entries.append(
            LOCK.SectionEntry(
                heading=lines[start].rstrip("\n"),
                text="".join(lines[start:end]),
                start_line=start + 1,
                end_line=end,
            )
        )
    return entries


def resolve(ledger: Path, entry: dict[str, Any], source: str) -> dict[str, Any]:
    heading = require_section_heading(entry, source)
    parsed = LOCK.parse_section_file(ledger, heading)
    rows = parsed.entries
    anchor_heading = entry.get("anchor_heading")
    anchor_digest = entry.get("anchor_digest")
    line_mark = entry.get("line_count_at_advance")

    if anchor_heading is None:
        unread = rows
        found_in = "bootstrap"
    else:
        matches = [row for row in rows if row.heading.strip() == anchor_heading.strip()]
        exact = [
            row for row in matches if anchor_digest and row.digest() == anchor_digest
        ]
        if anchor_digest and matches and not exact:
            raise UnresolvedAnchorError(
                f"anchor row {anchor_heading!r} is present in {ledger} but its digest "
                f"changed ({anchor_digest} -> {[m.digest() for m in matches]}): the row was "
                "rewritten after it was read. Re-anchor deliberately; resuming from a "
                "rewritten row would skip or replay rows silently."
            )
        chosen = exact[-1] if exact else (matches[-1] if matches else None)
        if chosen is not None:
            found_in = "live"
            index = rows.index(chosen)
            unread = rows[index + 1 :]
        else:
            located = find_in_archive(
                archive_files(entry), anchor_heading, anchor_digest
            )
            if located is None:
                raise UnresolvedAnchorError(
                    f"anchor row {anchor_heading!r} is in neither {ledger} nor any archive under "
                    f"{entry.get('archive_dir')!r}. The reader cannot prove where it stopped."
                )
            archive_path, archived_unread = located
            found_in = f"archive:{archive_path}"
            unread = archived_unread + rows

    live_unread = [row for row in unread if any(row is candidate for candidate in rows)]
    archived_unread_rows = [
        row for row in unread if not any(row is candidate for candidate in rows)
    ]
    if live_unread:
        # resume_line addresses the LIVE file only. Rows that were archived
        # before the reader got to them cannot be addressed by a line number
        # into this file at all, so they are named separately rather than
        # folded into one and lost.
        resume_line = live_unread[0].start_line
    else:
        resume_line = len(ledger.read_text(encoding="utf-8").splitlines()) + 1

    skipped = None
    if isinstance(line_mark, int):
        skipped = len(archived_unread_rows) + sum(
            1 for row in live_unread if row.start_line <= line_mark
        )

    tail = rows[-1] if rows else None
    return {
        "schema": WATERMARK_RESULT_SCHEMA,
        "source": source,
        "ledger": str(ledger),
        "section_heading": heading.strip(),
        "anchor_found_in": found_in,
        "anchor_heading": anchor_heading,
        "resume_line": resume_line,
        "unread_entries": len(unread),
        "unread_lines": sum(len(row.text.splitlines()) for row in unread),
        "unread_headings": [row.heading for row in unread],
        "unread_archived_entries": len(archived_unread_rows),
        "unread_archived_headings": [row.heading for row in archived_unread_rows],
        "skipped_by_line_watermark": skipped,
        "tail_heading": tail.heading if tail else None,
        "tail_digest": tail.digest() if tail else None,
        "line_count": len(ledger.read_text(encoding="utf-8").splitlines()),
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def advance(
    ledger: Path, state_path: Path, state: dict[str, Any], source: str
) -> dict[str, Any]:
    entry = source_entry(state, source)
    result = resolve(ledger, entry, source)
    entry["anchor_heading"] = result["tail_heading"]
    entry["anchor_digest"] = result["tail_digest"]
    entry["line_count_at_advance"] = result["line_count"]
    entry["path"] = str(ledger)
    state[SCHEMA_VERSION_KEY] = WATERMARK_SCHEMA_VERSION
    write_state(state_path, state)
    result["advanced"] = True
    return result


def migrate(
    ledger: Path, state_path: Path, state: dict[str, Any], source: str
) -> dict[str, Any]:
    version = state.get(SCHEMA_VERSION_KEY)
    if version == WATERMARK_SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"state file already declares {SCHEMA_VERSION_KEY}={WATERMARK_SCHEMA_VERSION}; "
            "there is no line-number watermark left to migrate"
        )
    if version is not None:
        raise SchemaMismatchError(
            f"state file declares {SCHEMA_VERSION_KEY}={version}, which this tool cannot migrate"
        )
    entry = source_entry(state, source)
    heading = require_section_heading(entry, source)
    line_mark = entry.get("lines")
    if not isinstance(line_mark, int):
        raise SchemaMismatchError(
            f"watermark {source!r} has no integer 'lines' field to migrate from"
        )
    rows = LOCK.parse_section_file(ledger, heading).entries
    anchor = None
    for row in rows:
        if row.start_line <= line_mark:
            anchor = row
        else:
            break
    entry.pop("lines", None)
    entry["anchor_heading"] = anchor.heading if anchor else None
    entry["anchor_digest"] = anchor.digest() if anchor else None
    entry["line_count_at_advance"] = line_mark
    entry["path"] = str(ledger)
    state[SCHEMA_VERSION_KEY] = WATERMARK_SCHEMA_VERSION
    state.pop("version", None)
    write_state(state_path, state)
    return {
        "schema": WATERMARK_RESULT_SCHEMA,
        "source": source,
        "migrated_from_line": line_mark,
        "anchor_heading": entry["anchor_heading"],
        "anchor_digest": entry["anchor_digest"],
        "state": str(state_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a reader's position in an append-only ledger section by row identity "
            "(heading + body digest) instead of by line number."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ledger", type=Path, help="the ledger file being read")
    parser.add_argument(
        "--state", type=Path, required=True, help="the reader's state JSON"
    )
    parser.add_argument(
        "--source", required=True, help="which watermark inside the state file"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--resolve", action="store_true", help="report the unread rows; write nothing"
    )
    action.add_argument(
        "--advance",
        action="store_true",
        help="resolve, then move the anchor to the section's current last row",
    )
    action.add_argument(
        "--migrate",
        action="store_true",
        help="convert a pre-OMN-17023 line-number watermark into a row anchor, once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = read_state(args.state)
        if args.migrate:
            result = migrate(args.ledger, args.state, state, args.source)
        else:
            require_current_schema(state)
            if args.advance:
                result = advance(args.ledger, args.state, state, args.source)
            else:
                result = resolve(
                    args.ledger, source_entry(state, args.source), args.source
                )
    except SchemaMismatchError as exc:
        print(f"ledger_watermark: SCHEMA -- {exc}", file=sys.stderr)
        return EXIT_SCHEMA
    except UnresolvedAnchorError as exc:
        print(f"ledger_watermark: UNRESOLVED -- {exc}", file=sys.stderr)
        return EXIT_UNRESOLVED
    except LOCK.SectionError as exc:
        print(f"ledger_watermark: SECTION -- {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
