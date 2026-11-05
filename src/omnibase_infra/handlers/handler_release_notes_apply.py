# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Write a release block where the target repository's docs policy accepts it.

OMN-17427: the release train wrote ``CHANGELOG.md`` for every repository.
omnibase_core deleted its root changelog under the README-only documentation
policy, so every core release PR was born red on ``kb-doc-gate``, the
``Root README only documentation gate`` hook and ``CI Summary``.

The owner of "does this repo take a changelog" is the repo's own
``.readme-only-docs.yaml``, the file the ``readme-only-docs`` validator reads. A
repo without it, or one that lists ``CHANGELOG.md`` under ``exceptions``, keeps the
historical ``CHANGELOG.md`` prepend. A README-only repo gets the block in root
``README.md`` above its newest release section, a root ``CHANGELOG.md`` removed,
and, where it carries its release-cut contract and identity test (omnibase_core),
the pinned test and ``dod_evidence`` item every accepted README-shaped cut has
carried: the DoD release-cut exemption reads a root ``CHANGELOG.md`` diff, so a
README-shaped cut is judged as an ordinary change and needs its own evidence.

The handler refuses, rather than guessing, on an unparseable policy file, a missing
README, a block in another shape, or a contract that does not end in its
``dod_evidence`` list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_parsed_release_entry import (
    ModelParsedReleaseEntry,
)
from omnibase_infra.models.model_release_notes_apply_request import (
    ModelReleaseNotesApplyRequest,
)
from omnibase_infra.models.model_release_notes_apply_result import (
    ModelReleaseNotesApplyResult,
)

README_ONLY_DOCS_FILE: Final = ".readme-only-docs.yaml"
_CHANGELOG_PATH: Final = "CHANGELOG.md"
_README_PATH: Final = "README.md"
_RELEASE_CONTRACT_PATH: Final = "contracts/OMN-18595.yaml"
_RELEASE_IDENTITY_TEST_PATH: Final = "tests/ci/test_release_cut_identity_omn18595.py"
_ENTRY_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^## v(\d+\.\d+\.\d+) \((\d{4}-\d{2}-\d{2})\)$"
)
_ENTRY_SINCE: Final[re.Pattern[str]] = re.compile(r"^### Included Since (.+)$")
_ENTRY_PR_NUMBER: Final[re.Pattern[str]] = re.compile(r"\(#\d+\)$")
_FENCE: Final[re.Pattern[str]] = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_PY_LINE_LIMIT: Final = 88


def release_notes_target(clone: Path) -> str:
    """Where this repo's release notes live: ``CHANGELOG.md`` or ``README.md``.

    A repo that carries ``.readme-only-docs.yaml`` and does not list
    ``CHANGELOG.md`` among its ``exceptions`` refuses a root changelog, so its
    notes go in the root README. Every other repo keeps the changelog. No
    train-side key: a second copy of the answer would drift from the gate that
    actually decides.
    """
    config = clone / README_ONLY_DOCS_FILE
    if not config.is_file():
        return _CHANGELOG_PATH
    try:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        msg = f"{README_ONLY_DOCS_FILE} does not parse: {exc}"
        raise ValueError(msg) from exc
    exceptions = loaded.get("exceptions", []) if isinstance(loaded, dict) else []
    allowed = {
        item.get("path")
        for item in exceptions
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    return _CHANGELOG_PATH if _CHANGELOG_PATH in allowed else _README_PATH


def parse_release_entry(entry: str) -> ModelParsedReleaseEntry:
    """Read the block ``render_changelog_entry`` wrote; refuse any other shape."""
    lines = entry.strip("\n").splitlines()
    heading = _ENTRY_HEADING.fullmatch(lines[0]) if lines else None
    if heading is None:
        msg = "release entry must start with '## v<version> (<date>)'"
        raise ValueError(msg)
    since = ""
    subjects: list[str] = []
    in_included = False
    for line in lines[1:]:
        found = _ENTRY_SINCE.fullmatch(line)
        if found is not None:
            since, in_included = found.group(1), True
        elif in_included and line.startswith("- "):
            subjects.append(line[2:])
    if not since:
        msg = "release entry has no '### Included Since' section"
        raise ValueError(msg)
    return ModelParsedReleaseEntry(
        version=heading.group(1),
        date=heading.group(2),
        since=since,
        subjects=tuple(subjects),
    )


def _release_sections(text: str) -> list[tuple[tuple[int, ...], int]]:
    """Read release headings outside Markdown fences, retaining character offsets."""
    sections: list[tuple[tuple[int, ...], int]] = []
    fence = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        marker = _FENCE.fullmatch(stripped)
        if marker is not None:
            run, suffix = marker.groups()
            if not fence:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not suffix.strip():
                fence = ""
        elif not fence and (heading := _ENTRY_HEADING.fullmatch(stripped)):
            sections.append((tuple(map(int, heading.group(1).split("."))), offset))
        offset += len(line)
    return sections


def _insert_entry_in_readme(readme: str, entry: str) -> str:
    """Put the block above the newest existing release section, else at the end."""
    block = entry.rstrip("\n") + "\n"
    sections = _release_sections(readme)
    if not sections:
        return readme.rstrip("\n") + "\n\n" + block
    _, offset = max(sections, key=lambda section: section[0])
    return readme[:offset] + block + "\n" + readme[offset:]


def _refuse_duplicate_release(text: str, parsed: ModelParsedReleaseEntry) -> None:
    version = tuple(map(int, parsed.version.split(".")))
    if any(existing == version for existing, _ in _release_sections(text)):
        msg = f"release {parsed.version} already has notes"
        raise ValueError(msg)


def _validate_companions(
    contract_text: str, test_text: str, parsed: ModelParsedReleaseEntry
) -> None:
    """Validate YAML structure and duplicate identities before any file is changed."""
    try:
        contract = yaml.compose(contract_text, Loader=yaml.SafeLoader)
        explicit_end = any(
            isinstance(event, yaml.DocumentEndEvent) and event.explicit
            for event in yaml.parse(contract_text, Loader=yaml.SafeLoader)
        )
    except yaml.YAMLError as exc:
        msg = f"{_RELEASE_CONTRACT_PATH} does not parse: {exc}"
        raise ValueError(msg) from exc
    if explicit_end or not isinstance(contract, yaml.MappingNode) or not contract.value:
        msg = f"{_RELEASE_CONTRACT_PATH} does not end with its dod_evidence list"
        raise ValueError(msg)
    key, value = contract.value[-1]
    if not (
        isinstance(key, yaml.ScalarNode)
        and key.value == "dod_evidence"
        and isinstance(value, yaml.SequenceNode)
        and not value.flow_style
        and value.start_mark.column == 2
    ):
        msg = f"{_RELEASE_CONTRACT_PATH} does not end with its dod_evidence list"
        raise ValueError(msg)
    item_id = f"dod-omn18595-ac1-release-{parsed.version.replace('.', '-')}-notes"
    for item in value.value:
        if isinstance(item, yaml.MappingNode) and any(
            isinstance(k, yaml.ScalarNode)
            and k.value == "id"
            and isinstance(v, yaml.ScalarNode)
            and v.value == item_id
            for k, v in item.value
        ):
            msg = f"release {parsed.version} already has dod_evidence"
            raise ValueError(msg)
    function = f"test_release_changelog_retains_version_{parsed.version.replace('.', '_')}_in_root_readme"
    if re.search(rf"^def {re.escape(function)}\(", test_text, flags=re.MULTILINE):
        msg = f"release {parsed.version} already has an identity test"
        raise ValueError(msg)


def _identity_test_source(parsed: ModelParsedReleaseEntry) -> str:
    slug = parsed.version.replace(".", "_")
    lines = [
        "",
        "",
        f"def test_release_changelog_retains_version_{slug}_in_root_readme() -> None:",
        f'    """Keep the {parsed.version} cut\'s public notes in root README, not a changelog."""',
        '    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")',
        f'    assert "## v{parsed.version} ({parsed.date})" in readme',
        f'    assert "### Included Since {parsed.since}" in readme',
    ]
    if parsed.subjects:
        literal = json.dumps(f"- {parsed.subjects[0]}", ensure_ascii=False)
        single = f"    assert {literal} in readme"
        inner = f"        {literal} in readme"
        if len(single) <= _PY_LINE_LIMIT:
            lines.append(single)
        elif len(inner) <= _PY_LINE_LIMIT:
            lines += ["    assert (", inner, "    )"]
        else:
            lines += [
                "    assert (",
                f"        {literal}",
                "        in readme",
                "    )",
            ]
    lines.append('    assert not (REPO_ROOT / "CHANGELOG.md").exists()')
    return "\n".join(lines) + "\n"


def _fold_statements(command: str, width: int = 90) -> list[str]:
    """Wrap a one-line command at its ``; `` boundaries, never inside a statement."""
    rows: list[str] = []
    statements = command.split("; ")
    for index, statement in enumerate(statements):
        piece = statement if index == len(statements) - 1 else statement + ";"
        if rows and len(rows[-1]) + 1 + len(piece) <= width:
            rows[-1] += " " + piece
        else:
            rows.append(piece)
    return rows


def _contract_item_source(parsed: ModelParsedReleaseEntry, repo_name: str) -> str:
    slug = parsed.version.replace(".", "-")
    checks = [
        f'assert "## v{parsed.version} ({parsed.date})" in notes',
        f'assert "### Included Since {parsed.since}" in notes',
    ]
    pins = [
        f'"{found.group(0)}" in notes'
        for subject in (parsed.subjects[:1] + parsed.subjects[-1:])
        if (found := _ENTRY_PR_NUMBER.search(subject)) is not None
    ]
    if pins:
        checks.append("assert " + " and ".join(dict.fromkeys(pins)))
    body = "; ".join(
        ["from pathlib import Path", 'notes = Path("README.md").read_text()']
        + ['assert not Path("CHANGELOG.md").exists()']
        + checks
    )
    return (
        f"  - id: dod-omn18595-ac1-release-{slug}-notes\n"
        "    description: The release tree passes the documentation policy and retains this release's notes in\n"
        "      root README.\n"
        "    source: manual\n"
        "    binds_ac: [AC1]\n"
        "    checks:\n"
        "      - check_type: test_passes\n"
        f"        check_value: uv run pytest {_RELEASE_IDENTITY_TEST_PATH} -q -k changelog\n"
        f"        cwd: ${{OMNI_HOME}}/{repo_name}\n"
        "      - check_type: command\n"
        "        check_value: >-\n"
        + "".join(
            f"          {row}\n"
            for row in _fold_statements(f"uv run --locked python -c '{body}'")
        )
        + f"        cwd: ${{OMNI_HOME}}/{repo_name}\n"
    )


def apply_release_notes(
    clone: Path, entry: str, *, repo_name: str | None = None
) -> tuple[str, list[str]]:
    """Write the release block into ``clone``; return the target and paths to stage."""
    parsed = parse_release_entry(entry)
    target = release_notes_target(clone)
    if target == _CHANGELOG_PATH:
        changelog = clone / _CHANGELOG_PATH
        existing = changelog.read_text(encoding="utf-8") if changelog.exists() else ""
        _refuse_duplicate_release(existing, parsed)
        changelog.write_text(entry.rstrip("\n") + "\n\n" + existing, encoding="utf-8")
        return target, [_CHANGELOG_PATH]

    readme = clone / _README_PATH
    if not readme.is_file():
        msg = f"{README_ONLY_DOCS_FILE} routes release notes to README.md, which is absent"
        raise ValueError(msg)
    contract = clone / _RELEASE_CONTRACT_PATH
    test = clone / _RELEASE_IDENTITY_TEST_PATH
    companions = contract.is_file() and test.is_file()
    contract_text = contract.read_text(encoding="utf-8") if companions else ""
    test_text = test.read_text(encoding="utf-8") if companions else ""
    if companions:
        _validate_companions(contract_text, test_text, parsed)
    readme_text = readme.read_text(encoding="utf-8")
    _refuse_duplicate_release(readme_text, parsed)

    readme.write_text(
        _insert_entry_in_readme(readme_text, entry),
        encoding="utf-8",
    )
    staged = [_README_PATH]
    stale = clone / _CHANGELOG_PATH
    if stale.exists():
        stale.unlink()
        staged.append(_CHANGELOG_PATH)
    if companions:
        item = _contract_item_source(parsed, repo_name or clone.resolve().name)
        contract.write_text(contract_text.rstrip("\n") + "\n" + item, encoding="utf-8")
        test.write_text(
            test_text.rstrip("\n") + "\n" + _identity_test_source(parsed),
            encoding="utf-8",
        )
        staged += [_RELEASE_CONTRACT_PATH, _RELEASE_IDENTITY_TEST_PATH]
    return target, staged


class HandlerReleaseNotesApply:
    """Place a rendered release block under the target repo's documentation policy."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelReleaseNotesApplyRequest
    ) -> ModelReleaseNotesApplyResult:
        try:
            entry = Path(request.entry_file).read_text(encoding="utf-8")
            target, staged = apply_release_notes(
                Path(request.clone_path), entry, repo_name=request.repo_name
            )
        except (OSError, ValueError) as exc:
            return ModelReleaseNotesApplyResult(success=False, error_message=str(exc))
        return ModelReleaseNotesApplyResult(
            success=True, target=target, staged=tuple(staged)
        )
