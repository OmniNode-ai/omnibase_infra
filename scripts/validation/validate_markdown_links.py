#!/usr/bin/env python3
"""Markdown link validation for ONEX repositories.

This script validates that all internal links in markdown files point to existing
files and anchors. This prevents broken documentation links from being committed.

VALIDATION SCOPE:
    - Internal links (relative paths like ./docs/foo.md or ../README.md)
    - Anchor links within same file (#section-name)
    - Cross-file anchors (./file.md#section)

    External links (http://, https://) are optionally validated with configurable
    timeout and can be disabled entirely.

CONFIGURATION:
    Configuration is loaded from .markdown-link-check.json in the repository root.
    See CONFIGURATION FORMAT below for details.

Usage:
    python scripts/validation/validate_markdown_links.py
    python scripts/validation/validate_markdown_links.py --verbose
    python scripts/validation/validate_markdown_links.py --check-external
    python scripts/validation/validate_markdown_links.py docs/

Exit Codes:
    0 - All links valid
    1 - Broken links found
    2 - Script error

CONFIGURATION FORMAT (.markdown-link-check.json):
    {
        "ignorePatterns": [
            {"pattern": "^https://example\\.com"},
            {"pattern": "^#"}
        ],
        "excludeFiles": [
            "CHANGELOG.md",
            "archived/**/*.md"
        ],
        "checkExternal": false,
        "externalTimeout": 5000
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# Constants
# =============================================================================

# Regex to match markdown links: [text](url) and [text](url "title")
MARKDOWN_LINK_PATTERN = re.compile(
    r'\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+"[^"]*")?\)'
)

# Regex to match reference-style links: [text][ref] with [ref]: url
MARKDOWN_REF_LINK_PATTERN = re.compile(r"\[(?P<text>[^\]]+)\]\[(?P<ref>[^\]]*)\]")
MARKDOWN_REF_DEFINITION_PATTERN = re.compile(
    r"^\[(?P<ref>[^\]]+)\]:\s*(?P<url>\S+)", re.MULTILINE
)

# Regex to match HTML anchor tags for heading extraction
HTML_ANCHOR_PATTERN = re.compile(r'<a\s+(?:name|id)=["\']([^"\']+)["\']', re.IGNORECASE)

# Default configuration
DEFAULT_CONFIG = {
    "ignorePatterns": [],
    "excludeFiles": [
        ".pytest_cache/**",
        ".venv/**",
        "venv/**",
        "node_modules/**",
        "archived/**",
    ],
    "checkExternal": False,
    "externalTimeout": 5000,
}

# Sentinel URL for reference-style links with missing definitions
MISSING_REF_SENTINEL = "__ONEX_MISSING_REF__"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LinkInfo:
    """Information about a markdown link."""

    url: str
    text: str
    line_number: int
    source_file: Path

    def __str__(self) -> str:
        return f"{self.source_file}:{self.line_number}: [{self.text}]({self.url})"


@dataclass
class BrokenLink:
    """A broken link with diagnostic information."""

    link: LinkInfo
    reason: str

    def __str__(self) -> str:
        return f"{self.link}\n    Reason: {self.reason}"


@dataclass
class ValidationResult:
    """Result of markdown link validation."""

    broken_links: list[BrokenLink] = field(default_factory=list)
    files_checked: int = 0
    links_checked: int = 0
    links_skipped: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.broken_links) == 0

    def __bool__(self) -> bool:
        """Returns True if validation passed (no broken links)."""
        return self.is_valid


@dataclass
class MarkdownLinkConfig:
    """Configuration for markdown link validation."""

    ignore_patterns: list[re.Pattern[str]] = field(default_factory=list)
    exclude_files: list[str] = field(default_factory=list)
    check_external: bool = False
    external_timeout: int = 5000  # milliseconds

    @classmethod
    def from_file(cls, config_path: Path) -> MarkdownLinkConfig:
        """Load configuration from JSON file."""
        if not config_path.exists():
            return cls.from_dict(DEFAULT_CONFIG)

        try:
            with open(config_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return cls.from_dict(DEFAULT_CONFIG)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MarkdownLinkConfig:
        """Create configuration from dictionary."""
        ignore_patterns = []
        raw_patterns = data.get("ignorePatterns", [])
        if isinstance(raw_patterns, list):
            for item in raw_patterns:
                if isinstance(item, dict) and "pattern" in item:
                    pattern_str = item["pattern"]
                    if isinstance(pattern_str, str):
                        try:
                            ignore_patterns.append(re.compile(pattern_str))
                        except re.error:
                            print(f"Warning: Invalid regex pattern: {pattern_str}")

        exclude_files = []
        raw_excludes = data.get("excludeFiles", DEFAULT_CONFIG["excludeFiles"])
        if isinstance(raw_excludes, list):
            exclude_files = [str(x) for x in raw_excludes if isinstance(x, str)]

        check_external = bool(data.get("checkExternal", False))
        external_timeout_raw = data.get("externalTimeout", 5000)
        external_timeout = (
            int(external_timeout_raw)
            if isinstance(external_timeout_raw, (int, float))
            else 5000
        )

        return cls(
            ignore_patterns=ignore_patterns,
            exclude_files=exclude_files,
            check_external=check_external,
            external_timeout=external_timeout,
        )


# =============================================================================
# Link Extraction
# =============================================================================


def _remove_code_blocks(content: str) -> str:
    """Remove fenced code blocks from content to avoid false positives.

    Replaces fenced code blocks (``` ... ```) with empty lines to preserve
    line number alignment while excluding code from link extraction.
    """
    result_lines = []
    in_code_block = False
    code_fence_pattern = re.compile(r"^```")

    for line in content.split("\n"):
        if code_fence_pattern.match(line):
            in_code_block = not in_code_block
            result_lines.append("")  # Preserve line count
        elif in_code_block:
            result_lines.append("")  # Preserve line count
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def _remove_inline_code(line: str) -> str:
    """Remove inline code spans from a line to avoid false positives.

    Inline code (backticks) can contain patterns that look like reference-style
    links, e.g., `dict["key"]["value"]` would match as [key][value].
    """
    return re.sub(r"`[^`]+`", "", line)


def extract_links_from_markdown(content: str, source_file: Path) -> Iterator[LinkInfo]:
    """Extract all links from markdown content."""
    # Build reference definitions map (before removing code blocks)
    ref_definitions: dict[str, str] = {}
    for match in MARKDOWN_REF_DEFINITION_PATTERN.finditer(content):
        ref_definitions[match.group("ref").lower()] = match.group("url")

    # Remove fenced code blocks to avoid false positives
    content_without_code = _remove_code_blocks(content)
    lines = content_without_code.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # Remove inline code to avoid matching dict["key"]["value"] as links
        line_without_inline_code = _remove_inline_code(line)

        # Extract inline links [text](url)
        for match in MARKDOWN_LINK_PATTERN.finditer(line_without_inline_code):
            yield LinkInfo(
                url=match.group("url"),
                text=match.group("text"),
                line_number=line_num,
                source_file=source_file,
            )

        # Extract reference-style links [text][ref]
        for match in MARKDOWN_REF_LINK_PATTERN.finditer(line_without_inline_code):
            ref = match.group("ref") or match.group("text")
            url = ref_definitions.get(ref.lower())
            if url:
                yield LinkInfo(
                    url=url,
                    text=match.group("text"),
                    line_number=line_num,
                    source_file=source_file,
                )
            else:
                # Yield sentinel for undefined reference - will be caught during validation
                yield LinkInfo(
                    url=f"{MISSING_REF_SENTINEL}:{ref}",
                    text=match.group("text"),
                    line_number=line_num,
                    source_file=source_file,
                )


def extract_headings_as_anchors(content: str) -> set[str]:
    """Extract all heading anchors from markdown content.

    GitHub-style anchor generation:
    - Lowercase
    - Replace spaces with hyphens
    - Remove punctuation except hyphens
    - Handle duplicates with -1, -2 suffix for disambiguation
    """
    anchors: set[str] = set()
    anchor_counts: dict[str, int] = {}

    # Match ATX headings: # Heading, ## Heading, etc.
    heading_pattern = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    for match in heading_pattern.finditer(content):
        heading_text = match.group(1).strip()
        base_anchor = _heading_to_anchor(heading_text)

        # Handle duplicate headings with -1, -2 suffixes (GitHub style)
        if base_anchor in anchor_counts:
            # This is a duplicate - add suffix
            suffix = anchor_counts[base_anchor]
            unique_anchor = f"{base_anchor}-{suffix}"
            anchor_counts[base_anchor] += 1
        else:
            # First occurrence - no suffix needed
            unique_anchor = base_anchor
            anchor_counts[base_anchor] = 1

        anchors.add(unique_anchor)

    # Match HTML anchors: <a name="anchor"> or <a id="anchor">
    for match in HTML_ANCHOR_PATTERN.finditer(content):
        anchors.add(match.group(1))

    return anchors


def _heading_to_anchor(heading: str) -> str:
    """Convert a heading to its GitHub-style anchor."""
    # Remove inline code backticks
    anchor = re.sub(r"`[^`]+`", "", heading)
    # Remove images
    anchor = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", anchor)
    # Remove links but keep text
    anchor = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", anchor)
    # Lowercase
    anchor = anchor.lower()
    # Replace spaces with hyphens
    anchor = anchor.replace(" ", "-")
    # Remove punctuation except hyphens and underscores
    anchor = re.sub(r"[^\w\-]", "", anchor)
    # Remove consecutive hyphens
    anchor = re.sub(r"-+", "-", anchor)
    # Strip leading/trailing hyphens
    anchor = anchor.strip("-")
    return anchor


# =============================================================================
# Link Validation
# =============================================================================


def is_external_link(url: str) -> bool:
    """Check if a URL is an external link."""
    return url.startswith(("http://", "https://", "mailto:", "ftp://", "//"))


def is_ignored(url: str, config: MarkdownLinkConfig) -> bool:
    """Check if a URL should be ignored based on configuration."""
    for pattern in config.ignore_patterns:
        if pattern.search(url):
            return True
    return False


def validate_internal_link(
    link: LinkInfo,
    repo_root: Path,
    file_anchors_cache: dict[Path, set[str]],
) -> str | None:
    """Validate an internal link.

    Returns None if valid, or an error message if broken.
    """
    url = link.url

    # Handle missing reference-style link definitions
    if url.startswith(MISSING_REF_SENTINEL):
        ref_name = url[len(MISSING_REF_SENTINEL) + 1 :]  # Skip sentinel and colon
        return f"Reference-style link [{ref_name}] has no definition"

    # Handle anchor-only links (#section)
    if url.startswith("#"):
        anchor = url[1:]
        if link.source_file not in file_anchors_cache:
            try:
                content = link.source_file.read_text(encoding="utf-8")
                file_anchors_cache[link.source_file] = extract_headings_as_anchors(
                    content
                )
            except OSError:
                file_anchors_cache[link.source_file] = set()

        if anchor not in file_anchors_cache[link.source_file]:
            return f"Anchor '{anchor}' not found in file"
        return None

    # Split URL into path and anchor
    anchor_part: str | None
    if "#" in url:
        path_part, anchor_part = url.split("#", 1)
    else:
        path_part, anchor_part = url, None

    # Handle empty path (just anchor reference)
    if not path_part:
        return None  # Already handled above

    # Resolve the target file path
    # Handle repo-root relative links (starting with /)
    if path_part.startswith("/"):
        # Resolve relative to repository root, not source directory
        target_path = (repo_root / path_part.lstrip("/")).resolve()
    else:
        # Resolve relative to source file directory
        source_dir = link.source_file.parent
        target_path = (source_dir / path_part).resolve()

    # Ensure target is within repo (security check)
    try:
        target_path.relative_to(repo_root)
    except ValueError:
        return f"Link points outside repository: {target_path}"

    # Check if target exists
    if not target_path.exists():
        # Check if it's a directory with implicit index
        if (target_path.parent / (target_path.name + ".md")).exists():
            target_path = target_path.parent / (target_path.name + ".md")
        else:
            return f"Target file not found: {path_part}"

    # Validate anchor if present
    if anchor_part and target_path.suffix.lower() == ".md":
        if target_path not in file_anchors_cache:
            try:
                content = target_path.read_text(encoding="utf-8")
                file_anchors_cache[target_path] = extract_headings_as_anchors(content)
            except OSError:
                file_anchors_cache[target_path] = set()

        if anchor_part not in file_anchors_cache[target_path]:
            return f"Anchor '{anchor_part}' not found in {path_part}"

    return None


def validate_external_link(link: LinkInfo, timeout_ms: int) -> str | None:
    """Validate an external link by making an HTTP HEAD request.

    Returns None if valid, or an error message if broken.

    Security Note:
        Only HTTP/HTTPS schemes are permitted. Other schemes (file://, ftp://, etc.)
        are rejected to prevent SSRF and local file access vulnerabilities.
    """
    from urllib.parse import urlparse

    # Security: Only allow HTTP/HTTPS schemes
    parsed = urlparse(link.url)
    if parsed.scheme.lower() not in ("http", "https"):
        return f"Unsupported scheme: {parsed.scheme} (only http/https allowed)"

    try:
        import urllib.request
        from urllib.error import HTTPError, URLError

        timeout_s = timeout_ms / 1000.0
        # S310: URL scheme already validated above (only http/https allowed)
        req = urllib.request.Request(  # noqa: S310
            link.url,
            method="HEAD",
            headers={"User-Agent": "ONEX-LinkChecker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
            if response.status >= 400:
                return f"HTTP {response.status}"
        return None
    except HTTPError as e:
        # Some servers don't support HEAD, try GET
        if e.code == 405:
            try:
                # S310: URL scheme already validated above (only http/https allowed)
                req = urllib.request.Request(  # noqa: S310
                    link.url,
                    headers={"User-Agent": "ONEX-LinkChecker/1.0"},
                )
                with urllib.request.urlopen(  # noqa: S310
                    req, timeout=timeout_ms / 1000.0
                ):
                    return None
            except (HTTPError, URLError, OSError) as e2:
                return str(e2)
        return f"HTTP {e.code}"
    except URLError as e:
        return str(e.reason)
    except OSError as e:
        return str(e)


# =============================================================================
# File Discovery
# =============================================================================


def find_markdown_files(
    root_path: Path,
    exclude_patterns: list[str],
) -> Iterator[Path]:
    """Find all markdown files, respecting exclusion patterns."""
    import fnmatch

    for md_file in root_path.rglob("*.md"):
        relative_path = str(md_file.relative_to(root_path))

        # Check if file matches any exclusion pattern
        excluded = False
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                excluded = True
                break
            # Also check just the filename
            if fnmatch.fnmatch(md_file.name, pattern):
                excluded = True
                break

        if not excluded:
            yield md_file


# =============================================================================
# Main Validation
# =============================================================================


def validate_markdown_links(
    repo_root: Path,
    config: MarkdownLinkConfig,
    verbose: bool = False,
    target_path: Path | None = None,
) -> ValidationResult:
    """Validate all markdown links in the repository.

    Args:
        repo_root: Repository root path
        config: Validation configuration
        verbose: Enable verbose output
        target_path: Optional specific path to validate (file or directory)

    Returns:
        ValidationResult with any broken links found
    """
    result = ValidationResult()
    file_anchors_cache: dict[Path, set[str]] = {}

    # Determine which files to check
    if target_path:
        if target_path.is_file():
            md_files = [target_path]
        else:
            md_files = list(find_markdown_files(target_path, config.exclude_files))
    else:
        md_files = list(find_markdown_files(repo_root, config.exclude_files))

    for md_file in md_files:
        result.files_checked += 1

        if verbose:
            print(f"Checking: {md_file.relative_to(repo_root)}")

        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError as e:
            if verbose:
                print(f"  Warning: Could not read file: {e}")
            continue

        for link in extract_links_from_markdown(content, md_file):
            # Check if link should be ignored
            if is_ignored(link.url, config):
                result.links_skipped += 1
                if verbose:
                    print(f"  Skipped (ignored): {link.url}")
                continue

            result.links_checked += 1

            if is_external_link(link.url):
                if config.check_external:
                    error = validate_external_link(link, config.external_timeout)
                    if error:
                        result.broken_links.append(BrokenLink(link=link, reason=error))
                        if verbose:
                            print(f"  BROKEN (external): {link.url} - {error}")
                else:
                    result.links_skipped += 1
                    if verbose:
                        print(f"  Skipped (external): {link.url}")
            else:
                error = validate_internal_link(link, repo_root, file_anchors_cache)
                if error:
                    result.broken_links.append(BrokenLink(link=link, reason=error))
                    if verbose:
                        print(f"  BROKEN: {link.url} - {error}")
                elif verbose:
                    print(f"  OK: {link.url}")

    return result


def generate_report(result: ValidationResult, repo_root: Path) -> str:
    """Generate a validation report."""
    if result.is_valid:
        return (
            f"Markdown Links: PASS "
            f"({result.files_checked} files, "
            f"{result.links_checked} links checked, "
            f"{result.links_skipped} skipped)"
        )

    lines = [
        "MARKDOWN LINK VALIDATION FAILED",
        "=" * 60,
        "",
        f"Found {len(result.broken_links)} broken link(s):",
        "",
    ]

    for broken in result.broken_links:
        rel_path = broken.link.source_file.relative_to(repo_root)
        lines.append(f"  {rel_path}:{broken.link.line_number}")
        lines.append(f"    Link: [{broken.link.text}]({broken.link.url})")
        lines.append(f"    Reason: {broken.reason}")
        lines.append("")

    lines.extend(
        [
            "=" * 60,
            f"Summary: {result.files_checked} files, "
            f"{result.links_checked} links checked, "
            f"{len(result.broken_links)} broken, "
            f"{result.links_skipped} skipped",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate markdown links in ONEX repositories"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Specific file or directory to check (default: entire repository)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all checked links, not just broken ones",
    )
    parser.add_argument(
        "--check-external",
        action="store_true",
        help="Also validate external links (http/https)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: .markdown-link-check.json)",
    )

    args = parser.parse_args()

    # Determine repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent

    # Load configuration
    config_path = args.config or (repo_root / ".markdown-link-check.json")
    config = MarkdownLinkConfig.from_file(config_path)

    # Override check_external from command line
    if args.check_external:
        config.check_external = True

    # Determine target path
    target_path = Path(args.path).resolve() if args.path else None

    try:
        if args.verbose:
            print(f"Validating markdown links in: {repo_root}")
            print(f"Configuration: {config_path}")
            print(f"Check external: {config.check_external}")
            print("")

        result = validate_markdown_links(
            repo_root=repo_root,
            config=config,
            verbose=args.verbose,
            target_path=target_path,
        )

        report = generate_report(result, repo_root)
        print(report)

        return 0 if result.is_valid else 1

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
