#!/usr/bin/env python3
"""
ONEX Baseline Reviewer Agent
Analyzes codebase against ONEX standards and generates findings.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class Finding:
    """Represents a single code review finding."""
    ruleset_version: str = "0.1"
    rule_id: str = ""
    severity: str = ""  # "error" or "warning"
    repo: str = ""
    path: str = ""
    line: int = 0
    message: str = ""
    evidence: Dict[str, Any] = None
    suggested_fix: str = ""
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint and self.path and self.rule_id:
            # Generate fingerprint from path:line:rule_id:evidence_hash
            evidence_str = json.dumps(self.evidence or {}, sort_keys=True)
            hash_input = f"{self.path}:{self.line}:{self.rule_id}:{evidence_str}"
            self.fingerprint = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def to_ndjson(self) -> str:
        """Convert to NDJSON format."""
        data = {k: v for k, v in asdict(self).items() if v}
        return json.dumps(data, separators=(',', ':'), ensure_ascii=True)


class ONEXBaselineReviewer:
    """ONEX Baseline Reviewer - analyzes code against ONEX standards."""

    def __init__(self, repo_name: str, policy: Dict[str, Any]):
        self.repo_name = repo_name
        self.policy = policy
        self.findings: List[Finding] = []

        # Compile regex patterns for efficiency
        self.patterns = {
            'CLASS_HEADER': re.compile(r'^\+class\s+([A-Z][A-Za-z0-9_]*)\s*\('),
            'PROTOCOL_DECORATOR': re.compile(r'^\+@runtime_checkable\s*$'),
            'IMPORT_LINE': re.compile(r'^\+(from|import)\s+[^#\n]+'),
            'DEF_HEADER': re.compile(r'^\+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\s*(?:->\s*(.+?))?\s*:'),
            'ANY_TOKEN': re.compile(r'^\+.*\bAny\b'),
            'OPTIONAL_TOKEN': re.compile(r'^\+.*\bOptional\['),
            'ASSERT_NOT_NONE': re.compile(r'^\+.*assert\s+.+\s+is\s+not\s+None'),
            'WAIVER': re.compile(r'#\s*onex:ignore\s+([A-Z_\.0-9]+)(?:\s+(.*))?'),
        }

        # Get forbidden imports for this repo
        self.forbidden_imports = []
        if repo_name in policy.get('repos', {}):
            repo_config = policy['repos'][repo_name]
            if 'forbids' in repo_config:
                self.forbidden_imports = [re.compile(p) for p in repo_config['forbids']]

    def analyze_diff(self, diff_content: str, file_path: str = None) -> List[Finding]:
        """Analyze a diff file/content for ONEX violations."""
        lines = diff_content.split('\n')
        current_file = None
        current_line = 0
        findings = []

        for i, line in enumerate(lines):
            # Track current file
            if line.startswith('diff --git'):
                match = re.search(r'b/(.+?)(?:\s|$)', line)
                if match:
                    current_file = match.group(1)
                    current_line = 0
                continue

            # Track line numbers
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1)) - 1
                continue

            # Only process added lines
            if line.startswith('+') and not line.startswith('+++'):
                current_line += 1

                if current_file:
                    # Check each rule
                    findings.extend(self._check_naming_rules(line, current_file, current_line, lines, i))
                    findings.extend(self._check_boundary_rules(line, current_file, current_line))
                    findings.extend(self._check_spi_rules(line, current_file, current_line, lines, i))
                    findings.extend(self._check_typing_rules(line, current_file, current_line))
                    findings.extend(self._check_waiver_rules(line, current_file, current_line))

            elif not line.startswith('-'):
                # Context line
                if line and current_line > 0:
                    current_line += 1

        return findings

    def _check_naming_rules(self, line: str, file_path: str, line_no: int,
                           all_lines: List[str], line_idx: int) -> List[Finding]:
        """Check naming convention rules."""
        findings = []

        # Check class naming
        class_match = self.patterns['CLASS_HEADER'].match(line)
        if class_match:
            class_name = class_match.group(1)

            # ONEX.NAMING.PROTOCOL_001
            if 'protocol_' in file_path and not class_name.startswith('Protocol'):
                findings.append(Finding(
                    rule_id="ONEX.NAMING.PROTOCOL_001",
                    severity="error",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="Protocol class does not start with 'Protocol'",
                    evidence={"class_name": class_name},
                    suggested_fix=f"Rename to Protocol{class_name}"
                ))

            # ONEX.NAMING.MODEL_001
            elif 'model_' in file_path and not class_name.startswith('Model'):
                findings.append(Finding(
                    rule_id="ONEX.NAMING.MODEL_001",
                    severity="warning",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="Model class does not start with 'Model'",
                    evidence={"class_name": class_name},
                    suggested_fix=f"Rename to Model{class_name}"
                ))

            # ONEX.NAMING.ENUM_001
            elif 'enum_' in file_path and not class_name.startswith('Enum'):
                findings.append(Finding(
                    rule_id="ONEX.NAMING.ENUM_001",
                    severity="warning",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="Enum class does not start with 'Enum'",
                    evidence={"class_name": class_name},
                    suggested_fix=f"Rename to Enum{class_name}"
                ))

            # ONEX.NAMING.NODE_001
            elif 'node_' in file_path and not class_name.startswith('Node'):
                findings.append(Finding(
                    rule_id="ONEX.NAMING.NODE_001",
                    severity="warning",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="Node class does not start with 'Node'",
                    evidence={"class_name": class_name},
                    suggested_fix=f"Rename to Node{class_name}"
                ))

        return findings

    def _check_boundary_rules(self, line: str, file_path: str, line_no: int) -> List[Finding]:
        """Check boundary/import rules."""
        findings = []

        # ONEX.BOUNDARY.FORBIDDEN_IMPORT_001
        if self.patterns['IMPORT_LINE'].match(line):
            for forbidden_pattern in self.forbidden_imports:
                if forbidden_pattern.match(line[1:]):  # Remove the '+' prefix
                    findings.append(Finding(
                        rule_id="ONEX.BOUNDARY.FORBIDDEN_IMPORT_001",
                        severity="error",
                        repo=self.repo_name,
                        path=file_path,
                        line=line_no,
                        message="Forbidden import detected",
                        evidence={"import_line": line[1:].strip()},
                        suggested_fix="Remove or replace this import"
                    ))
                    break

        return findings

    def _check_spi_rules(self, line: str, file_path: str, line_no: int,
                         all_lines: List[str], line_idx: int) -> List[Finding]:
        """Check SPI purity rules."""
        findings = []

        # Check if we're in SPI directory
        if 'omnibase_spi/' in file_path:
            # ONEX.SPI.FORBIDDEN_LIB_001
            forbidden_libs = ['os', 'pathlib', 'sqlite3', 'requests', 'httpx', 'socket']
            for lib in forbidden_libs:
                if re.search(rf'\b{lib}\b', line):
                    findings.append(Finding(
                        rule_id="ONEX.SPI.FORBIDDEN_LIB_001",
                        severity="error",
                        repo=self.repo_name,
                        path=file_path,
                        line=line_no,
                        message=f"Forbidden library '{lib}' used in SPI",
                        evidence={"library": lib, "line": line[1:].strip()},
                        suggested_fix=f"Remove usage of '{lib}' from SPI code"
                    ))

            # Check for open() calls
            if 'open(' in line:
                findings.append(Finding(
                    rule_id="ONEX.SPI.FORBIDDEN_LIB_001",
                    severity="error",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="File I/O operation 'open()' forbidden in SPI",
                    evidence={"operation": "open()", "line": line[1:].strip()},
                    suggested_fix="Remove file I/O operations from SPI code"
                ))

        # ONEX.SPI.RUNTIMECHECKABLE_001
        if 'Protocol' in line and 'class ' in line:
            # Check for @runtime_checkable decorator within 5 lines before
            has_decorator = False
            for j in range(max(0, line_idx - 5), line_idx):
                if self.patterns['PROTOCOL_DECORATOR'].match(all_lines[j]):
                    has_decorator = True
                    break

            if not has_decorator:
                class_match = self.patterns['CLASS_HEADER'].match(line)
                if class_match:
                    findings.append(Finding(
                        rule_id="ONEX.SPI.RUNTIMECHECKABLE_001",
                        severity="error",
                        repo=self.repo_name,
                        path=file_path,
                        line=line_no,
                        message="Protocol class missing @runtime_checkable decorator",
                        evidence={"class_name": class_match.group(1)},
                        suggested_fix="Add @runtime_checkable decorator before the Protocol class"
                    ))

        return findings

    def _check_typing_rules(self, line: str, file_path: str, line_no: int) -> List[Finding]:
        """Check typing hygiene rules."""
        findings = []

        # Skip test files
        if 'test_' in file_path or '_test.py' in file_path:
            return findings

        # ONEX.TYPE.UNANNOTATED_DEF_001
        def_match = self.patterns['DEF_HEADER'].match(line)
        if def_match:
            func_name = def_match.group(1)
            params = def_match.group(2)
            return_type = def_match.group(3)

            # Skip special methods
            if not func_name.startswith('__'):
                # Check for missing annotations
                if params and params != 'self' and ':' not in params:
                    findings.append(Finding(
                        rule_id="ONEX.TYPE.UNANNOTATED_DEF_001",
                        severity="warning",
                        repo=self.repo_name,
                        path=file_path,
                        line=line_no,
                        message="Function parameters lack type annotations",
                        evidence={"function": func_name, "params": params},
                        suggested_fix="Add type annotations to all parameters"
                    ))

                if not return_type:
                    findings.append(Finding(
                        rule_id="ONEX.TYPE.UNANNOTATED_DEF_001",
                        severity="warning",
                        repo=self.repo_name,
                        path=file_path,
                        line=line_no,
                        message="Function lacks return type annotation",
                        evidence={"function": func_name},
                        suggested_fix="Add return type annotation"
                    ))

        # ONEX.TYPE.ANY_001
        if self.patterns['ANY_TOKEN'].match(line):
            findings.append(Finding(
                rule_id="ONEX.TYPE.ANY_001",
                severity="warning",
                repo=self.repo_name,
                path=file_path,
                line=line_no,
                message="Use of 'Any' type detected",
                evidence={"line": line[1:].strip()},
                suggested_fix="Replace 'Any' with a specific type"
            ))

        # ONEX.TYPE.OPTIONAL_ASSERT_001
        if self.patterns['OPTIONAL_TOKEN'].match(line) and self.patterns['ASSERT_NOT_NONE'].match(line):
            findings.append(Finding(
                rule_id="ONEX.TYPE.OPTIONAL_ASSERT_001",
                severity="warning",
                repo=self.repo_name,
                path=file_path,
                line=line_no,
                message="Optional type immediately forced non-null",
                evidence={"line": line[1:].strip()},
                suggested_fix="Consider using non-optional type if value is always required"
            ))

        return findings

    def _check_waiver_rules(self, line: str, file_path: str, line_no: int) -> List[Finding]:
        """Check waiver hygiene rules."""
        findings = []

        waiver_match = self.patterns['WAIVER'].search(line)
        if waiver_match:
            rule_id = waiver_match.group(1)
            waiver_text = waiver_match.group(2) or ""

            # ONEX.WAIVER.MALFORMED_001
            if 'reason=' not in waiver_text or 'expires=' not in waiver_text:
                findings.append(Finding(
                    rule_id="ONEX.WAIVER.MALFORMED_001",
                    severity="warning",
                    repo=self.repo_name,
                    path=file_path,
                    line=line_no,
                    message="Waiver missing required 'reason=' or 'expires=' fields",
                    evidence={"waiver_rule": rule_id, "waiver_text": waiver_text},
                    suggested_fix="Add reason= and expires= fields to waiver"
                ))

            # ONEX.WAIVER.EXPIRED_001
            expires_match = re.search(r'expires=(\d{4}-\d{2}-\d{2})', waiver_text)
            if expires_match:
                expires_date = expires_match.group(1)
                try:
                    expires_dt = datetime.strptime(expires_date, '%Y-%m-%d')
                    if expires_dt.date() < datetime.now().date():
                        findings.append(Finding(
                            rule_id="ONEX.WAIVER.EXPIRED_001",
                            severity="error",
                            repo=self.repo_name,
                            path=file_path,
                            line=line_no,
                            message="Waiver has expired",
                            evidence={"expires": expires_date},
                            suggested_fix="Remove expired waiver or update expiration date"
                        ))
                except ValueError:
                    pass  # Invalid date format

        return findings

    def process_shard(self, shard_path: Path) -> List[Finding]:
        """Process a single diff shard."""
        with open(shard_path, 'r', encoding='utf-8', errors='ignore') as f:
            diff_content = f.read()

        findings = self.analyze_diff(diff_content)
        self.findings.extend(findings)
        return findings

    def generate_summary(self, total_files: int = 0, truncated: bool = False) -> str:
        """Generate a Markdown summary of findings."""
        # Count findings by severity and type
        error_count = sum(1 for f in self.findings if f.severity == 'error')
        warning_count = sum(1 for f in self.findings if f.severity == 'warning')

        # Group by rule category
        naming_count = sum(1 for f in self.findings if 'NAMING' in f.rule_id)
        boundary_count = sum(1 for f in self.findings if 'BOUNDARY' in f.rule_id)
        spi_count = sum(1 for f in self.findings if 'SPI' in f.rule_id)
        typing_count = sum(1 for f in self.findings if 'TYPE' in f.rule_id)
        waiver_count = sum(1 for f in self.findings if 'WAIVER' in f.rule_id)

        # Calculate risk score (0-100)
        risk_score = min(100, error_count * 10 + warning_count * 2)

        # Build summary
        summary = f"""## Executive Summary
Risk score: {risk_score}/100

Total findings: {len(self.findings)} ({error_count} errors, {warning_count} warnings)
- Naming violations: {naming_count}
- Boundary violations: {boundary_count}
- SPI violations: {spi_count}
- Typing violations: {typing_count}
- Waiver issues: {waiver_count}

## Top Violations
"""

        # Show top 5 errors first, then warnings
        top_findings = sorted(self.findings, key=lambda f: (f.severity != 'error', f.rule_id))[:5]
        for finding in top_findings:
            summary += f"- [{finding.severity.upper()}] {finding.rule_id}: {finding.path}:{finding.line}\n"
            summary += f"  {finding.message}\n"

        if waiver_count > 0:
            summary += f"\n## Waiver Issues\n"
            summary += f"- {waiver_count} waiver(s) found with issues\n"

        summary += f"\n## Next Actions\n"
        if error_count > 0:
            summary += f"1. Fix {error_count} critical errors blocking compliance\n"
        if naming_count > 0:
            summary += f"2. Rename {naming_count} classes/files to follow ONEX conventions\n"
        if typing_count > 0:
            summary += f"3. Add type annotations to {typing_count} functions\n"

        summary += f"\n## Coverage\n"
        summary += f"- Reviewed {total_files} files\n"
        if truncated:
            summary += f"- Some files were truncated due to size limits\n"

        return summary


def main():
    """Main entry point for the baseline reviewer."""
    import argparse

    parser = argparse.ArgumentParser(description='ONEX Baseline Reviewer')
    parser.add_argument('--repo', default='omnibase_infra', help='Repository name')
    parser.add_argument('--input-dir', required=True, help='Input directory with shards')
    parser.add_argument('--output', default='baseline_review.out', help='Output file')

    args = parser.parse_args()

    # Hardcoded policy for omnibase_infra
    policy = {
        'ruleset_version': '0.1',
        'repos': {
            'omnibase_infra': {
                'forbids': [
                    r'^from\s+omniagent\b',
                    r'^from\s+omnimcp\b',
                    r'^import\s+omniagent\b',
                    r'^import\s+omnimcp\b',
                ]
            }
        }
    }

    # Create reviewer
    reviewer = ONEXBaselineReviewer(args.repo, policy)

    # Process all shards
    input_path = Path(args.input_dir)
    shard_dir = input_path / 'shards'

    if shard_dir.exists():
        shard_files = sorted(shard_dir.glob('diff_shard_*'))
        print(f"Processing {len(shard_files)} shards...")

        for shard in shard_files:
            reviewer.process_shard(shard)

    # Count total files
    files_list = input_path / 'files.list'
    total_files = 0
    if files_list.exists():
        with open(files_list, 'r') as f:
            total_files = len(f.readlines())

    # Generate output
    with open(args.output, 'w') as f:
        # Write NDJSON findings
        for finding in reviewer.findings:
            f.write(finding.to_ndjson() + '\n')

        # Write separator
        f.write('---ONEX-SEP---\n')

        # Write summary
        summary = reviewer.generate_summary(total_files=total_files)
        f.write(summary)

    print(f"Review complete! Found {len(reviewer.findings)} violations")
    print(f"Output written to {args.output}")


if __name__ == '__main__':
    main()