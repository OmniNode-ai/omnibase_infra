#!/usr/bin/env python3
"""ONEX Findings Processor - Analyze and report on review findings"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


class FindingsProcessor:
    """Process and analyze ONEX review findings"""

    def __init__(self, findings_path: str):
        self.findings_path = Path(findings_path)
        self.findings = []
        self.load_findings()

    def load_findings(self) -> None:
        """Load NDJSON findings from file"""
        if not self.findings_path.exists():
            print(f"Error: Findings file not found: {self.findings_path}")
            sys.exit(1)

        with open(self.findings_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    finding = json.loads(line)
                    self.findings.append(finding)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")

        print(f"Loaded {len(self.findings)} findings")

    def analyze_by_severity(self) -> Dict[str, List[Dict]]:
        """Group findings by severity"""
        by_severity = defaultdict(list)
        for finding in self.findings:
            severity = finding.get("severity", "unknown")
            by_severity[severity].append(finding)
        return dict(by_severity)

    def analyze_by_rule(self) -> Dict[str, List[Dict]]:
        """Group findings by rule ID"""
        by_rule = defaultdict(list)
        for finding in self.findings:
            rule_id = finding.get("rule_id", "UNKNOWN")
            by_rule[rule_id].append(finding)
        return dict(by_rule)

    def analyze_by_file(self) -> Dict[str, List[Dict]]:
        """Group findings by file path"""
        by_file = defaultdict(list)
        for finding in self.findings:
            path = finding.get("path", "unknown")
            by_file[path].append(finding)
        return dict(by_file)

    def get_affected_files(self) -> Set[str]:
        """Get set of all affected files"""
        return {finding.get("path", "unknown") for finding in self.findings}

    def generate_fix_script(self) -> str:
        """Generate a script to help fix common issues"""
        script_lines = ["#!/usr/bin/env python3",
                        "# Auto-generated fix suggestions for ONEX findings",
                        "",
                        "import os",
                        "import re",
                        "from pathlib import Path",
                        "",
                        "def main():",
                        "    fixes = []",
                        ""]

        # Group naming fixes
        naming_fixes = [f for f in self.findings if "NAMING" in f.get("rule_id", "")]

        for finding in naming_fixes:
            path = finding.get("path", "")
            line = finding.get("line", 0)
            evidence = finding.get("evidence", {})
            old_name = evidence.get("class_name", "")
            fix = finding.get("suggested_fix", "")

            if old_name and fix:
                script_lines.append(f"    # Fix: {fix}")
                script_lines.append(f"    # File: {path}:{line}")
                script_lines.append(f"    fixes.append({{")
                script_lines.append(f"        'file': '{path}',")
                script_lines.append(f"        'line': {line},")
                script_lines.append(f"        'old': '{old_name}',")
                script_lines.append(f"        'fix': '{fix}'")
                script_lines.append(f"    }})")
                script_lines.append("")

        script_lines.extend([
            "    print(f'Found {len(fixes)} potential fixes')",
            "    for fix in fixes:",
            "        print(f\"  {fix['file']}:{fix['line']} - {fix['fix']}\")",
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])

        return '\n'.join(script_lines)

    def generate_github_issues(self) -> List[Dict]:
        """Generate GitHub issue templates for findings"""
        issues = []

        # Group errors by category
        errors = [f for f in self.findings if f.get("severity") == "error"]
        by_category = defaultdict(list)

        for error in errors:
            rule_id = error.get("rule_id", "UNKNOWN")
            category = rule_id.split('.')[1] if '.' in rule_id else "OTHER"
            by_category[category].append(error)

        # Create issue per category
        for category, findings in by_category.items():
            title = f"[ONEX] Fix {category} violations ({len(findings)} errors)"

            body_lines = [
                f"## {category} Violations",
                "",
                f"Found {len(findings)} {category.lower()} errors that need immediate attention.",
                "",
                "### Findings:",
                ""
            ]

            for finding in findings[:10]:  # Limit to 10 per issue
                body_lines.append(f"- **{finding['path']}:{finding['line']}**")
                body_lines.append(f"  - Rule: `{finding['rule_id']}`")
                body_lines.append(f"  - Message: {finding['message']}")
                if finding.get('suggested_fix'):
                    body_lines.append(f"  - Fix: {finding['suggested_fix']}")
                body_lines.append("")

            if len(findings) > 10:
                body_lines.append(f"... and {len(findings) - 10} more")

            body_lines.extend([
                "",
                "### Labels:",
                "- onex-compliance",
                f"- {category.lower()}",
                "- automated",
                "",
                "### Priority:",
                "- 🔴 High (errors must be fixed)"
            ])

            issues.append({
                "title": title,
                "body": '\n'.join(body_lines),
                "labels": ["onex-compliance", category.lower(), "automated"]
            })

        return issues

    def print_report(self) -> None:
        """Print detailed analysis report"""
        print("\n" + "="*60)
        print("ONEX FINDINGS ANALYSIS REPORT")
        print("="*60)

        # Summary
        by_severity = self.analyze_by_severity()
        print("\n📊 Summary by Severity:")
        for severity in ["error", "warning", "unknown"]:
            count = len(by_severity.get(severity, []))
            if count > 0:
                symbol = "🔴" if severity == "error" else "🟡" if severity == "warning" else "⚪"
                print(f"  {symbol} {severity.capitalize()}: {count}")

        # By rule
        by_rule = self.analyze_by_rule()
        print(f"\n📋 Findings by Rule ({len(by_rule)} unique rules):")
        for rule_id, findings in sorted(by_rule.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  - {rule_id}: {len(findings)} occurrences")

        # By file
        by_file = self.analyze_by_file()
        print(f"\n📁 Most Affected Files ({len(by_file)} files total):")
        for path, findings in sorted(by_file.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  - {path}: {len(findings)} findings")

        # Categories
        categories = defaultdict(int)
        for finding in self.findings:
            rule_id = finding.get("rule_id", "UNKNOWN")
            if '.' in rule_id:
                category = rule_id.split('.')[1]
                categories[category] += 1

        print(f"\n🏷️ Findings by Category:")
        for category, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  - {category}: {count}")

        # Risk score calculation
        risk_score = min(100,
                         len(by_severity.get("error", [])) * 10 +
                         len(by_severity.get("warning", [])) * 2)
        print(f"\n⚠️ Risk Score: {risk_score}/100")

        # Next actions
        print("\n🎯 Recommended Actions:")
        errors = by_severity.get("error", [])
        if errors:
            print(f"  1. Fix {len(errors)} errors immediately")
        print(f"  2. Review {len(self.findings)} total findings")
        print(f"  3. Update waivers for false positives")
        print(f"  4. Run fix script for automated corrections")

    def save_reports(self, output_dir: str) -> None:
        """Save various report formats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save fix script
        fix_script = self.generate_fix_script()
        fix_script_path = output_path / "suggested_fixes.py"
        with open(fix_script_path, 'w') as f:
            f.write(fix_script)
        print(f"Saved fix script to {fix_script_path}")

        # Save GitHub issues
        issues = self.generate_github_issues()
        for i, issue in enumerate(issues, 1):
            issue_path = output_path / f"github_issue_{i}.json"
            with open(issue_path, 'w') as f:
                json.dump(issue, f, indent=2)
        print(f"Saved {len(issues)} GitHub issue templates")

        # Save detailed JSON report
        report = {
            "total_findings": len(self.findings),
            "by_severity": {k: len(v) for k, v in self.analyze_by_severity().items()},
            "by_rule": {k: len(v) for k, v in self.analyze_by_rule().items()},
            "affected_files": len(self.get_affected_files()),
            "risk_score": min(100,
                             len([f for f in self.findings if f.get("severity") == "error"]) * 10 +
                             len([f for f in self.findings if f.get("severity") == "warning"]) * 2)
        }
        report_path = output_path / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Process ONEX review findings")
    parser.add_argument("findings", help="Path to findings.ndjson file")
    parser.add_argument("--output", "-o", help="Output directory for reports")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress detailed output")

    args = parser.parse_args()

    processor = FindingsProcessor(args.findings)

    if not args.quiet:
        processor.print_report()

    if args.output:
        processor.save_reports(args.output)


if __name__ == "__main__":
    main()