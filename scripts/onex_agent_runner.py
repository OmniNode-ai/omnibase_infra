#!/usr/bin/env python3
"""ONEX Agent Runner - Invoke Opus baseline or nightly review agents"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SEPARATOR = "---ONEX-SEP---"

class ONEXAgentRunner:
    """Run ONEX Opus review agents"""

    def __init__(self, mode: str, input_dir: str, policy_path: str):
        self.mode = mode
        self.input_dir = Path(input_dir)
        self.policy_path = Path(policy_path)
        self.today = datetime.utcnow().strftime("%Y-%m-%d")
        self.findings = []
        self.summaries = []

    def load_policy(self) -> str:
        """Load policy.yaml content"""
        with open(self.policy_path, 'r') as f:
            return f.read()

    def load_input_file(self, filename: str) -> str:
        """Load an input file from the input directory"""
        file_path = self.input_dir / filename
        if not file_path.exists():
            return ""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def get_metadata(self) -> Dict:
        """Load metadata.json if available"""
        metadata_path = self.input_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def format_baseline_prompt(self, shard_path: Path) -> str:
        """Format baseline agent prompt"""
        metadata = self.get_metadata()

        with open(shard_path, 'r', encoding='utf-8', errors='replace') as f:
            diff_content = f.read()

        prompt = f"""context.repo: {metadata.get('repo', 'unknown')}
context.range: {metadata.get('empty_tree', '4b825dc642cb6eb9a060e54bf8d69288fbee4904')}...{metadata.get('head_sha', 'HEAD')}
today: {self.today}
policy.yaml:
{self.load_policy()}
git.stats:
{self.load_input_file('nightly.stats')}
git.names:
{self.load_input_file('nightly.names')}
git.diff:
{diff_content}"""

        return prompt

    def format_nightly_prompt(self) -> str:
        """Format nightly agent prompt"""
        metadata = self.get_metadata()

        prompt = f"""context.repo: {metadata.get('repo', 'unknown')}
context.range: {metadata.get('prev_sha', 'PREV')}...{metadata.get('head_sha', 'HEAD')}
today: {self.today}
policy.yaml:
{self.load_policy()}
git.stats:
{self.load_input_file('nightly.stats')}
git.names:
{self.load_input_file('nightly.names')}
git.diff:
{self.load_input_file('nightly.diff')}"""

        return prompt

    def get_system_prompt(self) -> str:
        """Get the appropriate system prompt"""
        if self.mode == "baseline":
            return """You are ONEX Baseline Reviewer. Operate only on provided inputs. Apply deterministic regex and filename rules for naming, boundary, SPI purity, typing, and waiver hygiene. Do not restate diffs. If evidence is insufficient, emit no finding for that rule.
Produce two outputs in order, separated by a single line exactly equal to:
---ONEX-SEP---
1) NDJSON findings. One compact JSON object per line. ASCII only.
2) A concise Markdown summary capped at 400 words.
Constraints:
- Do not read external sources. Do not infer repository content beyond the supplied inputs.
- Never include the full diff in outputs. Quote only minimal evidence.
- Prefer deterministic checks. Use the provided ruleset and severities."""

        else:  # nightly
            return """You are ONEX Nightly Reviewer. Operate only on provided inputs. Apply deterministic regex and filename rules to detect drift against naming, boundary, SPI purity, typing, and waiver hygiene policies. Do not restate diffs.
Produce NDJSON findings then a Markdown summary, separated by:
---ONEX-SEP---
Constraints identical to the Baseline Reviewer."""

    def parse_agent_response(self, response: str) -> Tuple[List[Dict], str]:
        """Parse agent response into findings and summary"""
        parts = response.split(SEPARATOR)

        if len(parts) != 2:
            print(f"Warning: Response doesn't contain separator {SEPARATOR}")
            return [], response

        ndjson_part = parts[0].strip()
        summary_part = parts[1].strip()

        # Parse NDJSON findings
        findings = []
        for line in ndjson_part.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                finding = json.loads(line)
                findings.append(finding)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse NDJSON line: {e}")
                print(f"  Line: {line[:100]}...")

        return findings, summary_part

    def invoke_agent(self, prompt: str) -> str:
        """
        Invoke the Opus agent with the given prompt.
        In production, this would call the actual Opus API.
        For now, returns a mock response.
        """
        # TODO: Replace with actual Opus API call
        print("\n" + "="*60)
        print("AGENT INVOCATION")
        print("="*60)
        print(f"Mode: {self.mode}")
        print(f"Prompt length: {len(prompt)} chars")
        print("="*60)

        # Mock response for testing
        mock_findings = [
            {
                "ruleset_version": "0.1",
                "rule_id": "ONEX.NAMING.PROTOCOL_001",
                "severity": "error",
                "repo": "omnibase-core",
                "path": "src/omnibase_core/protocols/protocol_event.py",
                "line": 12,
                "message": "Protocol class does not start with 'Protocol'",
                "evidence": {"class_name": "EventHandler"},
                "suggested_fix": "Rename to ProtocolEventHandler",
                "fingerprint": "a1b2c3d4"
            }
        ]

        mock_summary = """## Executive summary
Risk score: 25
- 1 naming error, 0 boundary violations, 0 SPI errors, 2 typing warnings

## Top violations
- Protocol class without prefix in protocol_event.py:12

## Waiver issues
- None found

## Next actions
- Rename EventHandler to ProtocolEventHandler

## Coverage
- Reviewed 42 files, none truncated"""

        # Format mock response
        mock_ndjson = '\n'.join(json.dumps(f) for f in mock_findings)
        return f"{mock_ndjson}\n{SEPARATOR}\n{mock_summary}"

    def run_baseline(self) -> None:
        """Run baseline review on all shards"""
        shards_dir = self.input_dir / "shards"

        if not shards_dir.exists():
            print(f"Error: No shards directory found at {shards_dir}")
            sys.exit(1)

        shard_files = sorted(shards_dir.glob("diff_shard_*.diff"))

        if not shard_files:
            print(f"Error: No shard files found in {shards_dir}")
            sys.exit(1)

        print(f"Processing {len(shard_files)} shards...")

        for i, shard_path in enumerate(shard_files, 1):
            print(f"\nProcessing shard {i}/{len(shard_files)}: {shard_path.name}")

            prompt = self.format_baseline_prompt(shard_path)
            response = self.invoke_agent(prompt)

            findings, summary = self.parse_agent_response(response)
            self.findings.extend(findings)
            self.summaries.append(summary)

    def run_nightly(self) -> None:
        """Run nightly incremental review"""
        print("Running nightly review...")

        prompt = self.format_nightly_prompt()
        response = self.invoke_agent(prompt)

        findings, summary = self.parse_agent_response(response)
        self.findings.extend(findings)
        self.summaries.append(summary)

    def generate_final_summary(self) -> str:
        """Generate consolidated summary from all shards"""
        if len(self.summaries) == 1:
            return self.summaries[0]

        # Aggregate findings by type
        by_severity = {"error": 0, "warning": 0}
        by_category = {}

        for finding in self.findings:
            severity = finding.get("severity", "warning")
            by_severity[severity] += 1

            rule_id = finding.get("rule_id", "UNKNOWN")
            category = rule_id.split('.')[1] if '.' in rule_id else "OTHER"
            by_category[category] = by_category.get(category, 0) + 1

        # Calculate risk score
        risk_score = min(100, by_severity["error"] * 10 + by_severity["warning"] * 2)

        summary = f"""## Executive summary
Risk score: {risk_score}
- {by_severity['error']} errors, {by_severity['warning']} warnings
- Categories: {', '.join(f'{k}:{v}' for k, v in by_category.items())}

## Top violations
"""
        # Add top 5 errors
        errors = [f for f in self.findings if f.get("severity") == "error"]
        for finding in errors[:5]:
            summary += f"- {finding['message']} in {finding['path']}:{finding['line']}\n"

        summary += f"""
## Waiver issues
- {sum(1 for f in self.findings if 'WAIVER' in f.get('rule_id', ''))} waiver-related findings

## Next actions
- Address {len(errors)} errors immediately
- Review {len(self.findings)} total findings

## Coverage
- Processed {len(self.summaries)} shards
- Total findings: {len(self.findings)}"""

        return summary

    def save_outputs(self) -> None:
        """Save findings and summary to output files"""
        output_dir = self.input_dir / "review_output"
        output_dir.mkdir(exist_ok=True)

        # Save NDJSON findings
        findings_path = output_dir / "findings.ndjson"
        with open(findings_path, 'w') as f:
            for finding in self.findings:
                f.write(json.dumps(finding) + '\n')
        print(f"Saved {len(self.findings)} findings to {findings_path}")

        # Save summary
        summary = self.generate_final_summary()
        summary_path = output_dir / "summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary to {summary_path}")

        # Save combined output
        combined_path = output_dir / "combined_output.txt"
        with open(combined_path, 'w') as f:
            for finding in self.findings:
                f.write(json.dumps(finding) + '\n')
            f.write(f"\n{SEPARATOR}\n\n")
            f.write(summary)
        print(f"Saved combined output to {combined_path}")

    def update_marker(self) -> None:
        """Update the nightly marker file after successful run"""
        if self.mode != "nightly":
            return

        metadata = self.get_metadata()
        head_sha = metadata.get("head_sha")

        if not head_sha:
            print("Warning: No head_sha in metadata, not updating marker")
            return

        marker_file = Path(".onex_nightly_prev")
        with open(marker_file, 'w') as f:
            f.write(head_sha)
        print(f"Updated marker to {head_sha}")

    def run(self) -> None:
        """Main execution"""
        print(f"ONEX Agent Runner - {self.mode} mode")
        print(f"Input directory: {self.input_dir}")
        print(f"Policy file: {self.policy_path}")
        print(f"Today: {self.today}")

        if self.mode == "baseline":
            self.run_baseline()
        else:
            self.run_nightly()

        self.save_outputs()

        # Only update marker after successful completion
        if self.findings:
            self.update_marker()

        print(f"\nCompleted with {len(self.findings)} findings")


def main():
    parser = argparse.ArgumentParser(description="ONEX Agent Runner")
    parser.add_argument("mode", choices=["baseline", "nightly"],
                        help="Review mode")
    parser.add_argument("input_dir",
                        help="Input directory containing git outputs")
    parser.add_argument("--policy", default="config/policy.yaml",
                        help="Path to policy.yaml (default: config/policy.yaml)")

    args = parser.parse_args()

    runner = ONEXAgentRunner(args.mode, args.input_dir, args.policy)
    runner.run()


if __name__ == "__main__":
    main()