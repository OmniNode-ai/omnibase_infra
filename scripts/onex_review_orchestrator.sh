#!/usr/bin/env bash
# ONEX Review Orchestrator - Main entry point for baseline and nightly reviews

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config"
POLICY_FILE="$CONFIG_DIR/policy.yaml"

# Functions
print_header() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================================${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [MODE] [OPTIONS]

MODES:
  baseline    Run baseline review on entire non-archived codebase
  nightly     Run nightly incremental review since last run
  process     Process existing findings and generate reports

OPTIONS:
  -h, --help     Show this help message
  -p, --policy   Path to policy.yaml (default: config/policy.yaml)
  -o, --output   Output directory for reports
  -v, --verbose  Enable verbose output

EXAMPLES:
  # Run baseline review (first time setup)
  $(basename "$0") baseline

  # Run nightly review
  $(basename "$0") nightly

  # Process findings from previous run
  $(basename "$0") process .onex_nightly/*/review_output/findings.ndjson

  # Run with custom policy
  $(basename "$0") baseline --policy custom_policy.yaml

WORKFLOW:
  1. First run: ./$(basename "$0") baseline
  2. Daily runs: ./$(basename "$0") nightly
  3. Process results: ./$(basename "$0") process [findings.ndjson]

EOF
}

check_dependencies() {
    local missing=0

    # Check for required commands
    for cmd in git python3 csplit awk; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "Required command not found: $cmd"
            missing=1
        fi
    done

    # Check for policy file
    if [[ ! -f "$POLICY_FILE" ]]; then
        print_warning "Policy file not found at $POLICY_FILE"
        print_warning "Creating default policy..."
        mkdir -p "$CONFIG_DIR"
        # Could generate default policy here
        missing=1
    fi

    if [[ $missing -eq 1 ]]; then
        print_error "Missing dependencies. Please install required tools."
        exit 1
    fi

    print_success "All dependencies satisfied"
}

run_baseline() {
    print_header "ONEX BASELINE REVIEW"

    # Check if already run
    if [[ -d ".onex_baseline" ]]; then
        print_warning "Baseline directory already exists"
        read -p "Continue and create new baseline? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi

    # Run baseline producer
    print_header "Step 1: Generating Baseline Diff"
    bash "$SCRIPTS_DIR/onex_baseline_producer.sh"

    # Find latest baseline output
    LATEST_BASELINE=$(find .onex_baseline -maxdepth 2 -type d -name "[0-9]*" | sort -r | head -1)

    if [[ -z "$LATEST_BASELINE" ]]; then
        print_error "No baseline output found"
        exit 1
    fi

    print_success "Baseline diff generated: $LATEST_BASELINE"

    # Run agent
    print_header "Step 2: Running Baseline Agent Review"
    python3 "$SCRIPTS_DIR/onex_agent_runner.py" baseline "$LATEST_BASELINE" --policy "$POLICY_FILE"

    # Process results
    FINDINGS_FILE="$LATEST_BASELINE/review_output/findings.ndjson"
    if [[ -f "$FINDINGS_FILE" ]]; then
        print_header "Step 3: Processing Findings"
        python3 "$SCRIPTS_DIR/onex_findings_processor.py" "$FINDINGS_FILE" --output "$LATEST_BASELINE/reports"
        print_success "Baseline review complete!"
        print_success "Results: $LATEST_BASELINE/review_output/"
        print_success "Reports: $LATEST_BASELINE/reports/"
    else
        print_error "No findings generated"
        exit 1
    fi
}

run_nightly() {
    print_header "ONEX NIGHTLY REVIEW"

    # Check for baseline
    if [[ ! -f ".onex_nightly_prev" ]] && [[ ! -d ".onex_baseline" ]]; then
        print_error "No baseline found. Run 'baseline' mode first."
        exit 1
    fi

    # Run nightly producer
    print_header "Step 1: Generating Nightly Diff"
    bash "$SCRIPTS_DIR/onex_nightly_producer.sh"

    # Find latest nightly output
    LATEST_NIGHTLY=$(find .onex_nightly -maxdepth 2 -type d -name "[0-9]*" 2>/dev/null | sort -r | head -1)

    if [[ -z "$LATEST_NIGHTLY" ]]; then
        print_warning "No changes detected since last run"
        exit 0
    fi

    print_success "Nightly diff generated: $LATEST_NIGHTLY"

    # Run agent
    print_header "Step 2: Running Nightly Agent Review"
    python3 "$SCRIPTS_DIR/onex_agent_runner.py" nightly "$LATEST_NIGHTLY" --policy "$POLICY_FILE"

    # Process results
    FINDINGS_FILE="$LATEST_NIGHTLY/review_output/findings.ndjson"
    if [[ -f "$FINDINGS_FILE" ]]; then
        print_header "Step 3: Processing Findings"
        python3 "$SCRIPTS_DIR/onex_findings_processor.py" "$FINDINGS_FILE" --output "$LATEST_NIGHTLY/reports"

        # Update marker on success
        METADATA_FILE="$LATEST_NIGHTLY/metadata.json"
        if [[ -f "$METADATA_FILE" ]]; then
            HEAD_SHA=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['head_sha'])")
            echo "$HEAD_SHA" > .onex_nightly_prev
            print_success "Updated marker to $HEAD_SHA"
        fi

        print_success "Nightly review complete!"
        print_success "Results: $LATEST_NIGHTLY/review_output/"
        print_success "Reports: $LATEST_NIGHTLY/reports/"
    else
        print_warning "No findings generated"
    fi
}

process_findings() {
    local findings_file="$1"

    if [[ ! -f "$findings_file" ]]; then
        print_error "Findings file not found: $findings_file"
        exit 1
    fi

    print_header "PROCESSING FINDINGS"

    OUTPUT_DIR="${2:-$(dirname "$findings_file")/reports}"
    python3 "$SCRIPTS_DIR/onex_findings_processor.py" "$findings_file" --output "$OUTPUT_DIR"

    print_success "Processing complete!"
    print_success "Reports saved to: $OUTPUT_DIR"
}

# Main execution
main() {
    cd "$REPO_ROOT"

    # Parse arguments
    MODE="${1:-}"
    shift || true

    case "$MODE" in
        baseline)
            check_dependencies
            run_baseline
            ;;
        nightly)
            check_dependencies
            run_nightly
            ;;
        process)
            if [[ -z "${1:-}" ]]; then
                print_error "Please specify findings.ndjson file"
                usage
                exit 1
            fi
            process_findings "$@"
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            print_error "Invalid mode: $MODE"
            usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"