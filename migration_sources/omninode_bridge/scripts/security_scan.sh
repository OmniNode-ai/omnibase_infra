#!/bin/bash
# Container Security Scanning Script for OmniNode Bridge
# Implements comprehensive security scanning using multiple tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCAN_RESULTS_DIR="security-reports"
DOCKERFILE_PATTERNS=("Dockerfile*" "*.dockerfile")
IMAGE_REGISTRY=${IMAGE_REGISTRY:-"omninode"}
SEVERITY_THRESHOLD=${SEVERITY_THRESHOLD:-"HIGH"}

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Setup scan results directory
setup_scan_environment() {
    log_info "Setting up security scan environment..."
    mkdir -p "$SCAN_RESULTS_DIR"
    chmod 755 "$SCAN_RESULTS_DIR"
    log_success "Scan environment ready"
}

# Install security scanning tools
install_security_tools() {
    log_info "Installing security scanning tools..."

    # Install Trivy (vulnerability scanner)
    if ! command -v trivy &> /dev/null; then
        log_info "Installing Trivy..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin v0.48.3
    fi

    # Install Hadolint (Dockerfile linter)
    if ! command -v hadolint &> /dev/null; then
        log_info "Installing Hadolint..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install hadolint
        else
            wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
            chmod +x /usr/local/bin/hadolint
        fi
    fi

    # Install Docker Scout CLI
    if ! docker scout version &> /dev/null; then
        log_info "Installing Docker Scout..."
        curl -sSfL https://raw.githubusercontent.com/docker/scout-cli/main/install.sh | sh -s --
    fi

    # Install Grype (alternative vulnerability scanner)
    if ! command -v grype &> /dev/null; then
        log_info "Installing Grype..."
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    fi

    log_success "Security tools installed"
}

# Scan Dockerfiles for best practices
scan_dockerfiles() {
    log_info "Scanning Dockerfiles for security best practices..."

    local dockerfile_issues=0
    local report_file="$SCAN_RESULTS_DIR/dockerfile-security-report.txt"

    echo "# Dockerfile Security Scan Report" > "$report_file"
    echo "Generated: $(date)" >> "$report_file"
    echo "" >> "$report_file"

    # Find all Dockerfiles
    for pattern in "${DOCKERFILE_PATTERNS[@]}"; do
        while IFS= read -r -d '' dockerfile; do
            log_info "Scanning: $dockerfile"
            echo "## Scanning: $dockerfile" >> "$report_file"

            # Hadolint scan
            if hadolint "$dockerfile" >> "$report_file" 2>&1; then
                log_success "$dockerfile: No issues found"
            else
                log_warning "$dockerfile: Issues found"
                dockerfile_issues=$((dockerfile_issues + 1))
            fi
            echo "" >> "$report_file"

        done < <(find . -name "$pattern" -print0)
    done

    # Custom security checks
    echo "## Custom Security Checks" >> "$report_file"
    perform_custom_dockerfile_checks "$report_file"

    if [[ $dockerfile_issues -eq 0 ]]; then
        log_success "All Dockerfiles passed security checks"
        return 0
    else
        log_warning "$dockerfile_issues Dockerfile(s) have security issues"
        return 1
    fi
}

# Perform custom Dockerfile security checks
perform_custom_dockerfile_checks() {
    local report_file="$1"

    log_info "Performing custom security checks..."

    # Check for non-root user
    echo "### Non-root User Check" >> "$report_file"
    if grep -r "USER.*root\|^USER 0" Dockerfile* 2>/dev/null; then
        echo "❌ CRITICAL: Found containers running as root user" >> "$report_file"
    elif ! grep -r "^USER " Dockerfile* 2>/dev/null; then
        echo "⚠️  WARNING: No explicit USER directive found (may default to root)" >> "$report_file"
    else
        echo "✅ PASS: Non-root user configured" >> "$report_file"
    fi

    # Check for hardcoded secrets
    echo "### Hardcoded Secrets Check" >> "$report_file"
    if grep -ri "password\|secret\|key\|token" Dockerfile* | grep -v "COPY\|ENV.*_FILE\|ENV.*_URL"; then
        echo "❌ CRITICAL: Potential hardcoded secrets found" >> "$report_file"
    else
        echo "✅ PASS: No obvious hardcoded secrets" >> "$report_file"
    fi

    # Check for HTTPS usage
    echo "### HTTPS Usage Check" >> "$report_file"
    if grep -r "http://" Dockerfile* 2>/dev/null | grep -v "localhost\|127.0.0.1"; then
        echo "⚠️  WARNING: HTTP URLs found (should use HTTPS)" >> "$report_file"
    else
        echo "✅ PASS: Using HTTPS or local URLs" >> "$report_file"
    fi

    echo "" >> "$report_file"
}

# Scan container images for vulnerabilities
scan_container_images() {
    log_info "Scanning container images for vulnerabilities..."

    local images_to_scan=(
        "omninode/hook-receiver:latest"
        "omninode/model-metrics:latest"
        "omninode/workflow-coordinator:latest"
    )

    for image in "${images_to_scan[@]}"; do
        log_info "Scanning image: $image"

        # Check if image exists locally
        if ! docker image inspect "$image" &> /dev/null; then
            log_warning "Image $image not found locally, skipping..."
            continue
        fi

        # Trivy scan
        scan_with_trivy "$image"

        # Docker Scout scan
        scan_with_docker_scout "$image"

        # Grype scan
        scan_with_grype "$image"
    done
}

# Trivy vulnerability scanning
scan_with_trivy() {
    local image="$1"
    local report_file="$SCAN_RESULTS_DIR/trivy-$(echo $image | tr '/' '-' | tr ':' '-').json"

    log_info "Running Trivy scan on $image..."

    trivy image \
        --format json \
        --output "$report_file" \
        --severity "$SEVERITY_THRESHOLD,CRITICAL" \
        --no-progress \
        "$image"

    # Generate summary
    local vuln_count
    vuln_count=$(jq '[.Results[]?.Vulnerabilities[]?] | length' "$report_file" 2>/dev/null || echo "0")

    if [[ "$vuln_count" -eq 0 ]]; then
        log_success "Trivy: No $SEVERITY_THRESHOLD+ vulnerabilities found in $image"
    else
        log_warning "Trivy: $vuln_count $SEVERITY_THRESHOLD+ vulnerabilities found in $image"
    fi
}

# Docker Scout scanning
scan_with_docker_scout() {
    local image="$1"
    local report_file="$SCAN_RESULTS_DIR/scout-$(echo $image | tr '/' '-' | tr ':' '-').json"

    log_info "Running Docker Scout scan on $image..."

    if docker scout cves --format json --output "$report_file" "$image" 2>/dev/null; then
        log_success "Docker Scout scan completed for $image"
    else
        log_warning "Docker Scout scan failed for $image (may require authentication)"
    fi
}

# Grype vulnerability scanning
scan_with_grype() {
    local image="$1"
    local report_file="$SCAN_RESULTS_DIR/grype-$(echo $image | tr '/' '-' | tr ':' '-').json"

    log_info "Running Grype scan on $image..."

    grype "$image" \
        --output json \
        --file "$report_file" \
        --only-fixed

    # Generate summary
    local vuln_count
    vuln_count=$(jq '[.matches[]] | length' "$report_file" 2>/dev/null || echo "0")

    if [[ "$vuln_count" -eq 0 ]]; then
        log_success "Grype: No vulnerabilities found in $image"
    else
        log_warning "Grype: $vuln_count vulnerabilities found in $image"
    fi
}

# Generate consolidated security report
generate_security_report() {
    log_info "Generating consolidated security report..."

    local report_file="$SCAN_RESULTS_DIR/security-summary-$(date +%Y%m%d-%H%M%S).md"

    cat > "$report_file" << EOF
# OmniNode Bridge Security Scan Report

**Generated:** $(date)
**Scan Type:** Comprehensive Container Security Scan
**Severity Threshold:** $SEVERITY_THRESHOLD

## Executive Summary

This report provides a comprehensive security assessment of the OmniNode Bridge container images and Dockerfiles.

## Dockerfile Security Analysis

$(cat "$SCAN_RESULTS_DIR/dockerfile-security-report.txt" 2>/dev/null || echo "No Dockerfile scan results available")

## Vulnerability Scan Summary

### Trivy Results
$(find "$SCAN_RESULTS_DIR" -name "trivy-*.json" -exec echo "- {}" \; 2>/dev/null || echo "No Trivy results available")

### Docker Scout Results
$(find "$SCAN_RESULTS_DIR" -name "scout-*.json" -exec echo "- {}" \; 2>/dev/null || echo "No Docker Scout results available")

### Grype Results
$(find "$SCAN_RESULTS_DIR" -name "grype-*.json" -exec echo "- {}" \; 2>/dev/null || echo "No Grype results available")

## Recommendations

1. **Regular Scanning**: Implement automated security scanning in CI/CD pipeline
2. **Base Image Updates**: Regularly update base images to latest security patches
3. **Dependency Management**: Keep application dependencies updated
4. **Runtime Security**: Implement runtime security monitoring
5. **Access Controls**: Ensure proper RBAC and network policies in Kubernetes

## Next Steps

1. Review and remediate identified vulnerabilities
2. Implement continuous security monitoring
3. Update security scanning thresholds as needed
4. Document remediation activities

EOF

    log_success "Security report generated: $report_file"
}

# Build optimized images for scanning
build_optimized_images() {
    log_info "Building optimized container images for security scanning..."

    # Build optimized images
    docker build -f Dockerfile.hook-receiver.optimized -t omninode/hook-receiver:latest .
    docker build -f Dockerfile.model-metrics.optimized -t omninode/model-metrics:latest .
    docker build -f Dockerfile.workflow-coordinator.optimized -t omninode/workflow-coordinator:latest .

    log_success "Optimized images built successfully"
}

# Main execution
main() {
    log_info "Starting OmniNode Bridge security scan..."

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --severity)
                SEVERITY_THRESHOLD="$2"
                shift 2
                ;;
            --install-tools)
                INSTALL_TOOLS=true
                shift
                ;;
            --dockerfile-only)
                DOCKERFILE_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --build           Build optimized images before scanning"
                echo "  --severity LEVEL  Set vulnerability severity threshold (LOW|MEDIUM|HIGH|CRITICAL)"
                echo "  --install-tools   Install security scanning tools"
                echo "  --dockerfile-only Only scan Dockerfiles, skip image scanning"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Setup environment
    setup_scan_environment

    # Install tools if requested
    if [[ "${INSTALL_TOOLS:-false}" == "true" ]]; then
        install_security_tools
    fi

    # Build images if requested
    if [[ "${BUILD_IMAGES:-false}" == "true" ]]; then
        build_optimized_images
    fi

    # Scan Dockerfiles
    dockerfile_scan_result=0
    scan_dockerfiles || dockerfile_scan_result=$?

    # Scan container images (unless dockerfile-only mode)
    image_scan_result=0
    if [[ "${DOCKERFILE_ONLY:-false}" != "true" ]]; then
        scan_container_images || image_scan_result=$?
    fi

    # Generate report
    generate_security_report

    # Final status
    if [[ $dockerfile_scan_result -eq 0 && $image_scan_result -eq 0 ]]; then
        log_success "Security scan completed successfully"
        exit 0
    else
        log_warning "Security scan completed with warnings"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
