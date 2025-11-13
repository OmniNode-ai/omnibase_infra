#!/bin/bash
# OmniNode Bridge Security Configuration Script
# Simple script to enable/disable security scanning for bridge project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to show current security status
show_status() {
    log_info "Current Security Configuration Status"
    echo "======================================"

    if [ -f ".env" ] && grep -q "ENABLE_SECURITY_SCANS=true" .env; then
        log_success "Security scans are ENABLED in CI/CD"
    else
        log_warning "Security scans are DISABLED in CI/CD"
    fi

    echo ""
    log_info "Available Security Workflows:"
    echo "  📋 Simplified Security Scanning (.github/workflows/security-enhanced.yml)"
    echo "     - Basic Bandit scanning with proper error handling"
    echo "     - Dependency vulnerability scanning with pip-audit"
    echo "     - Optional container security scanning"
    echo "     - Comprehensive but bridge-project-optimized"
    echo ""
    echo "  ⚡ Basic Security in Main CI (.github/workflows/ci.yml)"
    echo "     - Lightweight Bandit and pip-audit scans"
    echo "     - Only runs when ENABLE_SECURITY_SCANS=true"
    echo "     - Minimal overhead for development"
    echo ""

    if [ -f "pyproject.toml" ]; then
        echo "📦 Security Dependencies Status:"
        if grep -q "bandit.*=" pyproject.toml; then
            log_success "Bandit security scanner installed"
        else
            log_warning "Bandit security scanner not found"
        fi

        if grep -q "pip-audit.*=" pyproject.toml; then
            log_success "pip-audit vulnerability scanner installed"
        else
            log_warning "pip-audit vulnerability scanner not found"
        fi

        if grep -q "detect-secrets.*=" pyproject.toml; then
            log_success "detect-secrets scanner installed"
        else
            log_warning "detect-secrets scanner not found"
        fi
    fi
}

# Function to enable basic security in CI
enable_basic_security() {
    log_info "Enabling basic security scans in CI/CD..."

    # Create or update .env file
    if [ ! -f ".env" ]; then
        log_info "Creating .env file..."
        cp .env.example .env 2>/dev/null || touch .env
    fi

    # Remove existing ENABLE_SECURITY_SCANS line and add new one
    grep -v "ENABLE_SECURITY_SCANS" .env > .env.tmp 2>/dev/null || touch .env.tmp
    echo "ENABLE_SECURITY_SCANS=true" >> .env.tmp
    mv .env.tmp .env

    log_success "Basic security scans enabled in CI/CD"
    log_info "Security scans will run in the main CI workflow"
}

# Function to disable basic security in CI
disable_basic_security() {
    log_info "Disabling basic security scans in CI/CD..."

    if [ -f ".env" ]; then
        # Remove ENABLE_SECURITY_SCANS line or set to false
        grep -v "ENABLE_SECURITY_SCANS" .env > .env.tmp 2>/dev/null || touch .env.tmp
        echo "ENABLE_SECURITY_SCANS=false" >> .env.tmp
        mv .env.tmp .env
    else
        echo "ENABLE_SECURITY_SCANS=false" > .env
    fi

    log_success "Basic security scans disabled in CI/CD"
    log_info "Only the simplified security workflow will run (on schedule)"
}

# Function to run security scans locally
run_local_scan() {
    log_info "Running local security scans..."

    # Check if we're in a poetry environment
    if ! command -v poetry >/dev/null 2>&1; then
        log_error "Poetry not found. Please install Poetry first."
        exit 1
    fi

    # Create reports directory
    mkdir -p security-reports

    # Run Bandit scan
    log_info "Running Bandit security scan..."
    poetry run bandit -r src -f json -o security-reports/bandit-local.json || log_warning "Bandit found security issues"
    poetry run bandit -r src -f txt -o security-reports/bandit-local.txt || log_warning "Bandit found security issues"

    # Run pip-audit scan
    log_info "Running pip-audit dependency scan..."
    poetry run pip-audit --format=json --output=security-reports/pip-audit-local.json || log_warning "pip-audit found vulnerabilities"

    # Run detect-secrets scan if baseline exists
    if [ -f ".secrets.baseline" ]; then
        log_info "Running detect-secrets scan..."
        poetry run detect-secrets audit --baseline .secrets.baseline || log_warning "Secrets detection found issues"
    fi

    log_success "Local security scans completed"
    log_info "Results saved in security-reports/ directory"
}

# Function to update security tools
update_security_tools() {
    log_info "Updating security scanning tools..."

    if command -v poetry >/dev/null 2>&1; then
        poetry update bandit pip-audit detect-secrets
        log_success "Security tools updated via Poetry"
    else
        log_warning "Poetry not found. Install tools manually if needed."
    fi
}

# Function to show security recommendations
show_recommendations() {
    log_info "Security Recommendations for Bridge Project"
    echo "==========================================="
    echo ""
    echo "🔒 Essential Security Practices:"
    echo "  1. Enable basic security scans: $0 enable"
    echo "  2. Run local scans before commits: $0 scan"
    echo "  3. Review security reports regularly"
    echo "  4. Keep dependencies updated"
    echo "  5. Use environment variables for secrets"
    echo ""
    echo "⚡ For Development Speed:"
    echo "  - Use basic security mode: $0 enable"
    echo "  - Security scans won't fail builds"
    echo "  - Focus on HIGH/CRITICAL issues only"
    echo ""
    echo "🏗️ For Production Readiness:"
    echo "  - Enable comprehensive scanning"
    echo "  - Review container security settings"
    echo "  - Implement proper secret management"
    echo "  - Enable security monitoring"
    echo ""
    echo "🚀 Bridge Project Optimizations:"
    echo "  - Simplified workflows for faster development"
    echo "  - Non-blocking security scans"
    echo "  - Essential security checks only"
    echo "  - Proper error handling and reporting"
}

# Function to check security tool installation
check_tools() {
    log_info "Checking security tool installation..."

    tools_ok=true

    if command -v poetry >/dev/null 2>&1; then
        log_success "Poetry found"

        # Check if security dependencies are installed
        if poetry run bandit --version >/dev/null 2>&1; then
            log_success "Bandit installed"
        else
            log_warning "Bandit not installed"
            tools_ok=false
        fi

        if poetry run pip-audit --version >/dev/null 2>&1; then
            log_success "pip-audit installed"
        else
            log_warning "pip-audit not installed"
            tools_ok=false
        fi

        if poetry run detect-secrets --version >/dev/null 2>&1; then
            log_success "detect-secrets installed"
        else
            log_warning "detect-secrets not installed"
            tools_ok=false
        fi
    else
        log_error "Poetry not found"
        tools_ok=false
    fi

    if [ "$tools_ok" = true ]; then
        log_success "All security tools are available"
    else
        log_warning "Some security tools are missing. Run 'poetry install --with dev' to install them."
    fi
}

# Main function
main() {
    case "${1:-status}" in
        "enable")
            enable_basic_security
            ;;
        "disable")
            disable_basic_security
            ;;
        "scan")
            run_local_scan
            ;;
        "status")
            show_status
            ;;
        "update")
            update_security_tools
            ;;
        "check")
            check_tools
            ;;
        "recommendations")
            show_recommendations
            ;;
        "help"|"--help"|"-h")
            echo "OmniNode Bridge Security Configuration"
            echo "====================================="
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  enable         Enable basic security scans in CI/CD"
            echo "  disable        Disable basic security scans in CI/CD"
            echo "  scan           Run security scans locally"
            echo "  status         Show current security configuration"
            echo "  update         Update security scanning tools"
            echo "  check          Check security tool installation"
            echo "  recommendations Show security recommendations"
            echo "  help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 enable      # Enable security scans"
            echo "  $0 scan        # Run local security scan"
            echo "  $0 status      # Check current configuration"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
