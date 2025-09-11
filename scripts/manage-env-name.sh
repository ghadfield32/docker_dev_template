#!/usr/bin/env bash
set -euo pipefail

# Environment Name Management Script
# ==================================
# This script helps you easily change and manage the environment name
# across all configuration files in your Docker development setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color functions for output
function log_info() { echo -e "\033[0;36mℹ️  $1\033[0m"; }
function log_success() { echo -e "\033[0;32m✅ $1\033[0m"; }
function log_warning() { echo -e "\033[0;33m⚠️  $1\033[0m"; }
function log_error() { echo -e "\033[0;31m❌ $1\033[0m"; }

# Function to show usage
show_usage() {
    cat << EOF
Environment Name Management Script
==================================

This script helps you manage the ENV_NAME across your Docker development setup.

USAGE:
    $0 <command> [options]

COMMANDS:
    show                    Show current environment name from all sources
    set <name>             Set new environment name in .env file
    validate               Validate current configuration consistency
    clean <old_name>       Clean up Docker resources for old environment name
    migrate <old> <new>    Migrate from old name to new name (full process)
    help                   Show this help message

EXAMPLES:
    $0 show                           # Show current settings
    $0 set my_ml_project             # Set ENV_NAME to my_ml_project
    $0 validate                      # Check configuration consistency  
    $0 clean old_project_name        # Clean up old Docker resources
    $0 migrate docker_dev_template my_project  # Full migration

NOTES:
    - The .env file is the primary source of truth for ENV_NAME
    - All Docker resources (containers, volumes, networks) are prefixed with ENV_NAME
    - Changing ENV_NAME requires rebuilding containers to take effect
    - Use 'clean' command to remove old Docker resources after changing names

EOF
}

# Function to get current env name from various sources
get_current_env_name() {
    local source=""
    local name=""
    
    # Check .env file first (primary source)
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "")
        if [[ -n "$name" ]]; then
            source=".env"
        fi
    fi
    
    # Check .devcontainer/.env as fallback
    if [[ -z "$name" && -f "$PROJECT_ROOT/.devcontainer/.env" ]]; then
        name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.devcontainer/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "")
        if [[ -n "$name" ]]; then
            source=".devcontainer/.env"
        fi
    fi
    
    # Check environment variable
    if [[ -n "${ENV_NAME:-}" ]]; then
        if [[ -n "$name" && "$name" != "$ENV_NAME" ]]; then
            log_warning "Environment variable ENV_NAME='$ENV_NAME' differs from file value '$name'"
        fi
        name="$ENV_NAME"
        source="environment variable"
    fi
    
    # Default fallback
    if [[ -z "$name" ]]; then
        name="docker_dev_template"
        source="default"
    fi
    
    echo "$name|$source"
}

# Function to show current environment name
show_current() {
    log_info "Current Environment Name Configuration:"
    echo "======================================"
    
    local result=$(get_current_env_name)
    local name="${result%|*}"
    local source="${result#*|}"
    
    echo "Active ENV_NAME: $name (from $source)"
    echo ""
    
    # Show all sources
    echo "Configuration Sources:"
    echo "----------------------"
    
    # .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        local env_name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "<not set>")
        echo ".env file:            $env_name"
    else
        echo ".env file:            <file not found>"
    fi
    
    # .devcontainer/.env file  
    if [[ -f "$PROJECT_ROOT/.devcontainer/.env" ]]; then
        local dev_env_name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.devcontainer/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "<not set>")
        echo ".devcontainer/.env:   $dev_env_name"
    else
        echo ".devcontainer/.env:   <file not found>"
    fi
    
    # Environment variable
    echo "ENV_NAME variable:    ${ENV_NAME:-<not set>}"
    
    echo ""
    echo "Docker Resources (if any):"
    echo "--------------------------"
    
    # Check for Docker resources with this name
    if command -v docker >/dev/null 2>&1; then
        echo "Containers:"
        docker ps -a --filter "name=${name}" --format "  {{.Names}} ({{.Status}})" 2>/dev/null || echo "  None found"
        
        echo "Volumes:"  
        docker volume ls --filter "name=${name}" --format "  {{.Name}}" 2>/dev/null || echo "  None found"
        
        echo "Compose Projects:"
        docker compose ls 2>/dev/null | grep "$name" | awk '{print "  " $1 " (" $2 ")"}' || echo "  None found"
    else
        echo "  Docker not available"
    fi
}

# Function to set new environment name
set_env_name() {
    local new_name="$1"
    
    if [[ -z "$new_name" ]]; then
        log_error "Environment name cannot be empty"
        return 1
    fi
    
    # Validate name (should be suitable for Docker resource naming)
    if [[ ! "$new_name" =~ ^[a-zA-Z][a-zA-Z0-9_-]*$ ]]; then
        log_error "Invalid environment name. Must start with letter and contain only letters, numbers, underscore, and hyphen."
        return 1
    fi
    
    local current_name=$(get_current_env_name | cut -d'|' -f1)
    
    if [[ "$current_name" == "$new_name" ]]; then
        log_warning "Environment name is already set to '$new_name'"
        return 0
    fi
    
    log_info "Setting environment name from '$current_name' to '$new_name'"
    
    # Create or update .env file
    local env_file="$PROJECT_ROOT/.env"
    
    if [[ -f "$env_file" ]]; then
        # Update existing file
        if grep -q "^ENV_NAME=" "$env_file"; then
            # Replace existing ENV_NAME line
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS sed
                sed -i '' "s/^ENV_NAME=.*/ENV_NAME=$new_name/" "$env_file"
            else
                # Linux sed
                sed -i "s/^ENV_NAME=.*/ENV_NAME=$new_name/" "$env_file"
            fi
        else
            # Add ENV_NAME to existing file
            echo "ENV_NAME=$new_name" >> "$env_file"
        fi
    else
        # Create new .env file from template
        if [[ -f "$PROJECT_ROOT/.devcontainer/.env.template" ]]; then
            cp "$PROJECT_ROOT/.devcontainer/.env.template" "$env_file"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/^ENV_NAME=.*/ENV_NAME=$new_name/" "$env_file"
            else
                sed -i "s/^ENV_NAME=.*/ENV_NAME=$new_name/" "$env_file"
            fi
        else
            # Create minimal .env file
            echo "ENV_NAME=$new_name" > "$env_file"
        fi
    fi
    
    log_success "Environment name set to '$new_name' in $env_file"
    
    # Update current session
    export ENV_NAME="$new_name"
    
    log_warning "You will need to rebuild containers for changes to take effect:"
    echo "  cd .devcontainer"
    echo "  docker compose -p $new_name down"  
    echo "  docker compose -p $new_name build --no-cache"
    echo "  docker compose -p $new_name up -d"
    echo ""
    echo "Or use: invoke down --name $current_name && invoke up --name $new_name --rebuild"
}

# Function to validate configuration consistency
validate_config() {
    log_info "Validating environment configuration..."
    
    local result=$(get_current_env_name)
    local name="${result%|*}"
    local issues=0
    
    echo "Primary ENV_NAME: $name"
    echo ""
    
    # Check file consistency
    local env_file_name=""
    local dev_env_file_name=""
    
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        env_file_name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "")
    fi
    
    if [[ -f "$PROJECT_ROOT/.devcontainer/.env" ]]; then
        dev_env_file_name=$(grep "^ENV_NAME=" "$PROJECT_ROOT/.devcontainer/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" || echo "")
    fi
    
    # Check for consistency issues
    if [[ -n "$env_file_name" && -n "$dev_env_file_name" && "$env_file_name" != "$dev_env_file_name" ]]; then
        log_error "ENV_NAME mismatch between .env ('$env_file_name') and .devcontainer/.env ('$dev_env_file_name')"
        issues=$((issues + 1))
    fi
    
    if [[ -n "${ENV_NAME:-}" && "$ENV_NAME" != "$name" ]]; then
        log_error "ENV_NAME environment variable ('$ENV_NAME') differs from configuration files ('$name')"
        issues=$((issues + 1))
    fi
    
    # Check required files exist
    local required_files=(
        ".devcontainer/docker-compose.yml"
        ".devcontainer/devcontainer.json"
        ".devcontainer/Dockerfile"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required file missing: $file"
            issues=$((issues + 1))
        fi
    done
    
    # Validate Docker Compose syntax
    if [[ -f "$PROJECT_ROOT/.devcontainer/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT/.devcontainer"
        if ! docker compose -p "$name" config >/dev/null 2>&1; then
            log_error "Docker Compose configuration is invalid"
            issues=$((issues + 1))
        else
            log_success "Docker Compose configuration is valid"
        fi
        cd - >/dev/null
    fi
    
    if [[ $issues -eq 0 ]]; then
        log_success "Configuration validation passed!"
    else
        log_error "Found $issues configuration issues"
        return 1
    fi
}

# Function to clean up Docker resources
clean_env_resources() {
    local env_name="$1"
    
    if [[ -z "$env_name" ]]; then
        log_error "Environment name cannot be empty"
        return 1
    fi
    
    log_info "Cleaning up Docker resources for environment '$env_name'..."
    
    # Stop and remove compose resources
    cd "$PROJECT_ROOT/.devcontainer"
    if docker compose -p "$env_name" ps -q >/dev/null 2>&1; then
        log_info "Stopping containers for $env_name..."
        docker compose -p "$env_name" down -v --remove-orphans 2>/dev/null || true
    fi
    cd - >/dev/null
    
    # Remove volumes with the env name prefix
    local volumes=$(docker volume ls --filter "name=${env_name}_" -q 2>/dev/null || true)
    if [[ -n "$volumes" ]]; then
        log_info "Removing volumes: $volumes"
        echo "$volumes" | xargs docker volume rm 2>/dev/null || true
    fi
    
    # Remove containers with the env name
    local containers=$(docker ps -a --filter "name=${env_name}" -q 2>/dev/null || true)
    if [[ -n "$containers" ]]; then
        log_info "Removing containers: $containers" 
        echo "$containers" | xargs docker rm -f 2>/dev/null || true
    fi
    
    log_success "Cleaned up Docker resources for '$env_name'"
}

# Function to migrate from old name to new name
migrate_env_name() {
    local old_name="$1"
    local new_name="$2"
    
    if [[ -z "$old_name" || -z "$new_name" ]]; then
        log_error "Both old and new environment names are required"
        return 1
    fi
    
    if [[ "$old_name" == "$new_name" ]]; then
        log_warning "Old and new names are identical. Nothing to migrate."
        return 0
    fi
    
    log_info "Migrating environment from '$old_name' to '$new_name'..."
    
    # Step 1: Set new name
    set_env_name "$new_name"
    
    # Step 2: Clean up old resources
    clean_env_resources "$old_name"
    
    # Step 3: Validate new configuration
    validate_config
    
    log_success "Migration completed! Next steps:"
    echo "  1. Build new containers: invoke up --name $new_name --rebuild"
    echo "  2. Test the new environment"
    echo "  3. Remove old Docker images if no longer needed: docker image prune"
}

# Main script logic
main() {
    local command="${1:-help}"
    
    case "$command" in
        "show")
            show_current
            ;;
        "set")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 set <environment_name>"
                exit 1
            fi
            set_env_name "$2"
            ;;
        "validate")
            validate_config
            ;;
        "clean")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 clean <environment_name>"
                exit 1
            fi
            clean_env_resources "$2"
            ;;
        "migrate")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 migrate <old_name> <new_name>"
                exit 1
            fi
            migrate_env_name "$2" "$3"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"







