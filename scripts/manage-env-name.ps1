# Environment Name Management Script for Windows
# =============================================
# This script helps you easily change and manage the environment name
# across all configuration files in your Docker development setup.

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Name1,
    
    [Parameter(Position=2)]
    [string]$Name2
)

# Color functions for output
function Write-Info($message) { Write-Host "ℹ️  $message" -ForegroundColor Cyan }
function Write-Success($message) { Write-Host "✅ $message" -ForegroundColor Green }
function Write-Warning($message) { Write-Host "⚠️  $message" -ForegroundColor Yellow }
function Write-Error($message) { Write-Host "❌ $message" -ForegroundColor Red }

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Function to show usage
function Show-Usage {
    Write-Host "Environment Name Management Script for Windows"
    Write-Host "============================================="
    Write-Host ""
    Write-Host "This script helps you manage the ENV_NAME across your Docker development setup."
    Write-Host ""
    Write-Host "USAGE:"
    Write-Host "    .\scripts\manage-env-name.ps1 [command] [options]"
    Write-Host ""
    Write-Host "COMMANDS:"
    Write-Host "    show                    Show current environment name from all sources"
    Write-Host "    set [name]             Set new environment name in .env file"
    Write-Host "    validate               Validate current configuration consistency"
    Write-Host "    clean [old_name]       Clean up Docker resources for old environment name"
    Write-Host "    migrate [old] [new]    Migrate from old name to new name (full process)"
    Write-Host "    help                   Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:"
    Write-Host "    .\scripts\manage-env-name.ps1 show                           # Show current settings"
    Write-Host "    .\scripts\manage-env-name.ps1 set my_ml_project             # Set ENV_NAME to my_ml_project"
    Write-Host "    .\scripts\manage-env-name.ps1 validate                      # Check configuration consistency"
    Write-Host "    .\scripts\manage-env-name.ps1 clean old_project_name        # Clean up old Docker resources"
    Write-Host "    .\scripts\manage-env-name.ps1 migrate docker_dev_template my_project  # Full migration"
    Write-Host ""
    Write-Host "NOTES:"
    Write-Host "    - The .env file is the primary source of truth for ENV_NAME"
    Write-Host "    - All Docker resources (containers, volumes, networks) are prefixed with ENV_NAME"
    Write-Host "    - Changing ENV_NAME requires rebuilding containers to take effect"
    Write-Host "    - Use 'clean' command to remove old Docker resources after changing names"
}

# Function to get current env name from various sources
function Get-CurrentEnvName {
    $source = ""
    $name = ""
    
    # Check .env file first (primary source)
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $content = Get-Content $envFile
        $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
        if ($envLine) {
            $name = ($envLine -split "=", 2)[1].Trim('"', "'")
            $source = ".env"
        }
    }
    
    # Check .devcontainer/.env as fallback
    if (-not $name) {
        $devEnvFile = Join-Path $ProjectRoot ".devcontainer\.env"
        if (Test-Path $devEnvFile) {
            $content = Get-Content $devEnvFile
            $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
            if ($envLine) {
                $name = ($envLine -split "=", 2)[1].Trim('"', "'")
                $source = ".devcontainer\.env"
            }
        }
    }
    
    # Check environment variable
    if ($env:ENV_NAME) {
        if ($name -and $name -ne $env:ENV_NAME) {
            Write-Warning "Environment variable ENV_NAME='$($env:ENV_NAME)' differs from file value '$name'"
        }
        $name = $env:ENV_NAME
        $source = "environment variable"
    }
    
    # Default fallback
    if (-not $name) {
        $name = "docker_dev_template"
        $source = "default"
    }
    
    return @{ Name = $name; Source = $source }
}

# Function to show current environment name
function Show-Current {
    Write-Info "Current Environment Name Configuration:"
    Write-Host "======================================"
    
    $result = Get-CurrentEnvName
    $name = $result.Name
    $source = $result.Source
    
    Write-Host "Active ENV_NAME: $name (from $source)"
    Write-Host ""
    
    # Show all sources
    Write-Host "Configuration Sources:"
    Write-Host "----------------------"
    
    # .env file
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $content = Get-Content $envFile
        $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
        $envName = if ($envLine) { ($envLine -split "=", 2)[1].Trim('"', "'") } else { "<not set>" }
        Write-Host ".env file:            $envName"
    } else {
        Write-Host ".env file:            <file not found>"
    }
    
    # .devcontainer/.env file  
    $devEnvFile = Join-Path $ProjectRoot ".devcontainer\.env"
    if (Test-Path $devEnvFile) {
        $content = Get-Content $devEnvFile
        $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
        $devEnvName = if ($envLine) { ($envLine -split "=", 2)[1].Trim('"', "'") } else { "<not set>" }
        Write-Host ".devcontainer\.env:   $devEnvName"
    } else {
        Write-Host ".devcontainer\.env:   <file not found>"
    }
    
    # Environment variable
    Write-Host "ENV_NAME variable:    $(if ($env:ENV_NAME) { $env:ENV_NAME } else { '<not set>' })"
    
    Write-Host ""
    Write-Host "Docker Resources (if any):"
    Write-Host "--------------------------"
    
    # Check for Docker resources with this name
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Host "Containers:"
        try {
            $containers = docker ps -a --filter "name=${name}" --format "{{.Names}} ({{.Status}})" 2>$null
            if ($containers) { $containers | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  None found" }
        } catch {
            Write-Host "  None found"
        }
        
        Write-Host "Volumes:"  
        try {
            $volumes = docker volume ls --filter "name=${name}" --format "{{.Name}}" 2>$null
            if ($volumes) { $volumes | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  None found" }
        } catch {
            Write-Host "  None found"
        }
        
        Write-Host "Compose Projects:"
        try {
            $projects = docker compose ls 2>$null | Select-String $name
            if ($projects) { $projects | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  None found" }
        } catch {
            Write-Host "  None found"
        }
    } else {
        Write-Host "  Docker not available"
    }
}

# Function to set new environment name
function Set-EnvName {
    param([string]$NewName)
    
    if (-not $NewName) {
        Write-Error "Environment name cannot be empty"
        return $false
    }
    
    # Validate name (should be suitable for Docker resource naming)
    if ($NewName -notmatch "^[a-zA-Z][a-zA-Z0-9_-]*$") {
        Write-Error "Invalid environment name. Must start with letter and contain only letters, numbers, underscore, and hyphen."
        return $false
    }
    
    $current = Get-CurrentEnvName
    $currentName = $current.Name
    
    if ($currentName -eq $NewName) {
        Write-Warning "Environment name is already set to '$NewName'"
        return $true
    }
    
    Write-Info "Setting environment name from '$currentName' to '$NewName'"
    
    # Create or update .env file
    $envFile = Join-Path $ProjectRoot ".env"
    
    if (Test-Path $envFile) {
        # Update existing file
        $content = Get-Content $envFile
        $hasEnvName = $content | Where-Object { $_ -match "^ENV_NAME=" }
        
        if ($hasEnvName) {
            # Replace existing ENV_NAME line
            $newContent = $content | ForEach-Object {
                if ($_ -match "^ENV_NAME=") {
                    "ENV_NAME=$NewName"
                } else {
                    $_
                }
            }
            $newContent | Set-Content $envFile
        } else {
            # Add ENV_NAME to existing file
            Add-Content $envFile "ENV_NAME=$NewName"
        }
    } else {
        # Create new .env file from template
        $templateFile = Join-Path $ProjectRoot ".devcontainer\.env.template"
        if (Test-Path $templateFile) {
            Copy-Item $templateFile $envFile
            $content = Get-Content $envFile
            $newContent = $content | ForEach-Object {
                if ($_ -match "^ENV_NAME=") {
                    "ENV_NAME=$NewName"
                } else {
                    $_
                }
            }
            $newContent | Set-Content $envFile
        } else {
            # Create minimal .env file
            "ENV_NAME=$NewName" | Set-Content $envFile
        }
    }
    
    Write-Success "Environment name set to '$NewName' in $envFile"
    
    # Update current session
    $env:ENV_NAME = $NewName
    
    Write-Warning "You will need to rebuild containers for changes to take effect:"
    Write-Host "  cd .devcontainer"
    Write-Host "  docker compose -p $NewName down"  
    Write-Host "  docker compose -p $NewName build --no-cache"
    Write-Host "  docker compose -p $NewName up -d"
    Write-Host ""
    Write-Host "Or use: invoke down --name $currentName && invoke up --name $NewName --rebuild"
    
    return $true
}

# Function to validate configuration consistency
function Test-Config {
    Write-Info "Validating environment configuration..."
    
    $result = Get-CurrentEnvName
    $name = $result.Name
    $issues = 0
    
    Write-Host "Primary ENV_NAME: $name"
    Write-Host ""
    
    # Check file consistency
    $envFileName = ""
    $devEnvFileName = ""
    
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        $content = Get-Content $envFile
        $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
        if ($envLine) {
            $envFileName = ($envLine -split "=", 2)[1].Trim('"', "'")
        }
    }
    
    $devEnvFile = Join-Path $ProjectRoot ".devcontainer\.env"
    if (Test-Path $devEnvFile) {
        $content = Get-Content $devEnvFile
        $envLine = $content | Where-Object { $_ -match "^ENV_NAME=" }
        if ($envLine) {
            $devEnvFileName = ($envLine -split "=", 2)[1].Trim('"', "'")
        }
    }
    
    # Check for consistency issues
    if ($envFileName -and $devEnvFileName -and $envFileName -ne $devEnvFileName) {
        Write-Error "ENV_NAME mismatch between .env ('$envFileName') and .devcontainer\.env ('$devEnvFileName')"
        $issues++
    }
    
    if ($env:ENV_NAME -and $env:ENV_NAME -ne $name) {
        Write-Error "ENV_NAME environment variable ('$($env:ENV_NAME)') differs from configuration files ('$name')"
        $issues++
    }
    
    # Check required files exist
    $requiredFiles = @(
        ".devcontainer\docker-compose.yml",
        ".devcontainer\devcontainer.json",
        ".devcontainer\Dockerfile"
    )
    
    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $ProjectRoot $file
        if (-not (Test-Path $filePath)) {
            Write-Error "Required file missing: $file"
            $issues++
        }
    }
    
    # Validate Docker Compose syntax
    $composeFile = Join-Path $ProjectRoot ".devcontainer\docker-compose.yml"
    if (Test-Path $composeFile) {
        Push-Location (Join-Path $ProjectRoot ".devcontainer")
        try {
            $null = docker compose -p $name config 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Docker Compose configuration is valid"
            } else {
                Write-Error "Docker Compose configuration is invalid"
                $issues++
            }
        } catch {
            Write-Error "Docker Compose configuration is invalid"
            $issues++
        }
        Pop-Location
    }
    
    if ($issues -eq 0) {
        Write-Success "Configuration validation passed!"
        return $true
    } else {
        Write-Error "Found $issues configuration issues"
        return $false
    }
}

# Function to clean up Docker resources
function Remove-EnvResources {
    param([string]$EnvName)
    
    if (-not $EnvName) {
        Write-Error "Environment name cannot be empty"
        return $false
    }
    
    Write-Info "Cleaning up Docker resources for environment '$EnvName'..."
    
    # Stop and remove compose resources
    Push-Location (Join-Path $ProjectRoot ".devcontainer")
    try {
        $null = docker compose -p $EnvName ps -q 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Stopping containers for $EnvName..."
            docker compose -p $EnvName down -v --remove-orphans 2>$null
        }
    } catch {
        # Ignore errors
    }
    Pop-Location
    
    # Remove volumes with the env name prefix
    try {
        $volumes = docker volume ls --filter "name=${EnvName}_" -q 2>$null
        if ($volumes) {
            Write-Info "Removing volumes: $volumes"
            $volumes | ForEach-Object { docker volume rm $_ 2>$null }
        }
    } catch {
        # Ignore errors
    }
    
    # Remove containers with the env name
    try {
        $containers = docker ps -a --filter "name=${EnvName}" -q 2>$null
        if ($containers) {
            Write-Info "Removing containers: $containers" 
            $containers | ForEach-Object { docker rm -f $_ 2>$null }
        }
    } catch {
        # Ignore errors
    }
    
    Write-Success "Cleaned up Docker resources for '$EnvName'"
    return $true
}

# Function to migrate from old name to new name
function Move-EnvName {
    param([string]$OldName, [string]$NewName)
    
    if (-not $OldName -or -not $NewName) {
        Write-Error "Both old and new environment names are required"
        return $false
    }
    
    if ($OldName -eq $NewName) {
        Write-Warning "Old and new names are identical. Nothing to migrate."
        return $true
    }
    
    Write-Info "Migrating environment from '$OldName' to '$NewName'..."
    
    # Step 1: Set new name
    if (-not (Set-EnvName $NewName)) {
        return $false
    }
    
    # Step 2: Clean up old resources
    Remove-EnvResources $OldName
    
    # Step 3: Validate new configuration
    Test-Config
    
    Write-Success "Migration completed! Next steps:"
    Write-Host "  1. Build new containers: invoke up --name $NewName --rebuild"
    Write-Host "  2. Test the new environment"
    Write-Host "  3. Remove old Docker images if no longer needed: docker image prune"
    
    return $true
}

# Main script logic
switch ($Command.ToLower()) {
    "show" {
        Show-Current
    }
    "set" {
        if (-not $Name1) {
            Write-Error "Usage: .\scripts\manage-env-name.ps1 set <environment_name>"
            exit 1
        }
        Set-EnvName $Name1
    }
    "validate" {
        Test-Config
    }
    "clean" {
        if (-not $Name1) {
            Write-Error "Usage: .\scripts\manage-env-name.ps1 clean <environment_name>"
            exit 1
        }
        Remove-EnvResources $Name1
    }
    "migrate" {
        if (-not $Name1 -or -not $Name2) {
            Write-Error "Usage: .\scripts\manage-env-name.ps1 migrate <old_name> <new_name>"
            exit 1
        }
        Move-EnvName $Name1 $Name2
    }
    "help" {
        Show-Usage
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}
