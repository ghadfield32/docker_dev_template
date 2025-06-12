#!/usr/bin/env pwsh
# Port Manager Script for Docker Dev Template
# Prevents and resolves port conflicts before container startup

param(
    [string]$Action = "check",  # check, fix, change-port, clean
    [int]$NewPort = 8891
)

Write-Host "=== Docker Dev Template Port Manager ===" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Yellow

function Test-PortAvailable {
    param([int]$Port)
    
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Get-PortProcesses {
    param([int]$Port)
    
    $processes = @()
    $netstatOutput = netstat -ano | findstr ":$Port"
    
    if ($netstatOutput) {
        foreach ($line in $netstatOutput) {
            $parts = $line -split '\s+' | Where-Object { $_ -ne '' }
            if ($parts.Length -ge 5) {
                $processId = $parts[-1]
                try {
                    $process = Get-Process -Id $processId -ErrorAction Stop
                    $processes += [PSCustomObject]@{
                        PID = $processId
                        Name = $process.ProcessName
                        Path = $process.Path
                        Line = $line.Trim()
                    }
                } catch {
                    $processes += [PSCustomObject]@{
                        PID = $processId
                        Name = "Unknown"
                        Path = "N/A"
                        Line = $line.Trim()
                    }
                }
            }
        }
    }
    
    return $processes
}

function Show-PortStatus {
    param([int]$Port)
    
    Write-Host "`nChecking port $Port..." -ForegroundColor Yellow
    
    if (Test-PortAvailable -Port $Port) {
        Write-Host "✅ Port $Port is available" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Port $Port is in use" -ForegroundColor Red
        
        $processes = Get-PortProcesses -Port $Port
        if ($processes.Count -gt 0) {
            Write-Host "Processes using port $Port:" -ForegroundColor Yellow
            foreach ($proc in $processes) {
                $pidValue = $proc.PID
                $nameValue = $proc.Name
                $pathValue = $proc.Path
                Write-Host "  PID: $pidValue | Name: $nameValue | Path: $pathValue" -ForegroundColor Red
            }
        }
        return $false
    }
}

function Stop-DockerContainers {
    Write-Host "`nStopping all Docker containers..." -ForegroundColor Yellow
    
    $containers = docker ps -q
    if ($containers) {
        docker stop $containers 2>$null
        Write-Host "✅ Stopped Docker containers" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  No running containers found" -ForegroundColor Blue
    }
    
    # Clean up
    docker container prune -f 2>$null
    docker network prune -f 2>$null
    Write-Host "✅ Cleaned up Docker resources" -ForegroundColor Green
}

function Update-PortInConfig {
    param([int]$NewPort)
    
    Write-Host "`nUpdating port configuration to $NewPort..." -ForegroundColor Yellow
    
    # Update dev.env
    if (Test-Path "dev.env") {
        $content = Get-Content "dev.env"
        $content = $content -replace 'HOST_JUPYTER_PORT=\d+', "HOST_JUPYTER_PORT=$NewPort"
        $content | Set-Content "dev.env"
        Write-Host "✅ Updated dev.env" -ForegroundColor Green
    }
    
    # Update devcontainer.env if it exists
    if (Test-Path ".devcontainer/devcontainer.env") {
        $content = Get-Content ".devcontainer/devcontainer.env"
        $content = $content -replace 'HOST_JUPYTER_PORT=\d+', "HOST_JUPYTER_PORT=$NewPort"
        $content | Set-Content ".devcontainer/devcontainer.env"
        Write-Host "✅ Updated .devcontainer/devcontainer.env" -ForegroundColor Green
    }
}

function Find-NextAvailablePort {
    param([int]$StartPort = 8890)
    
    for ($port = $StartPort; $port -lt ($StartPort + 100); $port++) {
        if (Test-PortAvailable -Port $port) {
            return $port
        }
    }
    return $null
}

# Main execution
switch ($Action.ToLower()) {
    "check" {
        Write-Host "`n🔍 Checking current configuration..." -ForegroundColor Cyan
        
        # Read current port from dev.env
        $currentPort = 8890
        if (Test-Path "dev.env") {
            $envContent = Get-Content "dev.env"
            $portLine = $envContent | Where-Object { $_ -match 'HOST_JUPYTER_PORT=(\d+)' }
            if ($portLine) {
                $currentPort = [int]$matches[1]
            }
        }
        
        Write-Host "Current configured port: $currentPort" -ForegroundColor Blue
        Show-PortStatus -Port $currentPort
        
        # Show Docker status
        Write-Host "`n🐳 Docker Status:" -ForegroundColor Cyan
        $containers = docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        Write-Host $containers
        
        # Show alternative ports
        Write-Host "`n🔄 Alternative ports:" -ForegroundColor Cyan
        $alternatives = @(8891, 8892, 8893, 8894, 8895)
        foreach ($alt in $alternatives) {
            Show-PortStatus -Port $alt | Out-Null
        }
    }
    
    "fix" {
        Write-Host "`n🔧 Auto-fixing port conflicts..." -ForegroundColor Cyan
        
        # Read current port
        $currentPort = 8890
        if (Test-Path "dev.env") {
            $envContent = Get-Content "dev.env"
            $portLine = $envContent | Where-Object { $_ -match 'HOST_JUPYTER_PORT=(\d+)' }
            if ($portLine) {
                $currentPort = [int]$matches[1]
            }
        }
        
        if (Show-PortStatus -Port $currentPort) {
            Write-Host "✅ Port $currentPort is already available. No action needed." -ForegroundColor Green
        } else {
            Write-Host "🔍 Finding alternative port..." -ForegroundColor Yellow
            $newPort = Find-NextAvailablePort -StartPort ($currentPort + 1)
            
            if ($newPort) {
                Write-Host "✅ Found available port: $newPort" -ForegroundColor Green
                Update-PortInConfig -NewPort $newPort
                Write-Host "🎉 Configuration updated! Try starting your DevContainer now." -ForegroundColor Green
            } else {
                Write-Host "❌ Could not find available port. Try stopping Docker containers first." -ForegroundColor Red
                Stop-DockerContainers
            }
        }
    }
    
    "change-port" {
        if (Test-PortAvailable -Port $NewPort) {
            Update-PortInConfig -NewPort $NewPort
            Write-Host "✅ Port changed to $NewPort successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ Port $NewPort is not available" -ForegroundColor Red
            Show-PortStatus -Port $NewPort | Out-Null
        }
    }
    
    "clean" {
        Write-Host "`n🧹 Cleaning Docker resources..." -ForegroundColor Cyan
        Stop-DockerContainers
        
        # Also clean up images and volumes if needed
        $choice = Read-Host "Clean unused images and volumes too? [y/N]"
        if ($choice -eq 'y' -or $choice -eq 'Y') {
            docker image prune -f 2>$null
            docker volume prune -f 2>$null
            Write-Host "✅ Cleaned unused images and volumes" -ForegroundColor Green
        }
    }
    
    default {
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Write-Host "Available actions: check, fix, change-port, clean" -ForegroundColor Yellow
        Write-Host "Examples:" -ForegroundColor Blue
        Write-Host "  .\port-manager.ps1 check" -ForegroundColor White
        Write-Host "  .\port-manager.ps1 fix" -ForegroundColor White
        Write-Host "  .\port-manager.ps1 change-port -NewPort 8891" -ForegroundColor White
        Write-Host "  .\port-manager.ps1 clean" -ForegroundColor White
    }
}

Write-Host "`n=== Port Manager Complete ===" -ForegroundColor Cyan 