# ci/deploy/jenkins_deploy_windows.ps1 — One-click Jenkins deployment on Windows (REQ-8.12)
#
# Usage: .\jenkins_deploy_windows.ps1 -RepoUrl <git-url>
#
# Installs Java 17, Jenkins LTS, build dependencies via winget, and creates pipeline jobs.
# Requires Administrator privileges.

#Requires -RunAsAdministrator

param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"
$JenkinsUrl = "http://localhost:8080"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Jenkins Deploy (Windows): repo=$RepoUrl ==="

# ---------- Helper: Test if command exists ----------
function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---------- Install Java 17 ----------
function Install-Java {
    if (Test-Command "java") {
        $ver = & java -version 2>&1 | Select-String "17\."
        if ($ver) {
            Write-Host "Java 17 already installed."
            return
        }
    }
    Write-Host "Installing Java 17 (Adoptium)..."
    winget install --id EclipseAdoptium.Temurin.17.JDK --accept-package-agreements --accept-source-agreements --silent
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# ---------- Install Jenkins LTS ----------
function Install-Jenkins {
    $svc = Get-Service -Name "Jenkins" -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -eq "Running") {
        Write-Host "Jenkins already running."
        return
    }
    Write-Host "Installing Jenkins LTS..."
    winget install --id Jenkins.Jenkins --accept-package-agreements --accept-source-agreements --silent
    # Start service if not running
    Start-Service -Name "Jenkins" -ErrorAction SilentlyContinue
}

# ---------- Install build dependencies ----------
function Install-BuildDeps {
    Write-Host "Installing build dependencies..."

    if (-not (Test-Command "cmake")) {
        winget install --id Kitware.CMake --accept-package-agreements --accept-source-agreements --silent
    }

    if (-not (Test-Command "python3")) {
        winget install --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --silent
    }

    if (-not (Test-Command "git")) {
        winget install --id Git.Git --accept-package-agreements --accept-source-agreements --silent
    }

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# ---------- Wait for Jenkins to be ready ----------
function Wait-ForJenkins {
    Write-Host "Waiting for Jenkins to be ready..."
    $retries = 60
    while ($retries -gt 0) {
        try {
            $resp = Invoke-WebRequest -Uri "$JenkinsUrl/login" -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -eq 200) {
                Write-Host "Jenkins is ready."
                return
            }
        } catch {}
        $retries--
        Start-Sleep -Seconds 5
    }
    Write-Warning "Jenkins did not become ready. Job creation may fail."
}

# ---------- Create Jenkins jobs via CLI ----------
function New-JenkinsJobs {
    $cliJar = "$env:TEMP\jenkins-cli.jar"

    if (-not (Test-Path $cliJar)) {
        try {
            Invoke-WebRequest -Uri "$JenkinsUrl/jnlpJars/jenkins-cli.jar" -OutFile $cliJar -UseBasicParsing
        } catch {
            Write-Warning "Could not download Jenkins CLI. Create jobs manually."
            return
        }
    }

    $stubsXml = Join-Path $ScriptDir "job_config_stubs.xml"
    $smokeXml = Join-Path $ScriptDir "job_config_smoke.xml"

    Write-Host "Creating/updating stubs-build job..."
    try {
        Get-Content $stubsXml | & java -jar $cliJar -s $JenkinsUrl create-job stubs-build
    } catch {
        Get-Content $stubsXml | & java -jar $cliJar -s $JenkinsUrl update-job stubs-build
    }

    Write-Host "Creating/updating smoke-tests job..."
    try {
        Get-Content $smokeXml | & java -jar $cliJar -s $JenkinsUrl create-job smoke-tests
    } catch {
        Get-Content $smokeXml | & java -jar $cliJar -s $JenkinsUrl update-job smoke-tests
    }

    Write-Host "Jobs created/updated successfully."
}

# ---------- Main ----------
Install-Java
Install-Jenkins
Install-BuildDeps
Wait-ForJenkins
New-JenkinsJobs

Write-Host "=== Jenkins deployment complete ==="
Write-Host "Access Jenkins at: $JenkinsUrl"
$pwdFile = "C:\ProgramData\Jenkins\.jenkins\secrets\initialAdminPassword"
if (Test-Path $pwdFile) {
    Write-Host "Initial admin password: $(Get-Content $pwdFile)"
}
