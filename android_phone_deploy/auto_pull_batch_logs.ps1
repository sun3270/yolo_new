param(
    [switch]$WaitForBatch,
    [string]$OutputDir = (Join-Path $PSScriptRoot "analysis_from_phone_logs")
)

$ErrorActionPreference = "Stop"

function Find-Adb {
    $candidates = @()
    if ($env:ANDROID_HOME) {
        $candidates += (Join-Path $env:ANDROID_HOME "platform-tools\adb.exe")
    }
    if ($env:ANDROID_SDK_ROOT) {
        $candidates += (Join-Path $env:ANDROID_SDK_ROOT "platform-tools\adb.exe")
    }
    $candidates += (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe")
    $candidates += "adb"

    foreach ($candidate in $candidates) {
        try {
            & $candidate version *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }
    throw "adb was not found. Install Android Studio platform-tools or add adb to PATH."
}

function Wait-ForBatchComplete {
    param([string]$AdbPath)

    & $AdbPath logcat -c *> $null
    Write-Host "Waiting for phone batch completion. Tap Batch 100 in the app."

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $AdbPath
    $startInfo.Arguments = "logcat -v time -s YoloCoffeeBatch:I *:S"
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()

    try {
        while (-not $process.HasExited) {
            $line = $process.StandardOutput.ReadLine()
            if ($null -eq $line) {
                Start-Sleep -Milliseconds 100
                continue
            }
            Write-Host $line
            if ($line -match "BATCH_COMPLETE") {
                return
            }
            if ($line -match "BATCH_FAILED") {
                throw "The phone reported that the batch failed. Check Android Studio logcat for details."
            }
        }
    } finally {
        if (-not $process.HasExited) {
            $process.Kill()
        }
    }
}

$adb = Find-Adb
$remoteDir = "/sdcard/Android/data/com.example.yolocoffee/files/detection_logs"
$destDir = Join-Path $OutputDir "detection_logs"
$files = @(
    "runs.csv",
    "detections.csv",
    "events.jsonl",
    "benchmark_runs.csv",
    "benchmark_matches.csv",
    "batch_summary.csv"
)

New-Item -ItemType Directory -Force -Path $destDir *> $null

Write-Host "Waiting for an Android device..."
& $adb wait-for-device

if ($WaitForBatch) {
    Wait-ForBatchComplete -AdbPath $adb
}

foreach ($file in $files) {
    $remoteFile = "$remoteDir/$file"
    $localFile = Join-Path $destDir $file
    & $adb pull $remoteFile $localFile
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not pull $remoteFile"
    }
}

Write-Host "Logs pulled to $destDir"
