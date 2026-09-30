# Schema-switch crash reproduction, verified.
#
# Every step reads status.schema_id out of the client's own output, so a step is
# only counted as a switch when the schema really changed.  Three builds:
#
#   old-hook   the build whose warm hook captured the filter (crashed in
#              production with 0xC0000005 reading 0xFFFFFFFFFFFFFFFF)
#   new-hook   the process-lifetime hook
#   no-hook    the pre-change build, as a control
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$OldHookDll = "D:\Workspace\rime\third-party\lm-test\crash-candidate\rime.dll",
  [string]$NewHookDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$NoHookDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-crash3",
  [string]$Digits = "2,3,1,2,3,1",
  [int]$KeyDelayMs = 100
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")

function Run-Step([string]$context, [string]$keys, [int]$digit) {
  $out = & $ClientExe $context $keys $KeyDelayMs 120 3000 1 $digit 2>&1
  $joined = ($out | Out-String)
  if ($joined -match "cannot connect") { return "NO-CONNECTION" }
  if ($joined -match "schema=(\S+)") { return $Matches[1] }
  return "?"
}

function Invoke-Variant([string]$name, [string]$dll) {
  $log = @()
  $log += ("=== {0}: rime.dll {1} ===" -f $name, (Get-FileHash $dll -Algorithm SHA256).Hash.Substring(0, 16))
  Copy-Item -Force $dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmsc$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  $server.Refresh()
  if ($server.HasExited) {
    $log += "   DIED DURING STARTUP (exit $($server.ExitCode))"
    return @{ Name = $name; Died = "startup"; Log = $log }
  }

  $current = Run-Step "meeting at three pm, " "ni" 0
  $log += "   warm-up: schema=$current"
  if ($server.HasExited -or $current -eq "NO-CONNECTION") {
    $log += "   DIED AFTER WARM-UP"
    return @{ Name = $name; Died = "warm-up"; Log = $log }
  }

  foreach ($digit in ($Digits.Split(",") | ForEach-Object { [int]$_ })) {
    $before = $current
    $after = Run-Step "meeting at three pm, " "beijing" $digit
    $switched = ($after -ne $before -and $after -notin @("?", "NO-CONNECTION"))
    $current = $after
    $log += ("   digit {0}: {1} -> {2}{3}" -f $digit, $before, $after, $(if ($switched) { " SWITCHED" } else { "" }))
    Start-Sleep -Milliseconds 300
    # A keystroke after the switch is where a stale callback would run.
    $typed = Run-Step "meeting at three pm, " "beijingdaxue" 0
    $log += "     typed -> schema=$typed"
    $server.Refresh()
    if ($server.HasExited -or $typed -eq "NO-CONNECTION") {
      $log += ("   DIED after digit {0} (switched={1}, exit {2})" -f $digit, $switched, $server.ExitCode)
      return @{ Name = $name; Died = "digit $digit"; Log = $log }
    }
  }
  $server.Kill(); $server.WaitForExit()
  $log += "   survived the whole loop"
  return @{ Name = $name; Died = ""; Log = $log }
}

$results = @()
$results += Invoke-Variant "old-hook" $OldHookDll
$results += Invoke-Variant "new-hook" $NewHookDll
$results += Invoke-Variant "no-hook" $NoHookDll

$text = ($results | ForEach-Object { $_.Log + "" }) -join "`n"
$text | Out-File (Join-Path $OutDir "crash3.log") -Encoding utf8
Write-Host $text
"`n=== summary ==="
$results | ForEach-Object { "   {0,-10} {1}" -f $_.Name, $(if ($_.Died) { "DIED ($($_.Died))" } else { "survived" }) }
"RESULT s1_crash3=done"
