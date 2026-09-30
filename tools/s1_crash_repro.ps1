# Reproduces the production crash seen while switching schemas.
#
# The suspicion: the warm hook connects to the context's notifiers in the
# filter's constructor with a lambda capturing `this`, but a schema switch
# recreates the filters while the context lives on -- the next update then calls
# into a destroyed filter.  This drives schema switches plus keystrokes in the
# isolated runtime, first with the current build (hook present) and then with the
# pre-change build (no hook), and reports whether the server survived.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$HookDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$NoHookDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-crash",
  [string]$Switches = "2,1,2,1,2,1",
  [int]$KeyDelayMs = 120
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")

function Invoke-Variant([string]$name, [string]$dll) {
  Copy-Item -Force $dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmcrash$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$name] server exited during startup ($($server.ExitCode))" }
  "`n=== {0}: rime.dll {1} ===" -f $name, (Get-FileHash $dll -Algorithm SHA256).Hash.Substring(0, 16)

  # Warm-up so the model is loaded before the switches.
  & $ClientExe "meeting at three pm, " "ni" $KeyDelayMs 120 3000 1 0 > $null 2>&1
  if ($server.HasExited) { "   died during warm-up"; return $false }

  foreach ($digit in $Switches.Split(",")) {
    # The client switches schema inside its own session (and ends that session);
    # a second call then types into the new schema's engine.
    & $ClientExe "meeting at three pm, " "nihao" $KeyDelayMs 120 3000 1 $digit > $null 2>&1
    Start-Sleep -Milliseconds 400
    & $ClientExe "meeting at three pm, " "beijing" $KeyDelayMs 120 3000 1 0 > $null 2>&1
    $server.Refresh()
    if ($server.HasExited) {
      "   CRASHED after switching with digit $digit (exit code $($server.ExitCode))"
      return $false
    }
  }
  "   survived all switches: $Switches"
  $server.Kill(); $server.WaitForExit()
  return $true
}

$hookOk = Invoke-Variant "with-hook" $HookDll
$noHookOk = Invoke-Variant "without-hook" $NoHookDll
"`n=== summary ==="
"   current build (warm hook): {0}" -f $(if ($hookOk) { "survived" } else { "CRASHED" })
"   pre-change build:          {0}" -f $(if ($noHookOk) { "survived" } else { "CRASHED" })
"RESULT s1_crash_repro=done"
