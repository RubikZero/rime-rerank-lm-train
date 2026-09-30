# Schema-switch crash reproduction, with the switch *verified*.
#
# The first attempt never proved that a switch happened, so this version reads
# status.schema_id out of the client's own output for every step and only counts
# a switch that really changed the schema.  Each variant runs the same loop:
#
#   warm up -> for each digit: switch, type, type again -> is the server alive?
#
# Run against the build with the warm hook and the pre-change build (no hook).
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$HookDll = "D:\Workspace\rime\third-party\lm-test\crash-candidate\rime.dll",
  [string]$NoHookDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-crash2",
  # The isolated runtime lists luna_pinyin_simp(1), luna_quanpin(2),
  # luna_pinyin_fluency(3) in schema_list; 4 is out of range on purpose.
  [string]$Digits = "2,3,1,2,3,1",
  [int]$KeyDelayMs = 100
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")

function Run-Step([string]$context, [string]$keys, [int]$digit) {
  $out = & $ClientExe $context $keys $KeyDelayMs 120 3000 1 $digit 2>&1
  $line = ($out | Select-String "run 1: schema=" | Select-Object -Last 1).Line
  if (-not $line) { return "?" }
  if ($line -match "schema=(\S+)") { return $Matches[1] }
  return "?"
}

function Invoke-Variant([string]$name, [string]$dll) {
  Copy-Item -Force $dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmcrash2$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { return @{ Name = $name; Died = "during startup"; Steps = @() } }

  $steps = @()
  $current = Run-Step "meeting at three pm, " "ni" 0
  $steps += "warm-up -> schema=$current"
  if ($server.HasExited) { return @{ Name = $name; Died = "after warm-up"; Steps = $steps } }

  foreach ($digit in ($Digits.Split(",") | ForEach-Object { [int]$_ })) {
    $before = $current
    $after = Run-Step "meeting at three pm, " "beijing" $digit
    $current = $after
    $switched = ($after -ne $before -and $after -ne "?")
    $steps += ("digit {0}: {1} -> {2}{3}" -f $digit, $before, $after, $(if ($switched) { " (switched)" } else { " (unchanged)" }))
    Start-Sleep -Milliseconds 300
    # A keystroke after the switch: this is when a stale filter callback would run.
    $typed = Run-Step "meeting at three pm, " "beijingdaxue" 0
    $steps += ("  typed after switch -> schema=$typed")
    $server.Refresh()
    if ($server.HasExited) {
      return @{ Name = $name; Died = "after digit $digit (switched=$switched, exit $($server.ExitCode))"; Steps = $steps }
    }
  }
  $server.Kill(); $server.WaitForExit()
  return @{ Name = $name; Died = ""; Steps = $steps }
}

$results = @()
$results += Invoke-Variant "with-hook" $HookDll
$results += Invoke-Variant "without-hook" $NoHookDll

foreach ($r in $results) {
  "`n=== {0} ===" -f $r.Name
  $r.Steps | ForEach-Object { "   $_" }
  if ($r.Died) { "   DIED: $($r.Died)" } else { "   survived the whole loop" }
}
"`nRESULT s1_crash2=done"
