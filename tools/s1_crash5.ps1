# Repetition of the schema-switch loop: the crash is a race, so one surviving
# run proves nothing.  The old build (hook owned by the filter) reproduced it on
# the first attempt, so the loop is repeated several times for both builds: the
# old one should fail at least once, the fixed one must never fail.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$OldHookDll = "D:\Workspace\rime\third-party\lm-test\crash-candidate\rime.dll",
  [string]$NewHookDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-crash5",
  [int]$Repetitions = 3,
  [string]$Digits = "2,3,1,2,3,1,2,3,1,2",
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

function Invoke-Loop([string]$name, [string]$dll, [int]$rep) {
  $tag = "$name-$rep"
  Copy-Item -Force $dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$tag"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmsc$tag"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  $server.Refresh()
  if ($server.HasExited) { return "died at startup" }

  $current = Run-Step "meeting at three pm, " "ni" 0
  $switches = 0
  foreach ($digit in ($Digits.Split(",") | ForEach-Object { [int]$_ })) {
    $after = Run-Step "meeting at three pm, " "beijing" $digit
    if ($after -ne $current -and $after -notin @("?", "NO-CONNECTION")) { ++$switches }
    $current = $after
    Start-Sleep -Milliseconds 200
    $typed = Run-Step "meeting at three pm, " "beijingdaxue" 0
    $server.Refresh()
    if ($server.HasExited -or $typed -eq "NO-CONNECTION") {
      return "CRASHED after $switches switches (exit $($server.ExitCode))"
    }
  }
  $server.Kill(); $server.WaitForExit()
  return "survived $switches switches"
}

$summary = @()
foreach ($rep in 1..$Repetitions) {
  $old = Invoke-Loop "old-hook" $OldHookDll $rep
  $new = Invoke-Loop "new-hook" $NewHookDll $rep
  Write-Host ("rep {0}:  old-hook -> {1}" -f $rep, $old)
  Write-Host ("rep {0}:  new-hook -> {1}" -f $rep, $new)
  $summary += [pscustomobject]@{ Rep = $rep; Old = $old; New = $new }
}
$summary | Export-Csv (Join-Path $OutDir "crash5.csv") -NoTypeInformation -Encoding UTF8
$oldCrashes = ($summary | Where-Object { $_.Old -like "CRASHED*" }).Count
$newCrashes = ($summary | Where-Object { $_.New -like "CRASHED*" }).Count
Write-Host ("`nold-hook crashes: {0}/{1}   new-hook crashes: {2}/{1}" -f $oldCrashes, $Repetitions, $newCrashes)
Write-Host "RESULT s1_crash5=done"
