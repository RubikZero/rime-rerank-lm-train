# Counts the rank jobs a composition actually schedules, before and after making
# the plan key order-insensitive.  The plugin prints "rank miss" (a job is
# requested) and "rank hit" (the cached ranking is applied) on stderr under
# RIME_LM_RANKER_DEBUG, so the server's stderr is captured to a file per build.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$OldDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$NewDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-jobcount",
  [string]$Context = "meeting at three pm, ",
  [string]$Cases = "beijing=word,beijingdaxue=sentence",
  [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")

foreach ($v in @(@{ Name = "old"; Dll = $OldDll }, @{ Name = "new"; Dll = $NewDll })) {
  Copy-Item -Force $v.Dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$($v.Name)"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmjc$($v.Name)"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $stderr = Join-Path $temp "server-stderr.txt"
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
    -WorkingDirectory $Runtime -RedirectStandardError $stderr -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$($v.Name)] server exited with $($server.ExitCode)" }
  & $LmClient $Context "beijingdaxue" 8000 150 > $null 2>&1   # warm-up / model load

  foreach ($case in $Cases.Split(",")) {
    $parts = $case.Split("=")
    $latency = & $LatencyClient $Context $parts[0] 150 20 8000 $Runs 2>&1
    $latency | Select-String "RESULT" | ForEach-Object { Write-Host "[$($v.Name) $($parts[1])] $($_.Line)" }
    & $LmClient $Context $parts[0] 8000 150 2>&1 |
      Out-File (Join-Path $OutDir "$($v.Name).menu.$($parts[1]).txt") -Encoding utf8
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()

  $lines = Get-Content $stderr -ErrorAction SilentlyContinue
  $miss = ($lines | Select-String "rank miss").Count
  $hit = ($lines | Select-String "rank hit").Count
  $skip = ($lines | Select-String "rank skip").Count
  Write-Host ("[{0}] rank miss={1} hit={2} skip={3}  (stderr lines={4})" -f $v.Name, $miss, $hit, $skip, $lines.Count)
}

Write-Host "`n== menu comparison (old vs new) =="
$fail = 0
foreach ($case in $Cases.Split(",")) {
  $label = $case.Split("=")[1]
  $report = python (Join-Path $OutDir "..\s1_compare_menus.py") `
    (Join-Path $OutDir "old.menu.$label.txt") (Join-Path $OutDir "new.menu.$label.txt")
  Write-Host ("{0}:" -f $label)
  $report | ForEach-Object { "  $_" }
  if (-not (($report -join "`n") -match "final menu identical: True")) { $fail = 1 }
}

Copy-Item -Force $NewDll (Join-Path $Runtime "rime.dll")
if ($fail -eq 0) { Write-Host "`nRESULT s1_jobcount=pass" } else { Write-Host "`nRESULT s1_jobcount=fail" }
