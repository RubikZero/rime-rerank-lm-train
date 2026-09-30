# Accounts for the per-case cost of a rerank job: how many candidate rows the
# plugin scores, and how many sequence positions each variant has to push
# through the model.
#
#   uncached: rows x (window + candidate tokens)
#   cached:   one prefix encode of (window - tail) positions + rows x (tail + candidate tokens + 2)
#
# Each case runs against its own TEMP directory so the plugin log of one case
# cannot be confused with another.  The latency client is used because it
# reproduces the measurement conditions of the latency table (polling client).
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-jobcost",
  [string]$Context = "Please write the meeting notes for the quarterly review in the shared document before the end of the day, and send a copy to the team so that everyone can comment on the numbers we discussed yesterday. ",
  [string]$Cases = "ni=char,beijing=word,beijingdaxue=sentence",
  [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")

$summary = @()
foreach ($case in $Cases.Split(",")) {
  $parts = $case.Split("=")
  $keys, $kind = $parts[0], $parts[1]
  $temp = Join-Path $OutDir "tmp-$kind"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp
  $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmcost$kind"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"

  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } |
    ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
    -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$kind] server exited with $($server.ExitCode)" }

  & $LatencyClient $Context $keys 150 20 8000 $Runs > $null 2>&1
  $result = (& $LatencyClient $Context $keys 150 20 8000 $Runs 2>&1 | Select-String "RESULT" | Select-Object -Last 1).Line

  Start-Sleep -Seconds 1
  $server.Kill()
  $server.WaitForExit()

  $jobs = @()
  Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue | ForEach-Object {
    $jobs += Select-String -Path $_.FullName -Pattern "cache scored" | ForEach-Object {
      if ($_.Line -match "scored (\d+) candidates window=(\d+) tail=(\d+) prefix=(\d+)") {
        [pscustomobject]@{ Rows = [int]$Matches[1]; Window = [int]$Matches[2]; Tail = [int]$Matches[3]; Prefix = [int]$Matches[4] }
      }
    }
  }
  $full = $jobs | Where-Object { $_.Rows -ge 20 }
  $window = ($jobs | Select-Object -First 1).Window
  $tail = ($jobs | Select-Object -First 1).Tail
  $rows_total = ($jobs | Measure-Object Rows -Sum).Sum
  $rows_full = ($full | Measure-Object Rows -Sum).Sum
  Write-Host ("[{0,-8}] {1}" -f $kind, $result)
  Write-Host ("           jobs={0} rows_total={1} rows_in_full_menu_jobs={2} (max {3} per job) window={4} tail={5}" -f `
      $jobs.Count, $rows_total, $rows_full, (($jobs | Measure-Object Rows -Maximum).Maximum), $window, $tail)
  $summary += [pscustomobject]@{
    Kind = $kind; Keys = $keys; Jobs = $jobs.Count; RowsTotal = $rows_total; RowsFull = $rows_full
    MaxRows = ($jobs | Measure-Object Rows -Maximum).Maximum; Window = $window; Tail = $tail; Result = $result
  }
}

$summary | Export-Csv (Join-Path $OutDir "jobcost.csv") -NoTypeInformation -Encoding UTF8
Write-Host "`nRESULT s1_jobcost=done"
