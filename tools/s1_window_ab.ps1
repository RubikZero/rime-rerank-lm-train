# Verifies the larger context window (I): a surrounding text longer than either
# window is driven through the real IPC protocol with max_context 256 and 384.
#
# Checks, for each setting:
#   * the encoded window the plugin actually used (from "cache scored window=")
#   * latency, and that no graph failed
#   * the cache geometry at the larger window (tail/prefix) and that the prefix
#     warm-up still covers it
# plus a short-context regression pass, where the window is 32 either way.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-window",
  [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"
$longContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
  "so please add your comments before the end of the day and send a copy to the team, " +
  "because the finance group needs the final version for the board package that goes out on Monday morning. " +
  "We also need the updated headcount plan for the second half of the year, together with a short note explaining any variance against the original budget, " +
  "and a summary of the hiring pipeline for the two new teams in Shanghai and Shenzhen. ")

function Set-MaxContext([int]$value) {
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s2=re.sub(r'(?m)^(\s*max_context:\s*)\d+',lambda m:m.group(1)+v,s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2)" $yaml $value
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")
"surrounding text length: $($longContext.Length) chars"

foreach ($maxCtx in 256, 384) {
  Set-MaxContext $maxCtx
  $temp = Join-Path $OutDir "tmp-$maxCtx"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmwin$maxCtx"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[max_context=$maxCtx] server exited with $($server.ExitCode)" }
  & $LmClient $longContext "beijingdaxue" 8000 150 > $null 2>&1   # warm-up / model load

  Write-Host "`n-- max_context=$maxCtx, long surrounding text --"
  $latency = & $LatencyClient $longContext "zheshiyigeceshi" 150 20 8000 $Runs 2>&1
  $latency | Select-String "RESULT" | ForEach-Object { Write-Host "   $($_.Line)" }
  $latency = & $LatencyClient $longContext "beijingdaxue" 150 20 8000 $Runs 2>&1
  $latency | Select-String "RESULT" | ForEach-Object { Write-Host "   $($_.Line)" }
  & $LmClient $longContext "zheshiyigeceshi" 8000 150 2>&1 |
    Out-File (Join-Path $OutDir "menu.$maxCtx.txt") -Encoding utf8

  Write-Host "-- max_context=$maxCtx, short context (window 32 either way) --"
  $short = & $LatencyClient "meeting at three pm, " "beijing" 150 20 8000 $Runs 2>&1
  $short | Select-String "RESULT" | ForEach-Object { Write-Host "   $($_.Line)" }

  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()

  $logs = Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
    ForEach-Object { Select-String -Path $_.FullName -Pattern "cache scored|prefix encoded|warmed prefix|decode steps=|failed" } |
    ForEach-Object { $_.Line }
  $windows = $logs | ForEach-Object { if ($_ -match "window=(\d+)") { [int]$Matches[1] } } | Select-Object -Unique | Sort-Object
  $tails = $logs | ForEach-Object { if ($_ -match "tail=(\d+)") { [int]$Matches[1] } } | Select-Object -Unique | Sort-Object
  $prefixLens = $logs | ForEach-Object { if ($_ -match "prefix encoded len=(\d+)") { [int]$Matches[1] } } | Select-Object -Unique | Sort-Object
  $errors = ($logs | Where-Object { $_ -match "failed" }).Count
  $cacheLines = ($logs | Where-Object { $_ -match "cache scored" }).Count
  Write-Host ("   windows={0} tails={1} prefix lengths encoded={2} cache scored={3} errors={4}" -f `
      ($windows -join ","), ($tails -join ","), ($prefixLens -join ","), $cacheLines, $errors)
}

Set-MaxContext 384
Write-Host "`nRESULT s1_window_ab=done"
