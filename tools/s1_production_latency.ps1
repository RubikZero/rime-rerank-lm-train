# Latency measured against the *installed* WeaselServer (the one the user is
# actually typing with), not against an isolated instance.
#
# The IPC client connects to the production pipe when RIME_WEASEL_PIPE_NAME is
# unset, and drives its own session through the same protocol the TSF uses: set
# the surrounding text, type the keys with a realistic gap, then poll until the
# menu carries an LM result.  Nothing is injected into the desktop, and the
# user's own session is untouched.
#
# Caveat: the service is shared, so typing (or another program loading the
# machine) during the run shows up in the numbers.
param(
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-production",
  [string]$ShortContext = "meeting at three pm, ",
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. " +
    "We also need the updated headcount plan for the second half of the year, together with a short note explaining any variance against the original budget, " +
    "and a summary of the hiring pipeline for the two new teams in Shanghai and Shenzhen. "),
  [string]$Cases = "ni=char,beijing=word,beijingdaxue=sentence",
  [string]$Polls = "120,20",
  [int]$Runs = 5,
  [int]$KeyDelayMs = 150
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Production pipe: make sure no isolation override leaks in from this session.
Remove-Item Env:RIME_WEASEL_PIPE_NAME -ErrorAction SilentlyContinue
Remove-Item Env:RIME_WEASEL_USER_DIR -ErrorAction SilentlyContinue
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"

$server = Get-Process WeaselServer -ErrorAction SilentlyContinue |
  Select-Object -First 1
Write-Host ("target: {0} (pid {1}, started {2})" -f $server.Path, $server.Id, $server.StartTime)
Write-Host ("rime.dll sha: " + (Get-FileHash (Join-Path (Split-Path $server.Path) "rime.dll") -Algorithm SHA256).Hash.Substring(0, 16))

Write-Host "`n== warm-up (first session loads the graphs) =="
$warm = & $ClientExe $ShortContext "beijingdaxue" $KeyDelayMs 120 8000 1 2>&1
$warm | Select-String "RESULT" | ForEach-Object { "   $($_.Line)" }
if (($warm -join "`n") -match "cannot connect") { throw "cannot reach the installed WeaselServer" }

Write-Host "`n== short context (window 32) =="
foreach ($poll in $Polls.Split(",")) {
  foreach ($case in $Cases.Split(",")) {
    $parts = $case.Split("=")
    $out = & $ClientExe $ShortContext $parts[0] $KeyDelayMs $poll 8000 $Runs 2>&1
    $out | Out-File (Join-Path $OutDir "short.$($parts[1]).poll$poll.txt") -Encoding utf8
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    Write-Host ("   [{0,-8} poll={1,3}ms] {2}" -f $parts[1], $poll, $line)
  }
}

Write-Host "`n== long context, window 384 (the configured maximum) =="
foreach ($case in "zheshiyigeceshi=long8,beijingdaxue=sentence".Split(",")) {
  $parts = $case.Split("=")
  $out = & $ClientExe $LongContext $parts[0] $KeyDelayMs 20 8000 $Runs 2>&1
  $out | Out-File (Join-Path $OutDir "long.$($parts[1]).poll20.txt") -Encoding utf8
  $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
  Write-Host ("   [{0,-8} poll= 20ms] {1}" -f $parts[1], $line)
}

Write-Host "`nRESULT s1_production_latency=done"
