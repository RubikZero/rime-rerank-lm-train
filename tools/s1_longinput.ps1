# Reproduces "the [LM] sentence mark stops appearing when the composition gets
# longer".  Types prefixes of 4..14 syllables of the same syllable string and
# records, per length:
#   * how many [LM] marks the menu carries
#   * every decision the plugin logged on stderr (rank skip/miss/hit, the
#     sentence worker's verdict, sentence submit/deny)
#   * whether the model produced a sentence candidate at all
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-longinput",
  [string]$Context = "meeting at three pm, ",
  [string]$Syllables = "zhe,shi,yi,ge,ce,shi,wo,men,zai,zuo,shen,me,ne",
  [string]$Lengths = "4,6,8,10,11,12,13,14",
  [int]$TimeoutMs = 8000,
  [int]$KeyDelayMs = 150
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")

$temp = Join-Path $OutDir "tmp"
New-Item -ItemType Directory -Force -Path $temp | Out-Null
Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
$env:TEMP = $temp; $env:TMP = $temp
$env:RIME_WEASEL_PIPE_NAME = "lmlongin"
$env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
$env:RIME_LM_RANKER_DEBUG = "1"
Get-Process WeaselServer -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }

$stderr = Join-Path $temp "server-stderr.txt"
$server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
  -WorkingDirectory $Runtime -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 14
if ($server.HasExited) { throw "isolated server exited with $($server.ExitCode)" }

$syl = $Syllables.Split(",")
& $ClientExe $Context "zheshiyige" $TimeoutMs $KeyDelayMs > $null 2>&1   # warm-up

$report = @()
foreach ($n in ($Lengths.Split(",") | ForEach-Object { [int]$_ })) {
  $keys = ($syl[0..($n - 1)] -join "")
  $before = (Get-Content $stderr -ErrorAction SilentlyContinue).Count
  $out = & $ClientExe $Context $keys $TimeoutMs $KeyDelayMs 2>&1
  $marks = ($out | Select-String -Pattern "\[LM\]" -AllMatches).Matches.Count
  $sentences = ($out | Select-String -Pattern "candidate.type=lm_ranker" -AllMatches).Matches.Count
  $lastMenu = ($out | Select-String -Pattern "ctx.cand=(\d+)" | Select-Object -Last 1).Line
  $count = if ($lastMenu -match "ctx\.cand=(\d+)") { [int]$Matches[1] } else { -1 }
  $lines = (Get-Content $stderr -ErrorAction SilentlyContinue | Select-Object -Skip $before)
  $decisions = ($lines | Select-String -Pattern "rank skip|rank miss|rank hit|worker |sentence" | ForEach-Object { $_.Line }) -join " | "
  $report += [pscustomobject]@{
    Syllables = $n; Keys = $keys; Marks = $marks; SentenceCandidates = $sentences
    MenuSize = $count; Decisions = $decisions
  }
  Write-Host ("[{0,2} 音节] keys={1,-26} [LM]={2} 整句候选={3} 菜单={4}" -f $n, $keys, $marks, $sentences, $count)
  if ($decisions) { Write-Host ("           决策: {0}" -f ($decisions -replace '^\[lm_ranker\] ', '')) }
}

Start-Sleep -Seconds 1
$server.Kill(); $server.WaitForExit()

Write-Host "`n=== 汇总 ==="
$report | Select-Object Syllables, Marks, SentenceCandidates, MenuSize | Format-Table -AutoSize
$report | Export-Csv (Join-Path $OutDir "longinput.csv") -NoTypeInformation -Encoding UTF8
Write-Host "RESULT s1_longinput=done"
