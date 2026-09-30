# Follow-up measurements for A (cached sentence decoding) and B (prefix warm-up).
#
# Part 1 -- B, cleanly: the surrounding text is longer than the maximum context,
# so every round's frozen prefix differs from every other round's.  Each round
# therefore starts with a prefix the cache has never seen, which is the state
# right after a commit: the pre-change build pays the prefix encode inside the
# first keystroke, the new build already did it when the text was set.
#
# Part 2 -- A, on more inputs: the decoded sentence is part of the candidate
# menu, so comparing the final menu of the pre-change binary (plain decode) with
# the new one (cached decode) over many inputs measures how often the cached
# search picks the same words.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$NewDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$OldDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-ab2",
  # Longer than max_context, so truncation shifts the frozen prefix every round.
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. "),
  [int]$WarmRounds = 5,
  [string]$Inputs = "beijingdaxue,womenshizhongguoren,jintiantianqihenbucuo,zheshiyigeceshi,mingtianwoyaoqugongsi,xiexienidebangzhu,wanshangyiqichifan,shoujimeidianle,xiaxuelehuijia,qingnitongzhiwo,zhongguogongchandang,zuotianwanshanghenwan"
)

$ErrorActionPreference = "Stop"
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

function Set-CacheEnabled([bool]$enabled) {
  $value = if ($enabled) { "true" } else { "false" }
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s2=re.sub(r'(?m)^(\s*context_cache:\s*)\w+',lambda m:m.group(1)+v,s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2)" $yaml $value
}

function Start-Isolated([string]$name, [string]$dll, [bool]$cache) {
  Copy-Item -Force $dll (Join-Path $Runtime "rime.dll")
  Set-CacheEnabled $cache
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp
  $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmab2$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } |
    ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
    -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$name] isolated server exited with $($server.ExitCode)" }
  return $server
}

function Get-LogLines([string]$name, [string]$pattern) {
  Get-ChildItem (Join-Path $OutDir "tmp-$name\rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
    ForEach-Object { Select-String -Path $_.FullName -Pattern $pattern } | ForEach-Object { $_.Line }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")

Write-Host "== part 1: prefix warm-up with a cold prefix every round =="
$warmSummary = @{}
foreach ($v in @(@{ Name = "warm_new"; Dll = $NewDll }, @{ Name = "warm_old"; Dll = $OldDll })) {
  $server = Start-Isolated $v.Name $v.Dll $true
  & $LmClient $LongContext "beijingdaxue" 8000 150 > $null 2>&1   # load the graphs
  $ms = @()
  for ($i = 0; $i -lt $WarmRounds; $i++) {
    $fresh = "$LongContext" + ("round$i word " * (1 + $i))
    $out = & $LatencyClient $fresh "n" 150 20 8000 1 2>&1
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    $value = if ($line -match "median=(\d+)") { [int]$Matches[1] } else { -1 }
    $ms += $value
    Write-Host ("[{0} round {1}] median={2} ms" -f $v.Name, $i, $value)
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()
  $warmed = (Get-LogLines $v.Name "warmed prefix").Count
  $warmSummary[$v.Name] = $ms
  Write-Host ("[{0}] medians={1} warmed lines={2}" -f $v.Name, ($ms -join ","), $warmed)
}

Write-Host "`n== part 2: cached decode vs plain decode over $((($Inputs.Split(',')).Count)) inputs =="
$menus = @{ new = @{}; old = @{} }
foreach ($v in @(@{ Name = "decode_new"; Dll = $NewDll; Key = "new" },
                 @{ Name = "decode_old"; Dll = $OldDll; Key = "old" })) {
  $server = Start-Isolated $v.Name $v.Dll $true
  & $LmClient $LongContext "beijingdaxue" 8000 150 > $null 2>&1   # warm-up
  foreach ($keys in $Inputs.Split(",")) {
    & $LmClient $LongContext $keys 8000 150 2>&1 |
      Out-File (Join-Path $OutDir "$($v.Name).menu.$keys.txt") -Encoding utf8
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()
  $decode = Get-LogLines $v.Name "decode steps="
  $rows = 0; $cached = 0; $positions = 0; $total_ms = 0
  foreach ($line in $decode) {
    if ($line -match "rows=(\d+) cached_rows=(\d+) .* positions=(\d+) ms=(\d+)") {
      $rows += [int]$Matches[1]; $cached += [int]$Matches[2]
      $positions += [int]$Matches[3]; $total_ms += [int]$Matches[4]
    }
  }
  Write-Host ("[{0}] decode calls={1} rows={2} cached_rows={3} positions={4} decode_ms={5}" -f `
      $v.Name, $decode.Count, $rows, $cached, $positions, $total_ms)
}

$same = 0; $total = 0
foreach ($keys in $Inputs.Split(",")) {
  $a = Join-Path $OutDir "decode_new.menu.$keys.txt"
  $b = Join-Path $OutDir "decode_old.menu.$keys.txt"
  $report = python (Join-Path $OutDir "..\s1_compare_menus.py") $a $b
  $identical = ($report -match "final menu identical: True")
  $total++
  if ($identical) { $same++ }
  Write-Host ("  {0,-26} final menu identical={1}" -f $keys, $identical)
  if (-not $identical) { $report | ForEach-Object { "      $_" } }
}
Write-Host ("`ncached decode agreement: {0}/{1} ({2:N1}%)" -f $same, $total, (100.0 * $same / [math]::Max(1, $total)))

Copy-Item -Force $NewDll (Join-Path $Runtime "rime.dll")
Set-CacheEnabled $true
Write-Host "`nRESULT s1_ab_followup=done"
