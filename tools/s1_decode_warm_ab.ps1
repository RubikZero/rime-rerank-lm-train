# A/B for the two changes in this round:
#
#   A  the frozen-prefix cache now also serves sentence decoding
#   B  the prefix is encoded as soon as the surrounding text changes (a commit),
#      instead of on the first keystroke of the next composition
#
# Three parts, all through the real IPC protocol in the isolated runtime:
#
#   1. decoding: cache on/off with the new DLL, plus the pre-change DLL (whose
#      decode is the plain full-window one) -- same context and keys, so the
#      latency and the generated sentences are directly comparable;
#   2. menus: new-on vs pre-change-on (does the cached decode change the
#      sentence?) and new-on vs new-off;
#   3. warm-up: a fresh surrounding text followed by a single keystroke, new vs
#      pre-change -- the prefix encode either happens before the key or during it.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$NewDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$OldDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-decode-ab",
  [string]$Context = "Please write the meeting notes for the quarterly review in the shared document before the end of the day, and send a copy to the team so that everyone can comment on the numbers we discussed yesterday. ",
  [string]$Cases = "zheshiyigeceshi=long8,jintiantianqihenbucuo=long9",
  [int]$Runs = 3,
  [int]$WarmRounds = 5
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
  $env:RIME_WEASEL_PIPE_NAME = "lmdab$name"
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
"new rime.dll sha: " + (Get-FileHash $NewDll -Algorithm SHA256).Hash.Substring(0, 16)
"old rime.dll sha: " + (Get-FileHash $OldDll -Algorithm SHA256).Hash.Substring(0, 16)

# ---------------------------------------------------------------- part 1 + 2
$variants = @(
  @{ Name = "new_on"; Dll = $NewDll; Cache = $true },
  @{ Name = "new_off"; Dll = $NewDll; Cache = $false },
  @{ Name = "old_on"; Dll = $OldDll; Cache = $true }
)

foreach ($v in $variants) {
  $server = Start-Isolated $v.Name $v.Dll $v.Cache
  # Warm-up with the same context: the prefix is cached, so what the measured
  # runs pay for is the per-keystroke work (rerank + decode), which is the
  # subject of part 1.
  & $LmClient $Context "beijingdaxue" 8000 150 > $null 2>&1
  foreach ($case in $Cases.Split(",")) {
    $parts = $case.Split("=")
    $latency = & $LatencyClient $Context $parts[0] 150 20 8000 $Runs 2>&1
    $latency | Out-File (Join-Path $OutDir "$($v.Name).latency.$($parts[1]).txt") -Encoding utf8
    $latency | Select-String "RESULT" | ForEach-Object { Write-Host "[$($v.Name) $($parts[1])] $($_.Line)" }
    & $LmClient $Context $parts[0] 8000 150 2>&1 |
      Out-File (Join-Path $OutDir "$($v.Name).menu.$($parts[1]).txt") -Encoding utf8
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()

  $decode = Get-LogLines $v.Name "decode steps="
  $warm = (Get-LogLines $v.Name "warmed prefix").Count
  $cache_lines = (Get-LogLines $v.Name "cache scored").Count
  $errors = (Get-LogLines $v.Name "failed").Count
  Write-Host ("[{0}] decode lines={1} warmed={2} cache scored={3} errors={4}" -f $v.Name, $decode.Count, $warm, $cache_lines, $errors)
  $decode | Select-Object -Last 4 | ForEach-Object { "        " + ($_ -replace '^.*lm_ranker: ', '') }
}

Write-Host "`n== decoded sentence / menu comparison =="
foreach ($pair in @(@("new_on", "old_on"), @("new_on", "new_off"))) {
  foreach ($case in $Cases.Split(",")) {
    $label = $case.Split("=")[1]
    Write-Host ("{0} vs {1} ({2}):" -f $pair[0], $pair[1], $label)
    python (Join-Path $OutDir "..\s1_compare_menus.py") `
      (Join-Path $OutDir "$($pair[0]).menu.$label.txt") `
      (Join-Path $OutDir "$($pair[1]).menu.$label.txt") | ForEach-Object { "  $_" }
  }
}

# ------------------------------------------------------------------- part 3
Write-Host "`n== warm-up: fresh surrounding text, then one keystroke =="
foreach ($v in @(@{ Name = "warm_new"; Dll = $NewDll }, @{ Name = "warm_old"; Dll = $OldDll })) {
  $server = Start-Isolated $v.Name $v.Dll $true
  & $LmClient $Context "beijingdaxue" 8000 150 > $null 2>&1   # load graphs
  $results = @()
  for ($i = 0; $i -lt $WarmRounds; $i++) {
    # A different surrounding text per round: each is a prefix the cache has
    # never seen, which is exactly the state right after a commit.
    $fresh = "$Context" + ("extra word $i " * (1 + $i))
    $out = & $LatencyClient $fresh "n" 150 20 8000 1 2>&1
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    $ms = if ($line -match "median=(\d+)") { [int]$Matches[1] } else { -1 }
    $results += $ms
    Write-Host ("[{0} round {1}] {2}" -f $v.Name, $i, $line)
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()
  $warm_lines = Get-LogLines $v.Name "warmed prefix"
  Write-Host ("[{0}] medians={1} warmed lines={2}" -f $v.Name, ($results -join ","), $warm_lines.Count)
}

# Leave the runtime in the shipped configuration.
Copy-Item -Force $NewDll (Join-Path $Runtime "rime.dll")
Set-CacheEnabled $true
Write-Host "`nRESULT s1_decode_warm_ab=done"
