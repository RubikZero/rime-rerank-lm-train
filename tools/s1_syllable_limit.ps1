# Checks the syllable limit for sentence generation, before and after raising it:
#   run 1  default (max_syllables 12)   -> 13/14 syllable inputs must be rejected,
#                                          and the rejection must appear in the log
#   run 2  sentence.max_syllables: 16   -> the same inputs must produce a sentence
# Also reports the per-length latency of the run that produces the mark.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-syllimit",
  [string]$Context = "meeting at three pm, ",
  [string]$Syllables = "zhe,shi,yi,ge,ce,shi,wo,men,zai,zuo,shen,me,ne",
  [string]$Lengths = "11,12,13,14",
  [int]$TimeoutMs = 8000,
  [int]$KeyDelayMs = 150
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

function Set-MaxSyllables([string]$value) {
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^(\s*max_syllables:\s*)\d+.*$',lambda m:m.group(1)+v,s);s2=re.sub(r'(?m)^(\s*min_syllables:\s*\d+.*)$',lambda m:m.group(1)+('\n    max_syllables: '+v if v!='' else ''),s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2)" $yaml $value
}

$syl = $Syllables.Split(",")

function Invoke-Round([string]$name, [string]$limit) {
  # limit "" removes the override (falls back to the shipped default).
  python -c "import io,re,sys;p=sys.argv[1];s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^\s*max_syllables:.*\n','',s);io.open(p,'w',encoding='utf-8',newline='').write(s)" $yaml
  if ($limit -ne "") { Set-MaxSyllables $limit }
  Write-Host ("`n=== {0} (sentence.max_syllables={1}) ===" -f $name, ($(if ($limit -eq "") { "default 12" } else { $limit })))

  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmsyl$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$name] server exited with $($server.ExitCode)" }
  & $ClientExe $Context "zheshiyige" $TimeoutMs $KeyDelayMs > $null 2>&1   # warm-up

  foreach ($n in ($Lengths.Split(",") | ForEach-Object { [int]$_ })) {
    $keys = ($syl[0..($n - 1)] -join "")
    $out = & $ClientExe $Context $keys $TimeoutMs $KeyDelayMs 2>&1
    $marks = ($out | Select-String -Pattern "\[LM\]" -AllMatches).Matches.Count
    $latency = & $LatencyClient $Context $keys $KeyDelayMs 20 8000 2 2>&1
    $line = ($latency | Select-String "RESULT" | Select-Object -Last 1).Line
    Write-Host ("  {0,2} 音节: [LM]={1}   {2}" -f $n, $marks, $line)
  }

  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()
  $logs = Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
    ForEach-Object { Select-String -Path $_.FullName -Pattern "no sentence for" } | ForEach-Object { $_.Line }
  Write-Host ("  拒绝日志 {0} 条:" -f $logs.Count)
  $logs | Select-Object -First 4 | ForEach-Object { "    " + ($_ -replace '^\S+ \S+ \S+ ', '') }
}

Invoke-Round "default" ""
Invoke-Round "raised" "16"

# Leave the runtime at the shipped default.
python -c "import io,re,sys;p=sys.argv[1];s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^\s*max_syllables:.*\n','',s);io.open(p,'w',encoding='utf-8',newline='').write(s)" $yaml
Write-Host "`nRESULT s1_syllable_limit=done"
