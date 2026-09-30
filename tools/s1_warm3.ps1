# Measures B (prefix warm-up) unambiguously: every round is a fresh session whose
# surrounding text the cache has never seen (the text is longer than the maximum
# context, so appending shifts the frozen prefix).  A single keystroke follows,
# and the log says whether the prefix was already there ("prefix hit") or had to
# be encoded inside that keystroke ("prefix encoded").
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$NewDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$OldDll = "D:\Workspace\rime\third-party\lm-test\pre-ab\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-warm3",
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. "),
  [int]$Rounds = 5
)

$ErrorActionPreference = "Stop"
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

function Set-Cache([bool]$enabled) {
  $value = if ($enabled) { "true" } else { "false" }
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s2=re.sub(r'(?m)^(\s*context_cache:\s*)\w+',lambda m:m.group(1)+v,s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2)" $yaml $value
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Set-Cache $true

$summary = @{}
foreach ($v in @(@{ Name = "new"; Dll = $NewDll }, @{ Name = "old"; Dll = $OldDll })) {
  Copy-Item -Force $v.Dll (Join-Path $Runtime "rime.dll")
  $temp = Join-Path $OutDir "tmp-$($v.Name)"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  $env:TEMP = $temp; $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmw3$($v.Name)"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"
  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 14
  if ($server.HasExited) { throw "[$($v.Name)] server exited with $($server.ExitCode)" }
  & $LmClient $LongContext "beijingdaxue" 8000 150 > $null 2>&1   # load the graphs

  $ms = @()
  for ($i = 0; $i -lt $Rounds; $i++) {
    $fresh = "$LongContext" + ("round$i word " * (1 + $i))
    $out = & $LatencyClient $fresh "n" 150 20 8000 1 2>&1
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    $value = if ($line -match "median=(\d+)") { [int]$Matches[1] } else { -1 }
    $ms += $value
  }
  Start-Sleep -Seconds 1
  $server.Kill(); $server.WaitForExit()

  $logs = Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
    ForEach-Object { Select-String -Path $_.FullName -Pattern "prefix hit|prefix encoded|warmed prefix" } | ForEach-Object { $_.Line }
  $hits = ($logs | Where-Object { $_ -match "prefix hit" }).Count
  $encoded = ($logs | Where-Object { $_ -match "prefix encoded" }).Count
  $warmed = ($logs | Where-Object { $_ -match "warmed prefix" }).Count
  Write-Host ("[{0}] medians={1}" -f $v.Name, ($ms -join ","))
  Write-Host ("[{0}] prefix hits={1} encodes={2} warm jobs={3}" -f $v.Name, $hits, $encoded, $warmed)
  $summary[$v.Name] = $ms
}

Copy-Item -Force $NewDll (Join-Path $Runtime "rime.dll")
Set-Cache $true
$newAvg = ($summary["new"] | Measure-Object -Average).Average
$oldAvg = ($summary["old"] | Measure-Object -Average).Average
Write-Host ("`naverage: new={0:N0} ms  old={1:N0} ms" -f $newAvg, $oldAvg)
Write-Host "RESULT s1_warm3=done"
