# Live A/B of the context cache at a *long* surrounding context, where the frozen
# prefix is large and the per-keystroke saving is expected to be several fold.
#
# The isolated runtime is driven twice through the real IPC protocol (cache on,
# cache off) with a ~200 character surrounding text, which pushes the adaptive
# window to its maximum.  Reports the client-visible latency and compares the
# final candidate menus.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$Int8Graph = "D:\Workspace\rime\third-party\lm-test\s1-models-int8\encode_prefix.onnx",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-longctx",
  [string]$Context = "Please write the meeting notes for the quarterly review in the shared document before the end of the day, and send a copy to the team so that everyone can comment on the numbers we discussed yesterday. ",
  [string]$Keys = "beijingdaxue",
  [string]$Cases = "ni=char,beijing=word,beijingdaxue=sentence",
  [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

function Set-CacheEnabled([bool]$enabled) {
  $value = if ($enabled) { "true" } else { "false" }
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s2=re.sub(r'(?m)^(\s*context_cache:\s*)\w+',lambda m:m.group(1)+v,s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2)" $yaml $value
}

function Invoke-Variant([string]$name, [bool]$cache) {
  Set-CacheEnabled $cache
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp
  $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmlong$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"

  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } |
    ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
    -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 15
  if ($server.HasExited) { throw "[$name] isolated server exited with $($server.ExitCode)" }

  & $LmClient $Context $Keys 8000 150 > $null 2>&1      # warm-up / graph load
  & $LmClient $Context $Keys 8000 150 2>&1 | Out-File (Join-Path $OutDir "$name.menu.txt") -Encoding utf8

  foreach ($case in $Cases.Split(",")) {
    $parts = $case.Split("=")
    $latency = & $LatencyClient $Context $parts[0] 150 20 8000 $Runs 2>&1
    $latency | Out-File (Join-Path $OutDir "$name.latency.$($parts[1]).txt") -Encoding utf8
    $latency | Select-String "RESULT" | ForEach-Object { Write-Host "[$name $($parts[1])] $($_.Line)" }
  }

  Start-Sleep -Seconds 1
  $server.Kill()
  $server.WaitForExit()

  $windows = @()
  $cache_lines = 0
  $errors = 0
  Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue | ForEach-Object {
    $hits = Select-String -Path $_.FullName -Pattern "cache scored"
    $cache_lines += $hits.Count
    $windows += $hits | ForEach-Object { if ($_.Line -match "window=(\d+)") { [int]$Matches[1] } }
    $errors += (Select-String -Path $_.FullName -Pattern "prefix encode failed|tail score failed" -AllMatches).Count
  }
  $shapes = Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
    ForEach-Object { Select-String -Path $_.FullName -Pattern "model loaded" } | ForEach-Object { $_.Line }
  $shapes | ForEach-Object { Write-Host "[$name] $($_ -replace '^.*lm_ranker: ', '')" }
  Write-Host ("[{0}] cache scored={1} windows={2} errors={3}" -f $name, $cache_lines,
    (($windows | Select-Object -Unique | Sort-Object) -join ","), $errors)
  return [pscustomobject]@{ Name = $name; CacheLines = $cache_lines; Windows = $windows; Errors = $errors }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")
Copy-Item -Force $Int8Graph (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx")
Copy-Item -Force $Int8Graph (Join-Path $Runtime "data\lm-mlm\encode_prefix.onnx")

$on = Invoke-Variant "on" $true
$off = Invoke-Variant "off" $false

Set-CacheEnabled $true   # leave the runtime in the shipped configuration

Write-Host "`n== menu comparison =="
python (Join-Path $OutDir "..\s1_compare_menus.py") (Join-Path $OutDir "on.menu.txt") (Join-Path $OutDir "off.menu.txt") | ForEach-Object { "  $_" }
$same = $LASTEXITCODE -eq 0

$ok = $on.CacheLines -gt 0 -and $off.CacheLines -eq 0 -and $on.Errors -eq 0 -and $off.Errors -eq 0
if ($ok) { Write-Host "RESULT s1_longctx_ab=pass" } else { Write-Host "RESULT s1_longctx_ab=fail" }
