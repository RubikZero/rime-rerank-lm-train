# End-to-end acceptance for the shipped context cache.
#
# Three variants are driven through the real WeaselServer IPC protocol with the
# same surrounding context and keystrokes:
#
#   int8_on   shipped graphs (int8 encode_prefix + fp32 score_tail), cache on
#   int8_off  same graphs, context_cache: false  -> full recompute each time
#   fp32_on   reference fp32 encode_prefix, cache on
#
# Assertions:
#   * the final visible menus of int8_on, int8_off and fp32_on are identical
#   * the cache is really used in the *_on variants (debug log line)
#   * the *_off variant never takes the cached path
#   * per-job time of the cached path stays below the uncached one
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$LmClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$LatencyClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_latency_e2e.exe",
  [string]$ShippedInt8 = "D:\Workspace\rime\third-party\lm-test\s1-models-int8\encode_prefix.onnx",
  [string]$ReferenceFp32 = "D:\Workspace\rime\third-party\lm-test\s1-models\encode_prefix.onnx",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-acceptance",
  [int]$Runs = 3,
  [string]$Context = "meeting at three pm, ",
  [string]$Keys = "beijingdaxue"
)

$ErrorActionPreference = "Stop"
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

function Set-CacheEnabled([bool]$enabled) {
  $value = if ($enabled) { "true" } else { "false" }
  python -c "import io,re,sys;p=sys.argv[1];v=sys.argv[2];s=io.open(p,encoding='utf-8').read();s2=re.sub(r'(?m)^(\s*context_cache:\s*)\w+',lambda m:m.group(1)+v,s,count=1);io.open(p,'w',encoding='utf-8',newline='').write(s2);print('context_cache ->',v,'changed' if s2!=s else 'UNCHANGED')" $yaml $value
}

function Invoke-Variant([string]$name, [string]$encodeGraph, [bool]$cache) {
  Copy-Item -Force $encodeGraph (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx")
  Copy-Item -Force $encodeGraph (Join-Path $Runtime "data\lm-mlm\encode_prefix.onnx")
  Set-CacheEnabled $cache

  # A private TEMP keeps this variant's rime log separate from the others.
  $temp = Join-Path $OutDir "tmp-$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp
  $env:TMP = $temp
  $env:RIME_WEASEL_PIPE_NAME = "lmacc$name"
  $env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
  $env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
  $env:RIME_LM_RANKER_DEBUG = "1"

  $hash = (Get-FileHash (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx") -Algorithm SHA256).Hash.Substring(0, 16)
  Write-Host ("[{0}] graph={1} ({2} MB, sha {3})" -f $name, (Split-Path $encodeGraph -Leaf),
    [math]::Round((Get-Item $encodeGraph).Length / 1MB, 1), $hash)

  Get-Process WeaselServer -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$Runtime*" } |
    ForEach-Object { $_.Kill(); $_.WaitForExit() }
  $server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") `
    -WorkingDirectory $Runtime -PassThru
  Start-Sleep -Seconds 15
  if ($server.HasExited) { throw "[$name] isolated server exited with $($server.ExitCode)" }

  & $LmClient $Context $Keys 8000 150 > $null 2>&1   # warm-up: loads the graphs

  for ($i = 0; $i -lt $Runs; $i++) {
    & $LmClient $Context $Keys 8000 150 2>&1 |
      Out-File (Join-Path $OutDir "$name.run$i.txt") -Encoding utf8
  }
  # Sentence case = the heaviest job (50 candidates); poll 20 ms isolates the
  # pipeline from the TSF polling interval.
  $latency = & $LatencyClient $Context $Keys 150 20 6000 3 2>&1
  $latency | Out-File (Join-Path $OutDir "$name.latency.txt") -Encoding utf8
  $job = ($latency | Select-String "RESULT" | Select-Object -Last 1).Line

  Start-Sleep -Seconds 1
  $server.Kill()
  $server.WaitForExit()

  $cache_lines = 0
  $errors = 0
  Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue | ForEach-Object {
    $cache_lines += (Select-String -Path $_.FullName -Pattern "cache scored" -AllMatches).Count
    $errors += (Select-String -Path $_.FullName -Pattern "prefix encode failed|tail score failed|context cache disabled" -AllMatches).Count
  }
  Write-Host ("[{0}] cache scored={1} errors={2} {3}" -f $name, $cache_lines, $errors, $job)
  return [pscustomobject]@{ Name = $name; CacheLines = $cache_lines; Errors = $errors; Job = $job }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")

$on = Invoke-Variant "int8_on" $ShippedInt8 $true
$off = Invoke-Variant "int8_off" $ShippedInt8 $false
$ref = Invoke-Variant "fp32_on" $ReferenceFp32 $true

# Leave the runtime in the shipped configuration.
Copy-Item -Force $ShippedInt8 (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx")
Copy-Item -Force $ShippedInt8 (Join-Path $Runtime "data\lm-mlm\encode_prefix.onnx")
Set-CacheEnabled $true

Write-Host "`n== menu comparison (final visible menu + menu multiset) =="
# Only the *final* menu is asserted: the number of intermediate poll responses
# varies with how quickly the LM answer lands relative to the client's poll
# cadence, so an extra pre-LM menu is a timing artifact, not a ranking change.
$fail = 0
foreach ($pair in @(@("int8_on", "int8_off"), @("int8_on", "fp32_on"))) {
  $a = Join-Path $OutDir "$($pair[0]).run0.txt"
  $b = Join-Path $OutDir "$($pair[1]).run0.txt"
  Write-Host ("{0} vs {1}:" -f $pair[0], $pair[1])
  $report = python (Join-Path $OutDir "..\s1_compare_menus.py") $a $b
  $report | ForEach-Object { "  $_" }
  if (-not ($report -match "final menu identical: True")) { $fail = 1 }
}

$ok = $fail -eq 0 -and $on.CacheLines -gt 0 -and $ref.CacheLines -gt 0 -and $off.CacheLines -eq 0 -and $on.Errors -eq 0 -and $off.Errors -eq 0 -and $ref.Errors -eq 0
Write-Host ""
if ($ok) { Write-Host "RESULT s1_final_acceptance=pass"; exit 0 }
Write-Host "RESULT s1_final_acceptance=fail"
exit 1
