# Verifies the *shipped* context-cache graphs in the live C++ path:
#   variant "int8" = shipped encode_prefix.onnx (int8) + score_tail.onnx (fp32)
#   variant "fp32" = reference encode_prefix.onnx (fp32) + score_tail.onnx (fp32)
#
# Both variants are driven through the real WeaselServer IPC protocol with the
# same surrounding context and keystrokes; the test asserts that the visible
# candidate menus are identical and that the cache was actually used (the
# "cache scored" debug line, which only the cached path emits).
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$ShippedInt8 = "D:\Workspace\rime\third-party\lm-test\s1-models-int8\encode_prefix.onnx",
  [string]$ReferenceFp32 = "D:\Workspace\rime\third-party\lm-test\s1-models\encode_prefix.onnx",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-int8-runtime",
  [int]$Runs = 3
)

$ErrorActionPreference = "Stop"
$logDir = Join-Path $Runtime "temp\rime.weasel"

function Invoke-Variant([string]$name, [string]$encodeGraph) {
  Copy-Item -Force $encodeGraph (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx")
  $stamp = (Get-Date).AddSeconds(-2)
  $pipe = "lmint8$name"
  $env:RIME_WEASEL_PIPE_NAME = $pipe
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

  # Warm-up: the first session loads both graphs (~1-2s).
  & $ClientExe "meeting at three pm, " "beijingdaxue" 8000 150 > $null 2>&1

  $menus = @()
  for ($i = 1; $i -le $Runs; $i++) {
    $out = & $ClientExe "meeting at three pm, " "beijingdaxue" 8000 150 2>&1
    $menus += ,@($out | Where-Object { $_ -notmatch "^RESULT" })
    Write-Host ("[{0}] run {1}: {2}" -f $name, $i, ($out | Select-String "RESULT").Line)
  }

  Start-Sleep -Seconds 1
  $server.Kill()
  $server.WaitForExit()

  $log = Get-ChildItem $logDir -Filter "*.INFO.*" |
    Where-Object { $_.LastWriteTime -gt $stamp } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $cache = 0
  if ($log) {
    $cache = (Select-String -Path $log.FullName -Pattern "cache scored" -AllMatches).Count
  }
  Write-Host ("[{0}] cache scored lines: {1}" -f $name, $cache)
  return [pscustomobject]@{ Menus = $menus; CacheLines = $cache }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")

$int8 = Invoke-Variant "int8" $ShippedInt8
$fp32 = Invoke-Variant "fp32" $ReferenceFp32

# Leave the runtime in the shipped configuration.
Copy-Item -Force $ShippedInt8 (Join-Path $Runtime "user\lm-mlm\encode_prefix.onnx")

$same = $true
for ($i = 0; $i -lt $Runs; $i++) {
  $a = ($int8.Menus[$i] -join "`n")
  $b = ($fp32.Menus[$i] -join "`n")
  Set-Content -Path (Join-Path $OutDir "run$i.int8.txt") -Value $a -Encoding UTF8
  Set-Content -Path (Join-Path $OutDir "run$i.fp32.txt") -Value $b -Encoding UTF8
  if ($a -ne $b) {
    $same = $false
    Write-Host "run $i differs:"
    Compare-Object ($a -split "`n") ($b -split "`n") | Format-Table -AutoSize
  }
}

$ok = $same -and $int8.CacheLines -gt 0 -and $fp32.CacheLines -gt 0
Write-Host ("menus identical: {0}; cache used: int8={1} fp32={2}" -f $same, $int8.CacheLines, $fp32.CacheLines)
if ($ok) { Write-Host "RESULT s1_int8_runtime_check=pass"; exit 0 }
Write-Host "RESULT s1_int8_runtime_check=fail"
exit 1
