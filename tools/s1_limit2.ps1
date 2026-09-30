# Verifies the two changes:
#   1. no fixed sentence length: an 18 syllable composition must produce a
#      sentence, with the context window shortened to make room;
#   2. canonical syllables: an abbreviated input ("bjdx") must produce the real
#      sentence (北京大学), which the typed letters alone cannot spell.
param(
  [string]$Runtime = "D:\Workspace\rime\third-party\lm-test\runtime",
  [string]$ServerExe = "D:\Workspace\rime\weasel\output\weaselserver.exe",
  [string]$RimeDll = "D:\Workspace\rime\weasel\output\rime.dll",
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_lm_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-limit2",
  [string]$Context = "meeting at three pm, "
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $ServerExe (Join-Path $Runtime "WeaselServer.exe")
Copy-Item -Force $RimeDll (Join-Path $Runtime "rime.dll")
$yaml = Join-Path $Runtime "user\lm_ranker.yaml"

# 0 = automatic: as long as the model positions allow.
python -c "import io,re,sys;p=sys.argv[1];s=io.open(p,encoding='utf-8').read();s=re.sub(r'(?m)^\s*max_syllables:.*$','    max_syllables: 0',s);io.open(p,'w',encoding='utf-8',newline='').write(s)" $yaml
Select-String -Path $yaml -Pattern "max_syllables" | ForEach-Object { "   config: " + $_.Line.Trim() }

$temp = Join-Path $OutDir "tmp"
New-Item -ItemType Directory -Force -Path $temp | Out-Null
Get-ChildItem $temp -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
$env:TEMP = $temp; $env:TMP = $temp
$env:RIME_WEASEL_PIPE_NAME = "lmlim2"
$env:RIME_WEASEL_USER_DIR = Join-Path $Runtime "user"
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"
$env:RIME_LM_RANKER_DEBUG = "1"
Get-Process WeaselServer -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$Runtime*" } | ForEach-Object { $_.Kill(); $_.WaitForExit() }
$server = Start-Process -FilePath (Join-Path $Runtime "WeaselServer.exe") -WorkingDirectory $Runtime -PassThru
Start-Sleep -Seconds 14
if ($server.HasExited) { throw "server exited with $($server.ExitCode)" }
& $ClientExe $Context "zheshiyige" 8000 150 > $null 2>&1   # warm-up

$cases = [ordered]@{
  "bjdx"                                                        = "abbrev"
  "beijingdaxue"                                                = "full"
  "zheshiyigeceshiwomenzaizuoshenmenewomenyaokaishile"          = "long18"
}
foreach ($keys in $cases.Keys) {
  $label = $cases[$keys]
  $out = & $ClientExe $Context $keys 8000 150 2>&1
  $marks = ($out | Select-String -Pattern "\[LM\]" -AllMatches).Matches.Count
  # The sentence itself: the [LM] candidate's text inside the serialized menu.
  $menu = ($out | Select-String -Pattern "ctx\.cand=" | Select-Object -Last 1).Line
  $out | Out-File (Join-Path $OutDir "$label.txt") -Encoding utf8
  $syl = ([regex]::Matches($keys, "zh|ch|sh|[a-z]")).Count
  Write-Host ("[{0,-7}] keys={1} chars={2}  [LM]={3}" -f $label, $keys, $keys.Length, $marks)
  if ($menu) { Write-Host ("           menu: {0}" -f $menu.Substring(0, [Math]::Min(150, $menu.Length))) }
}

Start-Sleep -Seconds 1
$server.Kill(); $server.WaitForExit()

Write-Host "`n== decode / window evidence =="
$logs = Get-ChildItem (Join-Path $temp "rime.weasel") -Filter "*.INFO.*" -ErrorAction SilentlyContinue |
  ForEach-Object { Select-String -Path $_.FullName -Pattern "decode steps=|cache scored|no sentence" } | ForEach-Object { $_.Line }
$logs | ForEach-Object { "  " + ($_ -replace '^\S+ \S+ \S+ ', '') } | Select-Object -Last 12
Write-Host "`nRESULT s1_limit2=done"
