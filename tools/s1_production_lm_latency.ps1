# Latency measured against the installed WeaselServer, on the schema that
# actually has the plugin.
#
# The user's remembered schema is wubi98, which does not list lm_ranker, so a
# measurement of the LM path has to select luna_pinyin_simp inside the session
# being measured (switcher: Control+grave opens the menu, item digit 3 picks
# 朙月拼音·简化字).  The schema is session scoped, so this does not change the
# user's own configuration -- which this script verifies afterwards.
param(
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-production",
  [string]$ShortContext = "meeting at three pm, ",
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. " +
    "We also need the updated headcount plan for the second half of the year, together with a short note explaining any variance against the original budget, " +
    "and a summary of the hiring pipeline for the two new teams in Shanghai and Shenzhen. "),
  [string]$Cases = "ni=char,beijing=word,beijingdaxue=sentence",
  [string]$Polls = "120,20",
  [int]$SchemaDigit = 3,
  [int]$Runs = 5,
  [int]$KeyDelayMs = 150
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Remove-Item Env:RIME_WEASEL_PIPE_NAME -ErrorAction SilentlyContinue
Remove-Item Env:RIME_WEASEL_USER_DIR -ErrorAction SilentlyContinue
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"

$server = Get-Process WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
"target: {0} (pid {1}, started {2})" -f $server.Path, $server.Id, $server.StartTime
"rime.dll sha: " + (Get-FileHash (Join-Path (Split-Path $server.Path) "rime.dll") -Algorithm SHA256).Hash.Substring(0, 16)
"`n== installed server, luna_pinyin_simp (plugin active) =="

foreach ($poll in $Polls.Split(",")) {
  foreach ($case in $Cases.Split(",")) {
    $parts = $case.Split("=")
    $out = & $ClientExe $ShortContext $parts[0] $KeyDelayMs $poll 8000 $Runs $SchemaDigit 2>&1
    $out | Out-File (Join-Path $OutDir "prod.$($parts[1]).poll$poll.txt") -Encoding utf8
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    "   [{0,-8} poll={1,3}ms] {2}" -f $parts[1], $poll, $line
  }
}

"`n== installed server, long context (window 384) =="
foreach ($case in "zheshiyigeceshi=long8,beijingdaxue=sentence".Split(",")) {
  $parts = $case.Split("=")
  $out = & $ClientExe $LongContext $parts[0] $KeyDelayMs 20 8000 3 $SchemaDigit 2>&1
  $out | Out-File (Join-Path $OutDir "prod.long.$($parts[1]).txt") -Encoding utf8
  $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
  "   [{0,-8} poll= 20ms] {1}" -f $parts[1], $line
}

"`n== side effects check =="
python -c "
import io
s = io.open(r'C:\Users\user\AppData\Roaming\Rime\user.yaml', encoding='utf-8').read()
for line in s.splitlines():
    if 'previously_selected_schema' in line:
        print('   user.yaml', line.strip())
"
"RESULT s1_production_lm_latency=done"
