# One-shot check after installing the fixed package.
#
#   1. the installed rime.dll is the fixed build (hash check)
#   2. the service is running and does not restart itself while we work
#   3. the LM path measures the same as in isolation, on the schema that has the
#      plugin (the installed server's remembered schema is wubi98, which does not
#      list lm_ranker, so the client switches inside its own session)
#   4. switching schemas repeatedly does not crash the service any more
#   5. user.yaml still remembers the user's own schema
param(
  [string]$InstallDir = "D:\Software\Rime\weasel-0.18.0",
  [string]$ExpectedDllSha = "6FDC33A771A1192F",
  [string]$SchemaClient = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$ShortContext = "meeting at three pm, ",
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. " +
    "We also need the updated headcount plan for the second half of the year, together with a short note explaining any variance against the original budget, " +
    "and a summary of the hiring pipeline for the two new teams in Shanghai and Shenzhen. "),
  [int]$SchemaDigit = 3,
  [int]$Runs = 5,
  [int]$KeyDelayMs = 150
)

$ErrorActionPreference = "Continue"
Remove-Item Env:RIME_WEASEL_PIPE_NAME -ErrorAction SilentlyContinue
Remove-Item Env:RIME_WEASEL_USER_DIR -ErrorAction SilentlyContinue
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"

Write-Host "== 1. installed build =="
$dll = Join-Path $InstallDir "rime.dll"
$sha = (Get-FileHash $dll -Algorithm SHA256).Hash.Substring(0, 16)
Write-Host ("   rime.dll {0}  (expected {1}) -> {2}" -f $sha, $ExpectedDllSha, $(if ($sha -eq $ExpectedDllSha) { "OK" } else { "MISMATCH" }))

Write-Host "`n== 2. service =="
$server = Get-Process WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $server) {
  Write-Host "   WeaselServer is not running - type a letter in a text field first"
  exit 2
}
Write-Host ("   pid {0}, started {1}" -f $server.Id, $server.StartTime)

Write-Host "`n== 3. LM latency on the plugin schema =="
foreach ($poll in 120, 20) {
  foreach ($case in "ni=char,beijing=word,beijingdaxue=sentence".Split(",")) {
    $parts = $case.Split("=")
    $line = (& $SchemaClient $ShortContext $parts[0] $KeyDelayMs $poll 8000 $Runs $SchemaDigit 2>&1 |
      Select-String "RESULT" | Select-Object -Last 1).Line
    Write-Host ("   [{0,-8} poll={1,3}ms] {2}" -f $parts[1], $poll, $line)
  }
}
$line = (& $SchemaClient $LongContext "zheshiyigeceshi" $KeyDelayMs 20 8000 3 $SchemaDigit 2>&1 |
  Select-String "RESULT" | Select-Object -Last 1).Line
Write-Host ("   [long8    poll= 20ms] {0}" -f $line)

Write-Host "`n== 4. schema switching must not kill the service =="
$before = (Get-Process WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1).Id
foreach ($digit in 2, 3, 1, 3, 2, 3, 1) {
  & $SchemaClient $ShortContext "beijing" $KeyDelayMs 120 3000 1 $digit > $null 2>&1
  Start-Sleep -Milliseconds 200
  & $SchemaClient $ShortContext "nihao" $KeyDelayMs 120 3000 1 0 > $null 2>&1
  $now = Get-Process WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $now) { Write-Host "   CRASHED during the switch loop (was pid $before)"; exit 3 }
  if ($now.Id -ne $before) { Write-Host "   service restarted during the loop (new pid $($now.Id))"; exit 3 }
}
Write-Host "   survived 7 switches, same pid $before"

Write-Host "`n== 5. user configuration =="
python -c "
import io
s = io.open(r'C:\Users\user\AppData\Roaming\Rime\user.yaml', encoding='utf-8').read()
for line in s.splitlines():
    if 'previously_selected_schema' in line:
        print('   user.yaml', line.strip())
"
Write-Host "`nRESULT post_install_check=done"
