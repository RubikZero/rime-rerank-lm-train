# Production latency, with the schema asserted instead of switched.
#
# The user's remembered schema is luna_pinyin_simp (the one that lists the
# plugin), so a fresh session already lands on it: no switcher keystrokes are
# needed, and every measurement is only accepted when the client reports that
# schema.  A case is retried a few times because a session can occasionally come
# up on another schema.
param(
  [string]$ClientExe = "D:\Workspace\rime\third-party\lm-test\weasel_ipc_schema_latency_e2e.exe",
  [string]$OutDir = "D:\Workspace\rime\third-party\lm-test\s1-production2",
  [string]$WantSchema = "luna_pinyin_simp",
  [string]$ShortContext = "meeting at three pm, ",
  [string]$LongContext = ("The quarterly review meeting has been moved to Thursday afternoon, and the shared document now contains the numbers we discussed yesterday, " +
    "so please add your comments before the end of the day and send a copy to the team, " +
    "because the finance group needs the final version for the board package that goes out on Monday morning. " +
    "We also need the updated headcount plan for the second half of the year, together with a short note explaining any variance against the original budget, " +
    "and a summary of the hiring pipeline for the two new teams in Shanghai and Shenzhen. "),
  [int]$Runs = 5,
  [int]$KeyDelayMs = 150,
  [int]$Attempts = 3
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Remove-Item Env:RIME_WEASEL_PIPE_NAME -ErrorAction SilentlyContinue
Remove-Item Env:RIME_WEASEL_USER_DIR -ErrorAction SilentlyContinue
$env:RIME_WEASEL_IPC_TIMEOUT_MS = "5000"

$server = Get-Process WeaselServer -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $server) { Write-Host "WeaselServer is not running"; exit 2 }
Write-Host ("target: pid {0}, started {1}" -f $server.Id, $server.StartTime)
Write-Host ("rime.dll {0}" -f (Get-FileHash (Join-Path (Split-Path $server.Path) "rime.dll") -Algorithm SHA256).Hash.Substring(0, 16))

# Only measure when the session really is on the plugin schema.
function Measure-Case([string]$label, [string]$context, [string]$keys, [int]$poll) {
  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    $out = & $ClientExe $context $keys $KeyDelayMs $poll 8000 $Runs 0 2>&1
    $line = ($out | Select-String "RESULT" | Select-Object -Last 1).Line
    if (-not $line) { continue }
    if ($line -match "schema=(\S+)") {
      $schema = $Matches[1]
      if ($schema -eq $WantSchema) {
        Write-Host ("   [{0,-9} poll={1,3}ms] {2}" -f $label, $poll, $line)
        return $line
      }
      Write-Host ("   [{0,-9} poll={1,3}ms] attempt {2}: schema={3} (not the plugin schema, retrying)" -f $label, $poll, $attempt, $schema)
    }
  }
  Write-Host ("   [{0,-9} poll={1,3}ms] NO VALID MEASUREMENT" -f $label, $poll)
  return ""
}

Write-Host "`n== installed server, short context =="
foreach ($poll in 120, 20) {
  Measure-Case "char" $ShortContext "ni" $poll | Out-Null
  Measure-Case "word" $ShortContext "beijing" $poll | Out-Null
  Measure-Case "sentence" $ShortContext "beijingdaxue" $poll | Out-Null
}

Write-Host "`n== installed server, long context (window 384) =="
Measure-Case "long8" $LongContext "zheshiyigeceshi" 20 | Out-Null
Measure-Case "sentence" $LongContext "beijingdaxue" 20 | Out-Null

Write-Host "`n== schema preference (must be unchanged) =="
python -c "
import io
s = io.open(r'C:\Users\user\AppData\Roaming\Rime\user.yaml', encoding='utf-8').read()
for line in s.splitlines():
    if 'previously_selected_schema' in line:
        print('   user.yaml', line.strip())
"
Write-Host "`nRESULT production_latency2=done"
