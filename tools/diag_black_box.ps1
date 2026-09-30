# 诊断：定位「光标附近黑色方块」到底是哪个窗口
#
# 用法（在普通 PowerShell 里运行，不需要管理员）：
#   powershell -ExecutionPolicy Bypass -File D:\Workspace\rime\tools\diag_black_box.ps1
# 运行后按提示复现现象（打字 / 移动光标 / 删除文字 / 按 F4），
# 脚本会记录可疑窗口（类名、进程、位置、扩展样式）并截图。
# 结束后把控制台输出贴回来，并把 diag_out 目录里的 PNG 发我。

param(
  [int]$Seconds = 25,
  [int]$IntervalMs = 250,
  [string]$OutDir = "$PSScriptRoot\diag_out"
)

Add-Type -AssemblyName System.Drawing

Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;

public class Diag {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr p);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int i);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool GetGUIThreadInfo(uint tid, ref GUITHREADINFO gti);

  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

  [StructLayout(LayoutKind.Sequential)] public struct GUITHREADINFO {
    public int cbSize; public int flags;
    public IntPtr hwndActive; public IntPtr hwndFocus; public IntPtr hwndCapture;
    public IntPtr hwndMenuOwner; public IntPtr hwndMoveSize; public IntPtr hwndCaret;
    public RECT rcCaret;
  }

  public static GUITHREADINFO Caret(uint tid) {
    GUITHREADINFO g = new GUITHREADINFO();
    g.cbSize = Marshal.SizeOf(typeof(GUITHREADINFO));
    GetGUIThreadInfo(tid, ref g);
    return g;
  }
}
'@

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $OutDir "diag-$stamp.log"
$log = New-Object System.Collections.Generic.List[string]

function Write-Log([string]$line) {
  $log.Add($line) | Out-Null
  Add-Content -Path $logPath -Value $line -Encoding UTF8
  Write-Host $line
}

Write-Log "# 诊断开始 $stamp  时长 ${Seconds}s  采样 ${IntervalMs}ms"
$fg0 = [Diag]::GetForegroundWindow()
$buf = New-Object System.Text.StringBuilder 512
[void][Diag]::GetWindowText($fg0, $buf, 512)
Write-Log ("# 当前前台窗口: [{0}] hwnd=0x{1:X}" -f $buf.ToString(), [int64]$fg0)

$deadline = (Get-Date).AddSeconds($Seconds)
$prevKey = ''
$shotIndex = 0
$seen = @{}

while ((Get-Date) -lt $deadline) {
  $fg = [Diag]::GetForegroundWindow()
  $fgPid = 0
  [void][Diag]::GetWindowThreadProcessId($fg, [ref]$fgPid)
  $gti = [Diag]::Caret($fgPid)
  $caret = $gti.rcCaret

  $items = New-Object System.Collections.Generic.List[string]
  $script:list = $items
  $cb = [Diag+EnumWindowsProc]{
    param([IntPtr]$h, [IntPtr]$p)
    if (-not [Diag]::IsWindowVisible($h)) { return $true }
    $r = New-Object Diag+RECT
    if (-not [Diag]::GetWindowRect($h, [ref]$r)) { return $true }
    $w = $r.Right - $r.Left
    $ht = $r.Bottom - $r.Top
    if ($w -le 0 -or $ht -le 0 -or $w -gt 900 -or $ht -gt 700) { return $true }
    $ex = [Diag]::GetWindowLong($h, -20)
    # 关注：置顶(0x8) / 工具窗(0x80) / 分层(0x80000) / 透明(0x20) —— Weasel 面板都是这种
    $interesting = (($ex -band 0x8) -ne 0) -and ((($ex -band 0x80) -ne 0) -or (($ex -band 0x80000) -ne 0) -or (($ex -band 0x20) -ne 0))
    if (-not $interesting) { return $true }
    $cls = New-Object System.Text.StringBuilder 256
    [void][Diag]::GetClassName($h, $cls, 256)
    $ttl = New-Object System.Text.StringBuilder 256
    [void][Diag]::GetWindowText($h, $ttl, 256)
    $pid = 0
    [void][Diag]::GetWindowThreadProcessId($h, [ref]$pid)
    $pname = '?'
    try { $pname = (Get-Process -Id $pid -ErrorAction Stop).ProcessName } catch { }
    $line = ("hwnd=0x{0:X} pid={1} proc={2} class=[{3}] title=[{4}] rect=({5},{6})-({7},{8}) {9}x{10} ex=0x{11:X}" -f `
        [int64]$h, $pid, $pname, $cls.ToString(), $ttl.ToString(), $r.Left, $r.Top, $r.Right, $r.Bottom, $w, $ht, $ex)
    $script:list.Add($line)
    return $true
  }
  [void][Diag]::EnumWindows($cb, [IntPtr]::Zero)

  $key = ($items | Sort-Object) -join "`n"
  if ($key -ne $prevKey) {
    $prevKey = $key
    $shotIndex++
    Write-Log ""
    Write-Log ("=== #{0} {1}  光标 rect=({2},{3})-({4},{5}) ===" -f $shotIndex, (Get-Date -Format 'HH:mm:ss.fff'), $caret.Left, $caret.Top, $caret.Right, $caret.Bottom)
    if ($items.Count -eq 0) { Write-Log "  (无可疑窗口)" }
    foreach ($it in $items) {
      Write-Log "  $it"
      if ($it -notmatch 'rect=\((-?\d+),(-?\d+)\)-\((-?\d+),(-?\d+)\)') { continue }
      $L = [int]$Matches[1]; $T = [int]$Matches[2]; $R = [int]$Matches[3]; $B = [int]$Matches[4]
      $hwndTxt = if ($it -match 'hwnd=0x([0-9A-F]+)') { $Matches[1] } else { 'x' }
      $clsTxt = if ($it -match 'class=\[([^\]]*)\]') { ($Matches[1] -replace '[^\w]', '_') } else { 'x' }
      if ($clsTxt.Length -gt 24) { $clsTxt = $clsTxt.Substring(0, 24) }
      $pad = 24
      $x = [Math]::Max(0, $L - $pad); $y = [Math]::Max(0, $T - $pad)
      $ww = [Math]::Min(2000, ($R - $L) + $pad * 2); $hh = [Math]::Min(1200, ($B - $T) + $pad * 2)
      try {
        $bmp = New-Object System.Drawing.Bitmap $ww, $hh
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.CopyFromScreen($x, $y, 0, 0, (New-Object System.Drawing.Size $ww, $hh))
        $name = Join-Path $OutDir ("shot-{0}-{1}-{2}_{3}x{4}.png" -f $shotIndex, $hwndTxt, $clsTxt, ($R - $L), ($B - $T))
        $bmp.Save($name, [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose(); $bmp.Dispose()
        $seen[$name] = $true
      } catch { Write-Log "  (截图失败: $($_.Exception.Message))" }
    }
  }
  Start-Sleep -Milliseconds $IntervalMs
}

Write-Log ""
Write-Log "# 诊断结束，共 $shotIndex 个状态快照"
Write-Log "# 截图与日志目录: $OutDir"
Write-Log "# 日志文件: $logPath"
