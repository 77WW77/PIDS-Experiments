# =====================================================================
# 通用采集启动器（必须以【管理员身份】运行 PowerShell）
#
# 用法：
#   .\run_collect.ps1 -Exp 3_2 -Script 02_experiments\parent_child.py
#   .\run_collect.ps1 -Exp 3_1 -Script 02_experiments\file_ops.py
#   .\run_collect.ps1 -Exp 3_3 -Script 02_experiments\http_client.py -Mask 0x89 -ScriptArgs "--spawn-server read"
#   .\run_collect.ps1 -Exp 3_3 -Script 02_experiments\http_client.py -Mask 0x89 -ScriptArgs "--spawn-server fixed"
#
# 参数：
#   -Exp     实验编号，决定日志目录 01_kellect_logs\exp<Exp>\ 与工作目录 exp<Exp>_work
#   -Script  被测程序（相对项目根目录）
#   -Mask    -e 事件掩码，默认 0x9 = PROCESS(0x1) + FILE(0x8)
#            网络实验要加 TCPIP(0x80)：0x89 = PROCESS + FILE + TCPIP
#   -ScriptArgs  传给被测程序的额外参数（字符串，按空格切分）
#   -Warmup  启动采集器后等待多少秒再跑被测程序（默认 3）
#   -Tail    被测程序跑完后等待多少秒再停止采集（默认 3）
# =====================================================================
param(
    [Parameter(Mandatory = $true)][string]$Exp,
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$Mask = "0x9",
    [string]$ScriptArgs = "",
    [int]$Warmup = 3,
    [int]$Tail = 3
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # 否则 Expand/IO 进度会刷成 CLIXML

$Root    = $PSScriptRoot   # 脚本所在目录即项目根目录（去掉硬编码路径）
$Release = Join-Path $Root "01_kellect_logs\_tool\KellectAgent-Windows-2.0\release"
$OutDir  = Join-Path $Root "01_kellect_logs\exp$Exp"
$OutFile = Join-Path $OutDir "log.json"
$WorkDir = Join-Path $Root "02_experiments\exp${Exp}_work"
$Py      = Join-Path $Root ".venv\Scripts\python.exe"
$Exe     = Join-Path $Release "kellect.exe"

# ---------------------------------------------------------- 1. 管理员检查
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[X] 必须以【管理员身份】运行" -ForegroundColor Red
    Write-Host "    Win 键 -> 输入 powershell -> 右键 -> 以管理员身份运行" -ForegroundColor Yellow
    exit 1
}
Write-Host "[1/5] 管理员权限 OK" -ForegroundColor Green

foreach ($p in @($Exe, $Py, (Join-Path $Root $Script))) {
    if (-not (Test-Path $p)) {
        Write-Host "[X] 找不到: $p" -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------- 2. 清场
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path $OutFile) { Remove-Item $OutFile -Force }
Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
Write-Host "[2/5] 已清场（旧日志 + $WorkDir）" -ForegroundColor Green

# ---------------------------------------------------------- 3. 启动采集器
$collector = Start-Process -FilePath $Exe `
    -ArgumentList @("-e", $Mask, "-f", $OutFile) `
    -WorkingDirectory $Release -NoNewWindow -PassThru
Write-Host "[3/5] 采集器已启动 (pid=$($collector.Id))，等待 $Warmup 秒 ..." -ForegroundColor Green
Start-Sleep -Seconds $Warmup

if (-not (Test-Path $OutFile) -or (Get-Item $OutFile).Length -eq 0) {
    Write-Host "[X] 采集器没有产出日志。常见原因：权限不足 / 目录不对" -ForegroundColor Red
    Stop-Process -Id $collector.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "      已开始写日志（$((Get-Item $OutFile).Length) 字节）" -ForegroundColor DarkGray

# ---------------------------------------------------------- 4. 跑被测程序
$argList = @()
if ($ScriptArgs.Trim().Length -gt 0) { $argList = $ScriptArgs.Trim() -split '\s+' }

Write-Host "[4/5] 运行被测程序: $Script" -ForegroundColor Green
Write-Host "      >>> 记下它打印的 PID，分析时要用 <<<" -ForegroundColor Yellow
& $Py (Join-Path $Root $Script) @argList

# ---------------------------------------------------------- 5. 停止采集
Start-Sleep -Seconds $Tail
Stop-Process -Id $collector.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

$size = if (Test-Path $OutFile) { (Get-Item $OutFile).Length } else { 0 }
$lines = if ($size -gt 0) { (Get-Content $OutFile -ReadCount 0).Count } else { 0 }
Write-Host "[5/5] 采集已停止" -ForegroundColor Green
Write-Host "      日志: $OutFile" -ForegroundColor Cyan
Write-Host "      大小: $size 字节   行数: $lines" -ForegroundColor Cyan
if ($lines -eq 0) {
    Write-Host "[!] 0 行 —— 检查是否真的以管理员身份运行" -ForegroundColor Red
}