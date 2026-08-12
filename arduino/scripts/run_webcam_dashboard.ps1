param(
    [string]$Camera = "auto",
    [int]$Width = 1280,
    [int]$Height = 720,
    [switch]$Browser,
    [switch]$UseQwen,
    [switch]$NoVlm
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$argsList = @(
    ".\main.py",
    "--camera", $Camera,
    "--width", $Width,
    "--height", $Height,
    "--use_mock_sensors"
)

if ($Browser) { $argsList += "--browser" }
if ($UseQwen) { $argsList += "--use_qwen" }
if ($NoVlm) { $argsList += "--no_vlm" }

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython @argsList
} else {
    & python @argsList
}
