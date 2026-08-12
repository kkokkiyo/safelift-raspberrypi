param(
    [string]$Camera = "auto",
    [string]$SideCamera = "",
    [int]$Width = 1280,
    [int]$Height = 720,
    [switch]$Browser,
    [switch]$NoGemini
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
if ($NoGemini) { $argsList += "--no_gemini" }
if (-not [string]::IsNullOrWhiteSpace($SideCamera)) {
    $argsList += "--side_camera"
    $argsList += $SideCamera
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython @argsList
} else {
    & python @argsList
}
