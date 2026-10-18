# Run from the project root in the VS Code PowerShell terminal
$dirs = @("data\raw\van_gogh","data\raw\monet","data\raw\madhubani","data\processed",
          "src\data","src\models","src\train","src\eval","src\app","configs","notebooks","outputs")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
foreach ($p in @("src","src\data","src\models","src\train","src\eval","src\app")) {
  New-Item -ItemType File -Force -Path "$p\__init__.py" | Out-Null }
Write-Host "Folders ready."
