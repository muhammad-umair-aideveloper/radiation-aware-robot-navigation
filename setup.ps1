# setup.ps1 - One-time repository bootstrap script. Run it yourself.
#
# Place these files in the project folder before running:
#   AGENTS.md  SPEC.md  STRUCTURE.md  requirements.txt  config.py  setup.ps1
#
# Usage (PowerShell, inside the project folder):
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Holder "Your Full Name"
#
# What it does: git init, small conventional commits, .gitignore, MIT license,
# folder skeleton, src/config.py, README placeholder, virtual environment,
# CPU PyTorch + requirements, import check and a tiny PPO speed check.
# It never pushes and never touches remotes. You can delete this file afterwards
# (it is listed in .gitignore, so it is never committed).

param([string]$Holder = "")

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit code $LASTEXITCODE)" }
}
function Commit($files, $msg) {
    git add -- $files
    Check "git add"
    git diff --cached --quiet -- $files
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Nothing new to commit for: $msg"
        $global:LASTEXITCODE = 0
        return
    }
    git commit -m $msg
    Check "git commit"
}

# ---------------------------------------------------------------- checks
foreach ($f in @("AGENTS.md", "SPEC.md", "STRUCTURE.md", "requirements.txt")) {
    if (-not (Test-Path $f)) { throw "Missing $f. Run this script inside the project folder." }
}
if (-not (Test-Path "src\config.py") -and -not (Test-Path "config.py")) {
    throw "Missing config.py (put it in the project folder next to this script)."
}
if ([string]::IsNullOrWhiteSpace($Holder)) { $Holder = (git config user.name) }
if ([string]::IsNullOrWhiteSpace($Holder)) { throw "Pass your name: -Holder 'Your Full Name'" }
$email = (git config user.email)
if ([string]::IsNullOrWhiteSpace($email)) { throw "git user.email is not set. Set it once with: git config --global user.email you@example.com" }
Write-Host "License holder : $Holder"
Write-Host "Git author     : $(git config user.name) <$email>"

# ---------------------------------------------------------------- git
Step "Git init and documentation commits"
if (-not (Test-Path ".git")) { git init -b main; Check "git init" }
Commit @("AGENTS.md", "SPEC.md", "STRUCTURE.md") "docs: add agent rules, specification and project structure"
Commit @("requirements.txt") "build: add python dependencies"

# ---------------------------------------------------------------- hygiene
Step ".gitignore and MIT license"
$gitignore = @'
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.idea/
.vscode/
.DS_Store
tmp/
*.tmp
tensorboard/
setup.ps1
'@
Set-Content -Path ".gitignore" -Value $gitignore -Encoding ASCII

$license = @"
MIT License

Copyright (c) 2026 $Holder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"@
Set-Content -Path "LICENSE" -Value $license -Encoding ASCII
Commit @(".gitignore", "LICENSE") "chore: add gitignore and MIT license"

# ---------------------------------------------------------------- skeleton
Step "Folder skeleton"
foreach ($d in @("src", "tests", "models", "results", "results\runs")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$keep = @("tests\.gitkeep", "models\.gitkeep", "results\.gitkeep", "results\runs\.gitkeep")
foreach ($k in $keep) {
    if (-not (Test-Path $k)) { New-Item -ItemType File -Path $k | Out-Null }
}
Commit $keep "chore: add project folder skeleton"

# ---------------------------------------------------------------- config
Step "Central configuration"
if (Test-Path "config.py") { Move-Item -Force "config.py" "src\config.py" }
Commit @("src\config.py") "feat(config): add central configuration constants"

# ---------------------------------------------------------------- README
Step "Placeholder README"
$readme = "# radiation-aware-robot-navigation`n`nRadiation-aware robot navigation on a 2D grid: A* planners and PPO agents compared on absorbed dose and path length.`n"
Set-Content -Path "README.md" -Value $readme -Encoding ASCII
Commit @("README.md") "docs: add placeholder README"

# ---------------------------------------------------------------- venv
Step "Virtual environment and packages (this can take several minutes)"
if (-not (Test-Path ".venv")) { python -m venv .venv; Check "python -m venv" }
$py = ".\.venv\Scripts\python.exe"
& $py --version
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
Check "pip install torch (cpu)"
& $py -m pip install -r requirements.txt
Check "pip install -r requirements.txt"

Step "Import check"
& $py -c "import torch, numpy, gymnasium, stable_baselines3, matplotlib, imageio, pandas; print('torch', torch.__version__); print('gymnasium', gymnasium.__version__); print('sb3', stable_baselines3.__version__)"
Check "import check"
& $py -m pytest --version
Check "pytest"

Step "Config check"
& $py -c "import sys; sys.path.insert(0, 'src'); import config; print('OBS_DIM', config.OBS_DIM, '| KAPPA_AWARE', config.KAPPA_AWARE, '| REPO_ROOT', config.REPO_ROOT.name)"
Check "config import"

Step "Tiny PPO speed check (CartPole, 2000 steps, CPU)"
& $py -c "import time; from stable_baselines3 import PPO; t = time.time(); m = PPO('MlpPolicy', 'CartPole-v1', device='cpu', verbose=0); m.learn(2000); print('PPO CartPole 2000 steps: %.1f s' % (time.time() - t))"
Check "ppo sanity check"

# ---------------------------------------------------------------- done
Step "Done"
git status -sb
git log --oneline
Write-Host ""
Write-Host "Nothing was pushed. Copy the output above to Claude if anything looks wrong." -ForegroundColor Green
