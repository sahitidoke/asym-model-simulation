# Auto-commit new ROC results and push to origin/hao.
#
# Registered as Scheduled Task "asym-autocommit", running every 30 minutes.
# Remove it with:   schtasks /delete /tn asym-autocommit /f
# Its own log:      %LOCALAPPDATA%\asym-autocommit.log   (outside the repo)
#
# Only results/ and roc_run.log are ever staged. Code changes are deliberately
# left alone so a half-finished edit cannot reach the shared repo unattended.

$repo   = 'C:\Users\celem\Research\asym-model-simulation'
$branch = 'hao-auto-results'
$log    = Join-Path $env:LOCALAPPDATA 'asym-autocommit.log'

function Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $log -Encoding utf8
}

Set-Location $repo

# Refuse to act unless the working tree is actually on the dedicated branch.
# Without this, checking out hao later would land auto-commits there while the
# push silently no-ops against an unchanged $branch ref.
$current = "$(git rev-parse --abbrev-ref HEAD)".Trim()
if ($current -ne $branch) {
    Log "on branch '$current', expected '$branch' -- skipping (git checkout $branch to resume)"
    exit 0
}

$paths = @()
foreach ($pth in 'results', 'roc_run.log') {
    if (Test-Path (Join-Path $repo $pth)) { $paths += $pth }
}
if ($paths.Count -eq 0) { Log 'nothing to stage'; exit 0 }

git add -- $paths

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) { Log 'no changes'; exit 0 }

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
git commit -m "Auto-commit results $stamp" | Out-Null
if ($LASTEXITCODE -ne 0) { Log 'commit FAILED'; exit 1 }
Log "committed $stamp"

# Push first. Pulling unconditionally would fail every cycle: `git pull
# --rebase` refuses to run with unstaged changes, and the tracked notebook is
# rewritten constantly while the kernel runs. Only rebase if the push is
# actually rejected, i.e. a collaborator pushed since last time.
git push origin $branch | Out-Null
if ($LASTEXITCODE -eq 0) { Log "pushed to origin/$branch"; exit 0 }

Log 'push rejected; rebasing onto origin'
git pull --rebase --autostash origin $branch | Out-Null
if ($LASTEXITCODE -ne 0) {
    git rebase --abort | Out-Null
    Log 'rebase FAILED (conflict?); commit kept local, retry next run'
    exit 1
}

git push origin $branch | Out-Null
if ($LASTEXITCODE -ne 0) { Log 'push FAILED after rebase; commit kept local'; exit 1 }
Log "pushed to origin/$branch after rebase"