# Run once after the local CVAT Compose stack is ready. The generated password
# stays in the ignored project .env file; this script never prints it.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot '.env'
if (-not (Test-Path -LiteralPath $envPath)) {
    New-Item -ItemType File -Path $envPath | Out-Null
}
if (Select-String -LiteralPath $envPath -Pattern '^CVAT_(DEV_MODE|USERNAME|PASSWORD)=' -Quiet) {
    throw 'CVAT credentials already exist in .env; edit those settings directly.'
}

$passwordBytes = New-Object byte[] 30
$generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try { $generator.GetBytes($passwordBytes) } finally { $generator.Dispose() }
$password = [Convert]::ToBase64String($passwordBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
& docker exec -e "DJANGO_SUPERUSER_PASSWORD=$password" cvat_server `
    python3 /home/django/manage.py createsuperuser --noinput `
    --username aithne-dev --email aithne-dev@localhost | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'CVAT account creation failed; .env was not changed.'
}

[System.IO.File]::AppendAllText($envPath, "`r`nCVAT_DEV_MODE=true`r`nCVAT_URL=http://host.docker.internal:8080`r`nCVAT_PUBLIC_URL=http://localhost:8080`r`nCVAT_USERNAME=aithne-dev`r`nCVAT_PASSWORD=$password`r`n")
Write-Output 'CVAT development account created and project .env configured.'
