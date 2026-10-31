function Get-ProjectNpmCommand {
    $configuredNodePath = $env:GUARDIAN_NODE_HOME
    if ($configuredNodePath) {
        $nodeDirectory = [IO.Path]::GetFullPath($configuredNodePath)
        $npmCandidate = Join-Path $nodeDirectory 'npm.cmd'
        if (-not (Test-Path -LiteralPath $npmCandidate)) {
            throw 'GUARDIAN_NODE_HOME must contain npm.cmd.'
        }
        $env:Path = $nodeDirectory + ';' + $env:Path
    } else {
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $npmCommand) { throw 'Install Node.js 22 or newer and add it to PATH.' }
        $npmCandidate = $npmCommand.Source
    }
    & node -e 'if (Number(process.versions.node.split(".")[0]) < 22) process.exit(1)'
    if ($LASTEXITCODE -ne 0) { throw 'Node.js 22 or newer is required.' }
    return $npmCandidate
}
