param(
    [string]$InputDir = "",

    [ValidateSet("ucl","bonn")]
    [string]$Site = "ucl",

    [string]$MatlabPath    = "D:\matlab\r2022b\bin\matlab.exe",
    [string]$PspmPath      = "D:\matlab\toolboxes\PsPM\git\PsPM\src",
    [string]$PspmPathFile  = "",

    [string[]]$Channels = @("scr","ecg","resp","marker"),

    [int]$Workers = 1,
    [switch]$Recurse,
    [switch]$Overwrite,
    [switch]$DebugMode,
    [switch]$Help,

    [string]$EyeDir = "",
    [string]$EyeExtension = ".edf",
    [switch]$MatchEyeFiles,
    [switch]$RequireEyeFiles,

    [double]$TrackDist = 700
)

function To-MatlabPath([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return "" }
    return ($p -replace '\\','/')
}

function Escape-MatlabString([string]$s) {
    if ($null -eq $s) { return "" }
    return ($s -replace "'", "''")
}

function Quote-MatlabString([string]$s) {
    return "'" + (Escape-MatlabString (To-MatlabPath $s)) + "'"
}

if ($Help) {
    Write-Host ""
    Write-Host "CALINET import batch runner"
    Write-Host "==============================================================="
    Write-Host ""
    Write-Host "USAGE"
    Write-Host "  .\calinet_import_smr.ps1 -InputDir <dir> -Site <ucl|bonn> [options]"
    Write-Host ""
    Write-Host "REQUIRED"
    Write-Host "  -InputDir        Directory containing source files"
    Write-Host "  -Site            ucl or bonn"
    Write-Host ""
    Write-Host "OPTIONAL"
    Write-Host "  -MatlabPath      Full path to matlab executable"
    Write-Host "  -PspmPath        Full PsPM directory"
    Write-Host "  -PspmPathFile    Text file containing PsPM path"
    Write-Host "  -Channels        Channel list, default: scr ecg resp marker"
    Write-Host "  -Workers         Parallel workers, default: 1"
    Write-Host "  -Recurse         Search subdirectories"
    Write-Host "  -Overwrite       Overwrite existing outputs"
    Write-Host "  -MatchEyeFiles   Try to match eye files by basename"
    Write-Host "  -EyeDir          Directory for eye files (default: InputDir)"
    Write-Host "  -EyeExtension    Eye file extension, default: .edf"
    Write-Host "  -RequireEyeFiles Fail when a matching eye file is missing"
    Write-Host "  -TrackDist       Eye tracker distance in mm, default: 700"
    Write-Host "  -Debug           Print generated MATLAB code and exit"
    Write-Host ""
    Write-Host "EXAMPLES"
    Write-Host "  .\calinet_import_smr.ps1 -InputDir 'Z:\CALINET2\sourcedata\london' -Site ucl"
    Write-Host "  .\calinet_import_smr.ps1 -InputDir 'Z:\CALINET2\sourcedata\london' -Site ucl -Recurse -Workers 4 -Overwrite"
    Write-Host "  .\calinet_import_smr.ps1 -InputDir 'Z:\CALINET2\sourcedata\london' -Site ucl -MatchEyeFiles -EyeDir 'D:\eye' -EyeExtension '.edf'"
    Write-Host ""
    exit 0
}

if (-not (Test-Path $InputDir)) {
    Write-Host "ERROR: InputDir not found: $InputDir"
    exit 1
}

$InputDirResolved = (Resolve-Path $InputDir).Path

if ($MatlabPath -eq "") {
    $MatlabExe = "matlab"
} else {
    if (-not (Test-Path $MatlabPath)) {
        Write-Host "ERROR: MatlabPath not found: $MatlabPath"
        exit 1
    }
    $MatlabExe = $MatlabPath
}

if ($PspmPath -ne "" -and -not (Test-Path $PspmPath)) {
    Write-Host "ERROR: PspmPath not found: $PspmPath"
    exit 1
}

if ($PspmPathFile -ne "" -and -not (Test-Path $PspmPathFile)) {
    Write-Host "ERROR: PspmPathFile not found: $PspmPathFile"
    exit 1
}

if ($EyeDir -eq "") {
    $EyeDirResolved = $InputDirResolved
} else {
    if (-not (Test-Path $EyeDir)) {
        Write-Host "ERROR: EyeDir not found: $EyeDir"
        exit 1
    }
    $EyeDirResolved = (Resolve-Path $EyeDir).Path
}

$gciParams = @{
    Path   = $InputDirResolved
    Filter = "*.smr"
    File   = $true
}

if ($Recurse) {
    $gciParams.Recurse = $true
}

$smrFiles = Get-ChildItem @gciParams | Sort-Object FullName

if ($smrFiles.Count -eq 0) {
    Write-Host "ERROR: No .smr files found in $InputDirResolved"
    exit 1
}

$jobs = @()
foreach ($f in $smrFiles) {
    $eyeFile = $null

    if ($MatchEyeFiles) {
        $candidate = Join-Path $EyeDirResolved ($f.BaseName + $EyeExtension)
        if (Test-Path $candidate) {
            $eyeFile = (Resolve-Path $candidate).Path
        } elseif ($RequireEyeFiles) {
            Write-Host "ERROR: Missing eye file for $($f.Name): expected $candidate"
            exit 1
        }
    }

    $jobs += [PSCustomObject]@{
        Physio = $f.FullName
        Eye    = $eyeFile
    }
}

$TmpRoot = Join-Path $env:TEMP ("calinet_import_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TmpRoot | Out-Null

$MatlabFuncFile = Join-Path $TmpRoot "calinet_import_smr.m"
$MatlabBatchFile = Join-Path $TmpRoot "run_calinet_import_batch.m"
$LogFile = Join-Path $TmpRoot "run_calinet_import_batch.log"

$overwriteVal = if ($Overwrite) { "1" } else { "0" }

$MatlabFunction = @"
function [outnames] = calinet_import_smr(inarg)
% adapted for batch import from PowerShell
%
% required:
%   inarg.pfn
%   inarg.site
%
% optional:
%   inarg.chan
%   inarg.efn
%   inarg.trackdist
%   inarg.overwrite

outnames = [];
physiochannels = {'scr', 'ecg', 'resp', 'marker'};
eyechannels = {'pupil_l', 'pupil_r', 'gaze_x_l', 'gaze_y_l', ...
               'gaze_x_r', 'gaze_y_r', 'marker'};

if nargin < 1, fprintf('Don''t know what to do.\n'); return; end
if ~isstruct(inarg), fprintf('Struct input needed\n'); return; end
if ~isfield(inarg, 'pfn'), fprintf('Physio file name required\n'); return; end
if ~isfield(inarg, 'site'), fprintf('Site information required\n'); return; end
if ~isfield(inarg, 'chan') || isempty(inarg.chan), inarg.chan = physiochannels; end
if ~isfield(inarg, 'trackdist') || isempty(inarg.trackdist), inarg.trackdist = 700; end
if ~isfield(inarg, 'overwrite') || isempty(inarg.overwrite), inarg.overwrite = 0; end

if strcmpi(inarg.site, 'bonn')
    datatype = 'acq_python';
    channellist = 1:4;
    transfer = struct( ...
        'c', 1, ...
        'Rs', 0, ...
        'offset', 0);
elseif strcmpi(inarg.site, 'ucl')
    % see calinet-main/misc/LDN_transfer_function.mat
    datatype = 'smr';
    channellist = [3, 5, 4, 6];
    transfer = struct( ...
        'c', 35.6128, ...
        'Rs', 2004, ...
        'offset', 113.4377);
else
    error('Unsupported site: %s', inarg.site);
end

options = struct('overwrite', inarg.overwrite);

n_chans = numel(inarg.chan);
import = cell(1, n_chans);
for iChannel = 1:n_chans
    import{iChannel}.type = inarg.chan{iChannel};
    import{iChannel}.channel = channellist(iChannel);
    if iChannel == 1
        import{iChannel}.transfer = transfer;
    end
    if strcmpi(inarg.chan{iChannel}, 'marker')
        import{iChannel}.flank = 'ascending';
    end
end

[sts, newpfn] = pspm_import(inarg.pfn, datatype, import, options);
if sts < 1
    fprintf('Import failed for %s\n', inarg.pfn);
    outnames = {[], []};
    return
end

[~, ~, data] = pspm_load_data(newpfn, 'marker');
pmarkerno = numel(data{1}.data);
fprintf('%d markers in imported file %s ... \n', pmarkerno, newpfn);

options = struct('overwrite', inarg.overwrite, 'drop_offset_markers', 1);
[sts, newpfn] = pspm_trim(newpfn, -5, 30, 'marker', options);
if sts < 1
    fprintf('Trim failed for %s\n', newpfn);
end

if isfield(inarg, 'efn') && ~isempty(inarg.efn)
    options = struct('overwrite', inarg.overwrite);
    datatype = 'eyelink';
    n_eye = numel(eyechannels);
    import = cell(n_eye, 1);

    for iChannel = 1:n_eye
        import{iChannel}.type = eyechannels{iChannel};
        import{iChannel}.channel = iChannel;

        if contains(import{iChannel}.type, 'pupil')
            import{iChannel}.eyelink_trackdist = inarg.trackdist;
            import{iChannel}.distance_unit = 'mm';
        end
    end

    [sts, newefn] = pspm_import(inarg.efn, datatype, import, options);

    if sts > 0
        [~, ~, data] = pspm_load_data(newefn, 'marker');
        emarkerno = numel(data{1}.data);
        fprintf('%d markers in imported file %s ... \n', emarkerno, newefn);

        options = struct('overwrite', inarg.overwrite, 'drop_offset_markers', 1);
        [sts, newefn] = pspm_trim(newefn, -5, 30, 'marker', options);
        if sts < 1
            fprintf('Eye trim failed for %s\n', newefn);
        end
    else
        fprintf('Eye import failed for %s\n', inarg.efn);
        newefn = [];
    end
else
    newefn = [];
end

outnames = {newpfn; newefn};
end
"@

Set-Content -Path $MatlabFuncFile -Value $MatlabFunction -Encoding UTF8

$matlabJobs = @()
foreach ($j in $jobs) {
    $physio = Quote-MatlabString $j.Physio
    $eye = if ($null -ne $j.Eye -and $j.Eye -ne "") { Quote-MatlabString $j.Eye } else { "''" }

    $matlabJobs += "jobs{end+1} = struct('pfn', $physio, 'efn', $eye);"
}

$matlabChannelList = ($Channels | ForEach-Object { "'" + (Escape-MatlabString $_) + "'" }) -join ", "

$PspmAdd = ""
if ($PspmPath -ne "") {
    $PspmAdd = "addpath(" + (Quote-MatlabString $PspmPath) + ");"
} elseif ($PspmPathFile -ne "") {
    $p = Quote-MatlabString $PspmPathFile
    $PspmAdd = @"
raw = strrep(fileread($p), '"', '');
pspm_path = strrep(strtrim(raw), '\', '/');
addpath(pspm_path);
"@
}

$parpoolCode = if ($Workers -gt 1) {
@"
try
    parpool('local', $Workers);
catch ME
    disp('Parallel pool already running or failed to start.');
    disp(ME.message);
end
"@
} else { "" }

$loopCode = if ($Workers -gt 1) {
@"
parfor i = 1:numel(jobs)
    inarg = struct();
    inarg.pfn = jobs{i}.pfn;
    inarg.site = '$Site';
    inarg.chan = channel_list;
    inarg.trackdist = $TrackDist;
    inarg.overwrite = $overwriteVal;

    if isfield(jobs{i}, 'efn') && ~isempty(jobs{i}.efn)
        inarg.efn = jobs{i}.efn;
    end

    fprintf('=== [%d/%d] Processing %s ===\n', i, numel(jobs), inarg.pfn);

    try
        outnames = calinet_import_smr(inarg);
        results{i} = outnames;
    catch ME
        fprintf(2, 'FAILED: %s\n', inarg.pfn);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        results{i} = {[], []};
    end
end
"@
} else {
@"
for i = 1:numel(jobs)
    inarg = struct();
    inarg.pfn = jobs{i}.pfn;
    inarg.site = '$Site';
    inarg.chan = channel_list;
    inarg.trackdist = $TrackDist;
    inarg.overwrite = $overwriteVal;

    if isfield(jobs{i}, 'efn') && ~isempty(jobs{i}.efn)
        inarg.efn = jobs{i}.efn;
    end

    fprintf('=== [%d/%d] Processing %s ===\n', i, numel(jobs), inarg.pfn);

    try
        outnames = calinet_import_smr(inarg);
        results{i} = outnames;
    catch ME
        fprintf(2, 'FAILED: %s\n', inarg.pfn);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        results{i} = {[], []};
    end
end
"@
}

$MatlabBatch = @"
cd($(Quote-MatlabString $TmpRoot));
addpath($(Quote-MatlabString $TmpRoot));
$PspmAdd

$parpoolCode

channel_list = {$matlabChannelList};

jobs = {};
$(($matlabJobs -join "`r`n"))

results = cell(numel(jobs), 1);

$loopCode

disp('Batch import completed.');
exit;
"@

Set-Content -Path $MatlabBatchFile -Value $MatlabBatch -Encoding UTF8

Write-Host "------------------------------------------------------------"
Write-Host "CALINET batch import"
Write-Host "------------------------------------------------------------"
Write-Host " InputDir       : $(To-MatlabPath $InputDirResolved)"
Write-Host " Site           : $Site"
Write-Host " MATLAB exe     : $MatlabExe"
Write-Host " Workers        : $Workers"
Write-Host " Overwrite      : $Overwrite"
Write-Host " Channels       : $($Channels -join ', ')"
Write-Host " SMR files      : $($jobs.Count)"
Write-Host " Eye matching   : $MatchEyeFiles"
Write-Host " EyeDir         : $(To-MatlabPath $EyeDirResolved)"
Write-Host " EyeExtension   : $EyeExtension"
Write-Host " Temp dir       : $(To-MatlabPath $TmpRoot)"
Write-Host " PsPM path      : $(To-MatlabPath $PspmPath)"
Write-Host "------------------------------------------------------------"

if ($DebugMode) {
    Write-Host ""
    Write-Host "DEBUG MODE - MATLAB WILL NOT RUN"
    Write-Host "------------------------------------------------------------"
    Write-Host "MATLAB function file:"
    Write-Host "  $MatlabFuncFile"
    Write-Host ""
    Write-Host "MATLAB batch file:"
    Write-Host "  $MatlabBatchFile"
    Write-Host ""
    Write-Host "Generated MATLAB batch code:"
    Write-Host "------------------------------------------------------------"
    Write-Host $MatlabBatch
    Write-Host "------------------------------------------------------------"
    exit 0
}

& $MatlabExe -batch "run('$(To-MatlabPath $MatlabBatchFile)')" | Tee-Object -FilePath $LogFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: MATLAB exited with code $LASTEXITCODE"
    Write-Host "See log: $LogFile"
    exit $LASTEXITCODE
}

Write-Host "Done."
Write-Host "Log: $LogFile"