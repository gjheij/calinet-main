function make_synth_calinet_bids(root_dir, overwrite)
% This function generates a simulated SCR time series with known event timing
% and writes it to a BIDS-style folder structure under:
%   root_dir/sub-<sub>/ses-<ses>/physio/
%
% What gets written (inside the physio folder):
%   - *_recording-scr_physio.tsv.gz
%   - *_recording-scr_physio.json
%   - *_events.tsv.gz
%   - *_events.json
%
% Trial schedule (acquisition-like):
%   - 16 CS trials total: 8 CS+ and 8 CS-
%   - 6/8 CS+ reinforced (US events), i.e. 75% reinforcement
%   - CS duration is 8s
%   - ITI gap (CS offset -> next CS onset) is sampled uniformly in [gap_min, gap_max]
%   - US occurs us_delay seconds after each reinforced CS+ onset
%
% Signal generation:
%   - Impulses at event onsets are weighted by condition amplitudes
%   - Impulses are convolved with PsPM's canonical SCRF (pspm_bf_scrf_f)
%   - Adds a slow drift and Gaussian noise
%
% Usage:
%   make_synth_calinet_bids('C:\path\to\SyntheticSCR');
%   make_synth_calinet_bids('C:\path\to\SyntheticSCR', true); (optional)

    if nargin < 2 || isempty(overwrite)
        overwrite = true;
    end

    % Add overwrite option if <root_dir> exists
    if exist(root_dir, 'dir') == 7
        if overwrite
            fprintf("Overwriting existing folder: %s\n", root_dir);
            rmdir(root_dir, 's');
        else
            error('Target folder already exists: %s (set overwrite=true to replace it)', root_dir);
        end
    end

    %% ---- parameters ----
    sub  = "Synth01";
    task = "acquisition";

    sr = 2000;               % sampling rate (Hz)
    td = 1/sr;

    % Stimulus timing parameters
    cs_dur   = 8.0;         % seconds
    us_dur   = 0.5;         % seconds
    us_delay = 7.5;         % seconds after CS+ onset (reinforced CS+)

    % ISI 9-15 seconds -> interpret as ITI gap (offset->onset)
    gap_min = 9;
    gap_max = 15;
    baseline_leadin = 10;   % seconds before first CS

    % Acquisition composition
    n_trials = 16;
    n_csp    = 8;           % CS+
    n_csm    = 8;           % CS-
    n_us     = 6;           % number of reinforced CS+ (USp count)

    % Amplitudes
    amp_csm  = 0.5;
    amp_cspu = 1.0;
    amp_cspr = 1.2;
    amp_usp  = 1.5;

    noise_sd  = 0.03;
    drift_amp = 0.05;

    %% ---- generate CALINET-like acquisition schedule ----
    rng(1);  % reproducible

    % Trial order with exactly 8 CS+ and 8 CS-
    trial_type = [repmat("CSp", n_csp, 1); repmat("CSm", n_csm, 1)];
    trial_type = trial_type(randperm(numel(trial_type)));

    % CS onsets
    cs_on = zeros(n_trials, 1);
    cs_on(1) = baseline_leadin;
    for i = 2:n_trials
        gap = gap_min + (gap_max - gap_min) * rand();
        cs_on(i) = cs_on(i-1) + cs_dur + gap;
    end

    % Split onsets into CS- and CS+
    csm_on = cs_on(trial_type == "CSm");
    csp_on = cs_on(trial_type == "CSp");

    % Choose 6 of 8 CS+ to be reinforced
    reinforced = false(size(csp_on));
    reinforced(randperm(numel(csp_on), n_us)) = true;

    cspr_on = csp_on(reinforced);   % reinforced CS+
    cspu_on = csp_on(~reinforced);  % unreinforced CS+

    % US after each reinforced CS+
    usp_on = cspr_on + us_delay;

    % Set total recording duration (include tail for decay)
    T = ceil(max([cs_on; usp_on]) + 30);
    t = (0:td:T-td)';
    N = numel(t);

    %% ---- build driver and convolve ----
    u = zeros(N,1);

    add_impulses(csm_on,  amp_csm);
    add_impulses(cspu_on, amp_cspu);
    add_impulses(cspr_on, amp_cspr);

    % include US-driven responses
    add_impulses(usp_on,  amp_usp);

    [scrf, ~, ~] = pspm_bf_scrf_f(td);
    scr = conv(u, scrf);
    scr = scr(1:N);

    drift = drift_amp * sin(2*pi*t/120) + 0.5*drift_amp*cos(2*pi*t/70);
    noise = noise_sd * randn(N,1);

    signal = scr + drift + noise;
    
    %% ---- write BIDS structure ----
    ses_dir    = fullfile(root_dir, "sub-" + sub);
    physio_dir = fullfile(ses_dir, "physio");
    if ~exist(physio_dir, "dir"), mkdir(physio_dir); end

    base = sprintf("sub-%s_task-%s", sub, task);

    ds = struct( ...
        "Name","Synthetic SCR validation dataset", ...
        "BIDSVersion","1.8.0", ...
        "DatasetType","raw" ...
    );
    write_json(fullfile(root_dir, "dataset_description.json"), ds);

    fid = fopen(fullfile(root_dir,"participants.tsv"),"w");
    fprintf(fid, "participant_id\nsub-%s\n", sub);
    fclose(fid);

    pj = struct();
    pj.participant_id = struct("Description","Participant identifier");
    write_json(fullfile(root_dir,"participants.json"), pj);

    %% ---- physio TSV (gzipped) ----
    timestamp = (0:numel(signal)-1)' / sr;   % seconds, starts at 0
    scr_table = [timestamp, signal];
    
    scr_tsv = fullfile(physio_dir, sprintf("%s_recording-scr_physio.tsv", base));
    fid = fopen(scr_tsv, "w");
    fprintf(fid, "%.8f\t%.8f\n", scr_table.');   % timestamp \t scr
    fclose(fid);
    
    gzip(scr_tsv);
    delete(scr_tsv);

    %% ---- physio JSON ----
    scr_json = struct();
    scr_json.Columns = {'timestamp', 'scr'};
    scr_json.Manufacturer = "Synthetic";
    scr_json.ManufacturersModelName = "N/A";
    scr_json.DeviceSerialNumber = "N/A";
    scr_json.SamplingFrequency = sr;
    scr_json.SoftwareVersion = "N/A";
    scr_json.StartTime = 0;
    scr_json.PhysioType = "generic";
    
    scr_json.timestamp = struct( ...
        "LongName", "Time", ...
        "Description", "a continuously increasing identifier of the sampling time registered by the device", ...
        "Origin", "System startup", ...
        "Units", "s" ...
    );
    
    scr_json.scr = struct( ...
        "Description", "Synthetic SCR Recording", ...
        "SCRCouplerType", [], ...
        "SCRCouplerVoltage", [], ...
        "Placement", [], ...
        "Units", "uS", ...
        "MeasureType", "EDA-total" ...
    );

    write_json(strrep(scr_tsv, ".tsv", ".json"), scr_json);

    %% ---- events.tsv (then gzip) ----
    % Acquisition events
    onset_acq = [csm_on; cspu_on; cspr_on; usp_on];

    duration_acq = [
        cs_dur*ones(size(csm_on));
        cs_dur*ones(size(cspu_on));
        cs_dur*ones(size(cspr_on));
        us_dur*ones(size(usp_on))
    ];

    event_type_acq = [
        repmat("CSm",  numel(csm_on), 1);
        repmat("CSpu", numel(cspu_on),1);
        repmat("CSpr", numel(cspr_on),1);
        repmat("USp",  numel(usp_on), 1)
    ];

    stim_acq = [
        repmat("square",  numel(csm_on), 1);
        repmat("diamond", numel(cspu_on),1);
        repmat("diamond", numel(cspr_on),1);
        repmat("shock",   numel(usp_on), 1)
    ];

    task_acq = repmat("acquisition", numel(onset_acq), 1);

    % Combine all events (habituation removed; extinction removed)
    onset = onset_acq;
    duration = duration_acq;
    event_type = event_type_acq;
    stim = stim_acq;
    task_name = task_acq;

    % sort by onset
    [onset, ix] = sort(onset);
    duration = duration(ix);
    event_type = event_type(ix);
    stim = stim(ix);
    task_name = task_name(ix);

    events_tsv = fullfile(physio_dir, sprintf("%s_events.tsv", base));
    fid = fopen(events_tsv, "w");
    fprintf(fid, "onset\tduration\tevent_type\tstimulus_name\ttask_name\n");
    for i = 1:numel(onset)
        fprintf(fid, "%.3f\t%.3f\t%s\t%s\t%s\n", onset(i), duration(i), event_type(i), stim(i), task_name(i));
    end
    fclose(fid);

    %% ---- events.json ----
    evj = struct();
    evj.onset = struct("Description","Event onset in seconds");
    evj.duration = struct("Description","Event duration in seconds");
    evj.event_type = struct("Description","Event type label (CSm/CSpu/CSpr/USp)");
    evj.stimulus_name = struct("Description","Stimulus name");
    evj.task_name = struct("Description","Task/phase label (habituation/acquisition/extinction)");
    write_json(fullfile(physio_dir, sprintf("%s_events.json", base)), evj);

    %% ---- events.json ----
    evj = struct();
    evj.onset = struct("Description","Event onset in seconds");
    evj.duration = struct("Description","Event duration in seconds");
    evj.event_type = struct("Description","Event type label (CSm/CSpu/CSpr/USp)");
    evj.stimulus_name = struct("Description","Stimulus name");
    write_json(fullfile(physio_dir, sprintf("%s_events.json", base)), evj);

    fprintf("Synthetic dataset created at %s\n", root_dir);
    print_tree(root_dir);
    
    function add_impulses(onsets, amp)
        idx = round(onsets*sr) + 1;
        idx = idx(idx>=1 & idx<=N);
        u(idx) = u(idx) + amp;
    end
end

function write_json(path, S)
    try
        txt = jsonencode(S, "PrettyPrint", true);
    catch
        txt = jsonencode(S);
    end
    fid = fopen(path, "w");
    fwrite(fid, txt, "char");
    fclose(fid);
end

function print_tree(folder, prefix)
    if nargin < 2
        prefix = "";
    end

    files = dir(folder);
    files = files(~ismember({files.name}, {'.','..'}));

    for i = 1:numel(files)
        name = files(i).name;
        fullpath = fullfile(folder, name);

        fprintf("%s|-- %s\n", prefix, name);

        if files(i).isdir
            print_tree(fullpath, prefix + "    ");
        end
    end
end