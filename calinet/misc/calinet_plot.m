function calinet_plot(data, E, fs)
%CALINET_PLOT Plot EDA time series with colored onset sticks.
%
%   sparseEDA_plot(driver, E)
%
% Inputs
%   driver : 1xN or Nx1 numeric vector
%   E      : table with variables:
%              - time : onset times in seconds
%              - name : stimulus names (string, char, or categorical)
%   fs     : sampling rate (Hz)
%
% Assumptions
%   - Sampling rate is fixed at 8 Hz
%
% Stimulus color mapping
%   CSpr : red
%   CSpu : orange
%   CSm  : blue
%   USp  : dark red

    % Validate inputs
    if ~isnumeric(data) || ~isvector(data)
        error('driver must be a numeric vector.');
    end

    if ~istable(E)
        error('E must be a table.');
    end

    if ~all(ismember({'time','name'}, E.Properties.VariableNames))
        error('E must contain variables named "time" and "name".');
    end

    % Ensure column/row consistency
    data = data(:)';   % row vector
    t = (0:numel(data)-1) / fs;

    % Normalize event names to string
    evt_names = string(E.name);
    evt_times = E.time;

    % Color map
    stim_colors = containers.Map( ...
        {'CSpr','CSpu','CSm','USp','USm','USo'}, ...
        {'#d62728','#ff7f0e','#1f77b4','#7f0000','#17becf','#7f7f7f'} );

    % Legend/display order
    stim_order = {'CSpr','CSpu','CSm','USp'};
    % Figure
    fig = figure('Color', [1 1 1]);
    fig.Units = 'inches';
    fig.Position = [1 1 14 3.54];   % [x y width height]

    % Prevent MATLAB from resizing on save
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 14 3.54];
    fig.PaperSize = [14 3.54];

    % Axes
    ax = axes('Parent', fig);
    hold(ax, 'on');
    ax.Color = [1, 1, 1];
    box(ax, 'off');

    % Driver trace
    plot(ax, t, data, 'Color', [0.5 0.5 0.5], 'LineWidth', 2.5);

    % Labels
    xlabel(ax, 'time (s)', 'FontSize', 10);
    ylabel(ax, 'amplitude [uS]', 'FontSize', 10);

    % Limits and stick placement
    xlim(ax, [0 max(t)]);
    yl = ylim(ax);
    stick_y0 = yl(1);
    stick_y1 = yl(1) + 0.28 * (yl(2) - yl(1));

    % Plot onset sticks
    hleg = gobjects(numel(stim_order), 1);

    for i = 1:numel(stim_order)
        stim = stim_order{i};
        idx = evt_names == stim;
        times = evt_times(idx);

        if ~isKey(stim_colors, stim)
            continue;
        end

        col = hex2rgb_local(stim_colors(stim));
        alpha = 1;
        bg = ax.Color;
        
        col = alpha * col + (1 - alpha) * bg;

        for k = 1:numel(times)
            line( ...
                ax, ...
                [times(k) times(k)], ...
                [stick_y0 stick_y1], ...
                'Color', col, ...
                'LineWidth', 2 ...
            );
        end

        % Dummy line for legend
        hleg(i) = plot(ax, nan, nan, '-', 'Color', col, 'LineWidth', 8);
    end

    % Legend
    legend(ax, hleg, stim_order, ...
        'Location', 'northeastoutside', ...
        'Box', 'off', ...
        'FontSize', 10);

    % Style mapped from Python defaults
    ax.FontName = 'Helvetica';
    ax.FontSize = 14;
    ax.XAxis.FontSize = 10;
    ax.YAxis.FontSize = 10;
    ax.LineWidth = 0.5;
    ax.TickDir = 'out';
    ax.TickLength = [0 0];
    ax.XColor = [0.2 0.2 0.2];
    ax.YColor = [0.2 0.2 0.2];

end

function rgb = hex2rgb_local(hex)
%HEX2RGB_LOCAL Convert hex color string to RGB triple in [0,1].
    hex = char(hex);
    if ~isempty(hex) && hex(1) == '#'
        hex = hex(2:end);
    end
    rgb = sscanf(hex, '%2x%2x%2x', [1 3]) / 255;
end