% Reads pl2 plexon file and saves spikes and messages
%
% Revision history:
% v1.0 - September 6, 2016 Initial script written.
% v1.1 - October 25, 2016 Added channel names; removed event detection (due to pl2 problem)
% Paths provided link directly to the file
%
% Donatas Jonikaitis


function plexon_spikes_v11(path_in_full, path_out_full)


%===================
% Import the file
% This is based on standard plexon provided code, minimal changes


% Basic info about opened file
[f_name, Version, ~, ~, ~, ~, ~, ~, ~, ~, ~, Duration, DateTime] = plx_information(path_in_full);

% tscounts - timestamp counts
% wfcounts - waveform counts
% evcounts - event counts
[tscounts, wfcounts, evcounts, contcounts] = plx_info(f_name,1);


%% Spikes

% Number of units (row 1 - unsorted, row 2 - unit a)
% Number of channels + 1 (plexon property, corrected it in the code)
[nunits1, nchannels1] = size(tscounts);
nchannels1 = nchannels1-1; % Correct channel count
nunits1 = nunits1-1; % Correct unit count

% Creat structures for spike timing and channel names
spike_ts = cell(nunits1, nchannels1);

% We will read in the timestamps of all [units,channels] into a two-dim cell
% array named spike_ts, with each cell containing the timestamps for a [unit,channel].
% Note that spike_ts second dim is indexed by the 1-based channel number.
for i_unit = 0:nunits1   % To start with unsorted channels [0:numunits1-1] to start with sorted units [0:numunits1-1]
    for i_ch = 1:nchannels1
        if ( tscounts(i_unit+1, i_ch+1) > 0 ) % +1 as channel n is saved as n+1 and unit m is saved as m+1 in tscounts matrix
            % Get the timestamps for this channel and unit
            [~, spike_ts{i_unit+1,i_ch}] = plx_ts(f_name, i_ch, i_unit);
        end
    end
end

[~, channel_names] = plx_chan_names(f_name);


%% Save data into structure

% Spikes
plexon.spike_ts = spike_ts;
plexon.channel_names = channel_names;

% Other
plexon.file_name = path_in_full;
plexon.pl_off_sorter_version = Version;
plexon.exp_dur = Duration;
plexon.date = DateTime;


%% Save data into folder

save(eval('path_out_full'), 'plexon')


