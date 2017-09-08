% Reads pl2 plexon file and saves spikes and messages
%
% Revision history:
% v1.0 - October 31, 2016 Reading out analog channels;
% Paths provided link directly to the file
% Provide as many channels as you wish
%
% Donatas Jonikaitis


function plexon_analog_v10(path_in_full, path_out_full, var1)

% For debugging purposes use a path to a real file
% var1 = 'AI01';

%===================
% Import the file
% This is based on standard plexon provided code, minimal changes


% Basic info about opened file
[f_name, Version, ~, ~, ~, ~, ~, ~, ~, ~, ~, Duration, DateTime] = plx_information(path_in_full);

% tscounts - timestamp counts
% wfcounts - waveform counts
% evcounts - event counts
% contcounts - samples per continous channel
[~, ~, ~, contcounts] = plx_info(f_name,1);


%% Analog data

% get the a/d data into a cell array also.
% This is complicated by channel numbering.
% The number of samples for analog channel 0 is stored at slowcounts(1).
% Note that analog ch numbering starts at 0, not 1 in the data, but the
% 'allad' cell array is indexed by ich+1

adfreq = cell(1); nad = cell(1); tsad = cell(1); allad=cell(1);
for i=1:length(var1)
    ch_n = var1{i};
    [adfreq{i}, nad{i}, tsad{i}, fnad{i}, allad{i}] = plx_ad(f_name, ch_n);
end


%% Save data into structure

% Spikes
plexon.analog = allad;
plexon.freq = adfreq;
plexon.total_data_points = nad;
plexon.fragment_time_stamps = tsad;
plexon.data_points_per_fragment = fnad;


% Other
plexon.file_name = path_in_full;
plexon.pl_off_sorter_version = Version;
plexon.exp_dur = Duration;
plexon.date = DateTime;


%% Save data into folder

save(eval('path_out_full'), 'plexon')


