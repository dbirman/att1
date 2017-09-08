% Reads pl2 plexon file and saves spikes and messages
%
% Revision history:
% v1.0 - October 27, 2016 
% Paths provided link directly to the file
%
% Donatas Jonikaitis


function plexon_events_v10(path_in_full, path_out_full)


%===================
% Import the file
% This is based on standard plexon provided code, minimal changes


% Basic info about opened file
[f_name, Version, ~, ~, ~, ~, ~, ~, ~, ~, ~, Duration, DateTime] = plx_information(path_in_full);

% tscounts - timestamp counts
% wfcounts - waveform counts
% evcounts - event counts
[tscounts, wfcounts, evcounts, contcounts] = plx_info(f_name,1);


%% Events


% Get the event channel map to make any sense of these
[n,evchans] = plx_event_chanmap(f_name);
[n,evnames] = plx_event_names(f_name);
pl2=PL2ReadFileIndex(f_name);

% If events exist
if n > 0
    
    % Initialize cell
    event_ts = cell(1, n); % Event timestamps
    event_n = cell(1, n); % Event number
    
    for i_ev = 1:length(evchans)
        
        if evcounts(i_ev)>0
            
            evch = evnames(i_ev,:); % Use name, not channel number due to plexon bug in pl2 export files
            
            if strcmp ( evch, 'STROBED' ) % Strobed words
                [event_n{i_ev}, event_ts{i_ev}, svStrobed] = plx_event_ts(f_name, evch);
            else % Non-strobed
                [event_n{i_ev}, event_ts{i_ev}, ~] = plx_event_ts(f_name, evch);
            end
        end
        
    end
    
end

% Get event names in plexon
[nev,event_names] = plx_event_names(f_name);


%% Save data into structure

% Events
plexon.event_ts = event_ts;
plexon.event_names = event_names;

% Other
plexon.file_name = path_in_full;
plexon.pl_off_sorter_version = Version;
plexon.exp_dur = Duration;
plexon.date = DateTime;


%% Save data into folder

save(eval('path_out_full'), 'plexon')


