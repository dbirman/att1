% Script running main experiment file. Initializes and closes all devices.
% V2.1 Update as of May 16, 2017

close all;
clear all;
clc;

fprintf ('\nRun experiment file used is V2.0, update as of April 19, 2017\n\n')
 
global expsetup
global ni

%% General settings to run the code

expsetup.general.expname = 'att1';
expsetup.general.exp_location = 'dj'; % 'dj'; 'mbox'
expsetup.general.debug = 2; % 0: default; 1: reward off, eyelink off; 2: reward off, eyelink off, display transparent

% Devices and routines
expsetup.general.record_plexon = 0;  % 0 - no recording; 1 - yes recording;
expsetup.general.plexon_online_spikes = 0;  % 0 - no; 1 - yes;
expsetup.general.stimulator_input_on = 0; % 0 - no input, 1 - yes input
expsetup.general.recordeyes = 0; % 0 - no recording; 1 - yes recording; 2 - ask experimenter
expsetup.general.reward_on = 0; % 0 - no reward, 1 - reward
expsetup.general.arduino_on = 0; % 0 - no arduino, 1 - yes arduno
expsetup.general.psychaudio = 0; % 0 - no psych audio; 1 - yes psych audio
expsetup.general.convert_edf = 0; % 1 - convert to .asc & .dat

% Displays and keyboard settings 
expsetup.general.hidecursor = 1; % 1 - show cursor; 2 - hide cursor;
expsetup.general.empty_key = 'q'; % % Which key is reserved for as empty 
expsetup.general.quit_key = 'x'; % Quits the task, Windows 'esc', Apple 'ESCAPE'
expsetup.general.image_database = 0; % Do we need to load image database?
expsetup.general.filename_auto = 2; % 1 - manual filename selection; 2 - automatic
expsetup.general.code_instruction_on = 1; % 0 - no instruction; 1 - instruction;
expsetup.general.trials_before_saving = 10; % How many trials to run before saving data

% Plexon events
expsetup.general.plex_event_start = 1; % Code saved as trial start
expsetup.general.plex_event_end = 2; % Code saved as trial start
expsetup.general.plex_trial_timeout_sec = 30; % How many seconds before plexon stops checking data
expsetup.general.plex_data_rate = []; % At which rate data is collected (40000 Hz), determined during tcpip connection

% Get subject name
if isfield(expsetup.general, 'debug')
    if expsetup.general.debug >0
        expsetup.general.subject_id = 'xxx'; % Enter subject initials
    else
        expsetup.general.subject_id = input('Enter subject name:  ', 's');
    end
else
    expsetup.general.subject_id = input('Enter subject name:  ', 's');
end


% Get file name for computer settings
if strcmp (expsetup.general.exp_location, 'dj')
    expsetup.general.code_computer_setup = sprintf('%s_computer_settings_dj_v22', expsetup.general.expname); % Path to file containing computer settings
elseif strcmp (expsetup.general.exp_location, 'mbox')
    expsetup.general.code_computer_setup = sprintf('%s_computer_settings_mbox_v22', expsetup.general.expname); % Path to file containing computer settings
elseif strcmp (expsetup.general.exp_location, 'dan')
    expsetup.general.code_computer_setup = sprintf('%s_computer_settings_dan_v22', expsetup.general.expname); % Path to file containing computer settings
elseif strcmp (expsetup.general.exp_location, 'unknown')
else
    error ('Undefined exp location - could not find matching computer settings file')
end

% Plexon events
if expsetup.general.plexon_online_spikes == 1 % Get channel number used in the recording
    expsetup.general.plex_num_act_channels = input('How many channels are recorded (for online spike monitoring): ');
end

% Get recording coordinates
if expsetup.general.record_plexon ==1
    a = input('Select: front well = 1; back well = 2; both wells = 3): ');
    if a==1 || a==3
        expsetup.general.microdrive_coord_bottom_front = input('Front well, bottom coordinate: ', 's');
        expsetup.general.microdrive_coord_top_front = input('Front well,  top coordinate: ', 's');
        expsetup.general.microdrive_gt_depth_front = input('Front well, guide tube zero: ', 's');
        expsetup.general.microdrive_neuro_depth_front = input('Front well, depth to recorded neurons from guide tube zero: ', 's');
    end
    if a==2 || a==3
        expsetup.general.microdrive_coord_bottom_back = input('Back well, bottom coordinate: ', 's');
        expsetup.general.microdrive_coord_top_back = input('Back well, top coordinate: ', 's');
        expsetup.general.microdrive_gt_depth_back = input('Back well, guide tube zero: ', 's');
        expsetup.general.microdrive_neuro_depth_back = input('Back well, depth to recorded neurons from guide tube zero: ', 's');
    end
end


% Reward and eye recording
if expsetup.general.debug==1
    expsetup.general.reward_on = 0; % 0 - no reward, 1 - reward
    expsetup.general.recordeyes = 0; % 0 - no recording; 1 - yes recording; 2 - ask experimenter
elseif expsetup.general.debug==2
    PsychDebugWindowConfiguration(0.5);
    expsetup.general.reward_on = 0; % 0 - no reward, 1 - reward
    expsetup.general.recordeyes = 0; % 0 - no recording; 1 - yes recording; 2 - ask experimenter
end

expsetup.general.code_settings = sprintf('%s_settings', expsetup.general.expname); % Path to file containing trial settings
expsetup.general.code_trial = sprintf('%s_trial', expsetup.general.expname); % Path to file containing trial presentation


%====================
% Calibration settings
expsetup.eyecalib.backgroundcolor = [127, 127, 127];
expsetup.eyecalib.foregroundcolor = 'random';
% expsetup.eyecalib.foregroundcolor = [20, 20, 20];
expsetup.eyecalib.target_shape = 'circle'; % circle; square
expsetup.eyecalib.target_size_deg = [0, 0, 1, 1];
expsetup.eyecalib.target_positions = 0:90:359;
expsetup.eyecalib.target_radius = 0.7;
expsetup.eyecalib.fixation_cond_blink = 1; % Is it blinkig target?
expsetup.eyecalib.fixation_cond_ring = 0; % To show the ring?
expsetup.eyecalib.reward_ms = 100; % How long reward lasts


%% Initialize all computer specific settings

eval(expsetup.general.code_computer_setup)


%% Add the folder with experimental code to the path

expsetup.general.directory_experiment_code = [expsetup.general.directory_baseline_code, expsetup.general.expname, '/experiment/'];
if isdir (expsetup.general.directory_experiment_code)
    addpath (expsetup.general.directory_experiment_code);
else
    error ('Experiment name specified does not exist')
end


%% Create directories for data recording (main file)

% Initialize folder where psychtooblox data is stored
expsetup.general.directory_data_psychtoolbox = [expsetup.general.directory_baseline_data, expsetup.general.expname, '/data_psychtoolbox/', expsetup.general.subject_id, '/'];
if ~isdir (expsetup.general.directory_data_psychtoolbox)
    mkdir(expsetup.general.directory_data_psychtoolbox);
end

runexp_session_ini_v17 (expsetup.general.directory_data_psychtoolbox);

% Initialize folder where psychtoolbox data of the current session will be stored
expsetup.general.directory_data_psychtoolbox_subject = [expsetup.general.directory_data_psychtoolbox, expsetup.general.subject_filename, '/'];
if isdir (expsetup.general.directory_data_psychtoolbox_subject)
    rmdir(expsetup.general.directory_data_psychtoolbox_subject, 's'); % Remove old contents
    mkdir(expsetup.general.directory_data_psychtoolbox_subject); % Add new folder
    fprintf('\nOver-writing the directory %s \n', expsetup.general.subject_filename)
else
    mkdir(expsetup.general.directory_data_psychtoolbox_subject);
    fprintf('\nCreated a new directory %s \n', expsetup.general.subject_filename)
end

% Initialize folder where eyelink data is stored
expsetup.general.directory_data_eyelink_edf = [expsetup.general.directory_baseline_data, expsetup.general.expname, '/data_eyelink_edf/', expsetup.general.subject_id, '/'];
expsetup.general.directory_data_eyelink_asc = [expsetup.general.directory_baseline_data, expsetup.general.expname, '/data_temp1/', expsetup.general.subject_id, '/'];
if ~isdir (expsetup.general.directory_data_eyelink_edf)
    mkdir(expsetup.general.directory_data_eyelink_edf);
end
if ~isdir (expsetup.general.directory_data_eyelink_asc)
    mkdir(expsetup.general.directory_data_eyelink_asc);
end

% Initialize folder where eyelink data of current session will be stored
expsetup.general.directory_data_eyelink_edf_subject = [expsetup.general.directory_data_eyelink_edf, expsetup.general.subject_filename, '/'];
expsetup.general.directory_data_eyelink_asc_subject = [expsetup.general.directory_data_eyelink_asc, expsetup.general.subject_filename, '/'];
if isdir (expsetup.general.directory_data_eyelink_edf_subject)
    rmdir(expsetup.general.directory_data_eyelink_edf_subject, 's'); % Remove old contents
    mkdir(expsetup.general.directory_data_eyelink_edf_subject); % Add new folder
else
    mkdir(expsetup.general.directory_data_eyelink_edf_subject);
end
if isdir (expsetup.general.directory_data_eyelink_asc_subject)
    rmdir(expsetup.general.directory_data_eyelink_asc_subject, 's'); % Remove old contents
    mkdir(expsetup.general.directory_data_eyelink_asc_subject); % Add new folder
else
    mkdir(expsetup.general.directory_data_eyelink_asc_subject);
end


%% Load image database

if expsetup.general.image_database==1
    expsetup.general.directory_image_database = [expsetup.general.directory_baseline_code, 'image_database\'];
    if isdir (expsetup.general.directory_image_database)
        addpath(expsetup.general.directory_image_database); % Change to directory
        expsetup.general.image_listing = dir(expsetup.general.directory_image_database);
    end
end


%% Initialize sounds

if expsetup.general.psychaudio == 1
    runexp_audio_ini_v10;
end


%% Initialize arduino

if expsetup.general.arduino_on==1
    try
        expsetup.general.arduino_session = arduino('com3', 'Mega2560');
    catch
        error('Failed to initialize arduino setup')
    end
end


%% Initialize reward

if expsetup.general.reward_on == 1
    if isfield (expsetup.ni_daq, 'device_name')
        
        % Create session
        devices = daq.getDevices; % Just for info
        temp = get(daq.getVendors());
        ni.session_reward = daq.createSession(temp.ID);
        
        % Settings
        device_name = expsetup.ni_daq.device_name;
        ni_ch_id = expsetup.ni_daq.reward_channel_id;
        ni_m_type = expsetup.ni_daq.reward_measurement_type;
        ni.session_reward.Rate = expsetup.ni_daq.reward_rate;
        
        % Add analog output channel
        addAnalogOutputChannel(ni.session_reward, device_name, ni_ch_id, ni_m_type);
        
    else
        error('ni-daq device name for reward system not specified');
    end
end



%% Initialize digital output to plexon events


if expsetup.general.record_plexon == 1
    if isfield (expsetup.ni_daq, 'device_name')
        
        % Create session
        devices = daq.getDevices; % Just for info
        temp = get(daq.getVendors());
        ni.session_plexon_events = daq.createSession(temp.ID);
        
        % Add digintal channel
        % Bits 0-15
        device_name = expsetup.ni_daq.device_name;
        addDigitalChannel(ni.session_plexon_events, device_name, 'Port0/Line8', 'OutputOnly'); % Bit 0 (Pin1)
        addDigitalChannel(ni.session_plexon_events, device_name, 'Port0/Line17:31', 'OutputOnly'); % Bit 1-15 (Pin 2-16)
        expsetup.ni_daq.digital_channel_total_bits = size(ni.session_plexon_events.Channels,2);
        
        % Strobe
        addDigitalChannel(ni.session_plexon_events, device_name, 'Port0/Line16', 'OutputOnly'); % STROBE (Pin22)
        expsetup.ni_daq.digital_channel_STROBE = size(ni.session_plexon_events.Channels,2);
        
        % RSTART
        addDigitalChannel(ni.session_plexon_events, device_name, 'Port0/Line15', 'OutputOnly'); % RSTART (Pin24)
        expsetup.ni_daq.digital_channel_RSTART = size(ni.session_plexon_events.Channels,2);
        
        % Check wether adding channel numbers worked
        expsetup.ni_daq.digital_channel_total = size(ni.session_plexon_events.Channels,2);
        if expsetup.ni_daq.digital_channel_total == 0
            error ('Failed to initialize digital channels for plexon triggers')
        end
        
        % Save number of channels used
        fprintf ('\nRunning experiment with of total %i National Instruments card analog & digital channels\n\n\n',  expsetup.ni_daq.digital_channel_total)
        
    else
        error('ni-daq device name not specified');
    end
end


%% Initialize connection with plexon computer.
% This code will trigger constant tcpip monitoring

if expsetup.general.plexon_online_spikes == 1
    
    % Connection details
    ip_address = expsetup.tcpip.plex_address;
    socket = expsetup.tcpip.plex_port;
    
    if isfield (expsetup.tcpip, 'plex_address')
        
        %============
        % 1st connection
        %============
        
        % Initialize data matrix which will transfer crucial info to plexon
        el1 = expsetup.tcpip.data_file_size;
        data_mat = NaN(el1,1);
        data_mat(2:2:8) = -1;
        data_mat(1) = expsetup.general.plex_event_start;
        data_mat(3) = expsetup.general.plex_event_end;
        data_mat(5) = expsetup.general.plex_num_act_channels;
        data_mat(7) = expsetup.general.plex_trial_timeout_sec;
        data_prop = whos('data_mat');
        
        % Initialize client
        tserver = tcpip(ip_address, socket);
        set(tserver, 'OutputBufferSize', data_prop.bytes)
        fopen(tserver);
        fprintf('\nTCPIP connection established\n')
        
        % Write data & close
        fwrite(tserver, data_mat, data_prop.class);
        fclose (tserver);
        delete (tserver);
        fprintf('Experimental setup instructions sent successfully\n')
        fprintf('Will reverse connection direction\n')
        
        %============
        % 2nd connection
        %============
        % Before reversing connection initialize new data file
        el1 = expsetup.tcpip.data_file_size;
        data_mat = NaN(el1,1);
        data_prop = whos('data_mat');
        
        % Now reverse the connection
        fprintf('\nWaiting for connection with the server\n')
        tclient = tcpip(ip_address, socket, 'NetworkRole', 'server');
        set(tclient, 'InputBufferSize', data_prop.bytes)
        fopen(tclient);
        fprintf('Connection established\n')
        
        % Check for data. Allow a timeout.
        t1 = cputime;
        t2 = cputime;
        t_out = expsetup.general.plex_trial_timeout_sec;
        count = tclient.BytesAvailable;
        while count~=data_prop.bytes && t2-t1<t_out
            count = tclient.BytesAvailable;
            t2 = cputime;
        end
        if count==data_prop.bytes
            data_received = fread(tclient, length(data_mat), data_prop.class);
            fprintf('Received first data packet from the server\n')
        else
            data_received = NaN(el1, 1);
            fprintf('Failed to obtain first data packet within timeout period\n')
        end
        
        % Decision about connection validity
        if ~isnan(data_received(1))
            expsetup.general.plex_data_rate = data_received(1);
            settings.tcpip.success_ini = 1;
            fprintf ('Connection between two computers matched, will start experiment\n')
        else
            fprintf ('Failed to receive instructions from plexon computer; will run without plex recording\n')
            expsetup.general.plexon_online_spikes = 0;
        end
        
    else
        fprintf('tcpip address not specified, experiment will run without online plex collection\n')
        expsetup.general.plexon_online_spikes = 0;
    end
    
    
end


%% Initialize stimulator input

if expsetup.general.stimulator_input_on == 1
    if isfield (expsetup.ni_daq, 'device_name')
        
        % Create session
        devices = daq.getDevices; % Just for info
        temp = get(daq.getVendors());
        ni.session_stimulator_input = daq.createSession(temp.ID);
        
        % Settings
        device_name = expsetup.ni_daq.device_name;
        ni_ch_id = expsetup.ni_daq.stimulator_input_channel_id;
        ni_m_type = expsetup.ni_daq.stimulator_input_measurement_type;
        ni.session_stimulator_input.Rate = expsetup.ni_daq.stimulator_input_rate;
        ni.session_stimulator_input.IsContinuous = true;
        
        % Add analog output channel
        addAnalogInputChannel(ni.session_stimulator_input, device_name, ni_ch_id, ni_m_type);
        
    else
        error('ni-daq device name not specified');
    end
end

%% Initialize displays

runexp_display_ini_v10;

% Load normalized gamma table (if using one)
try
    load MyGammaTable.csv
    Screen ('LoadNormalizedGammaTable', expsetup.screenini.window, MyGammaTable);
end


%% Initialize eye-tracker recording
% It is necessary that display is initialized beforehand (eyetracker needs display coordinates)

if expsetup.general.recordeyes == 2
    fprintf ('\n')
    eyelink_selection=input('Is this an eyetracking experiment? "y" - yes, "n" - no: ', 's');
    fprintf ('\n')
    if strcmp(eyelink_selection, 'y')
        expsetup.general.recordeyes=1;
    else
        expsetup.general.recordeyes=0;
    end
end

if expsetup.general.recordeyes == 1
    runexp_eyelink_ini_v11
end


%%  RUN THE EXPERIMENT

% Initialize experimental matrix
eval(expsetup.general.code_settings)

time_expstart = GetSecs;

% Run the loop
tid=1; % tid = Trial ID (for example, 2nd out of 100 trials)
endexp1=0;

while endexp1==0
    
    if ismac
        ListenChar(2); % 2 turns the keyboard output to matlab
    end
    
    char = expsetup.general.empty_key; % Easy way to get rid of keyboard presses recorded before each trial
    tno1 = expsetup.general.trials_before_saving; % How many trials to run before saving data
    
    % Hide or show cursor
    if expsetup.general.hidecursor == 2
        HideCursor;
    end
    
    % Show welcome pic
    if expsetup.general.code_instruction_on == 1
        if tid==1
            instr_pic = 'image_welcome';
            runexp_instruction_v10(instr_pic, 1);
        end
    end
   
    
    % Calibrate (on first trial or on each block)
    if tid==1
        if expsetup.general.recordeyes == 1
            fprintf ('\n\nUse "space"  to initialize calibration\n')
            instr_pic = 'image_calibration';
            char = runexp_instruction_v10(instr_pic, 1);
            if iscell(char)
                char=char{1};
            end
            if strcmp(char,'space')
                runexp_eyelink_calib_primate_v14;
                %===========
                % Automated calibration
                % EyelinkDoTrackerSetup(expsetup.eyelink); % Calibrate the eye tracker before each block
            end
        end
    end
    
    %================
    %================
    %================
    % Run trials loop
    try
        
        time_tstart_1 = GetSecs;

        % Run trials
        fprintf('\nTrial number is %i\n', tid);
        eval(expsetup.general.code_trial)
        
%         if tid>1
%             fprintf('Trial duration was %i ms \n', round((GetSecs-time_tstart_1)*1000))
%         end
        
        % Save data structure if more than tno trials were run (does not save early terminations)
        if tid>tno1
            dir1 = expsetup.general.directory_data_psychtoolbox_subject;
            f_name = sprintf( '%s%s', expsetup.general.subject_filename,  expsetup.general.data_structure_appendix);
            d1 = sprintf('%s%s', dir1, f_name);
            save (d1, 'expsetup');
        end
        
    catch
        
        fprintf('\nCrashed on trial %i\n\n', tid);
        
        % Save data structure if more than tno trials were run (does not save early terminations)
        if tid>tno1
            dir1 = expsetup.general.directory_data_psychtoolbox_subject;
            f_name = sprintf( '%s%s', expsetup.general.subject_filename,  expsetup.general.data_structure_appendix);
            d1 = sprintf('%s%s', dir1, f_name);
            save (d1, 'expsetup');
        end
        
        % Save eyelink file
        if tid>tno1 && expsetup.general.recordeyes == 1
            directory1 = expsetup.general.directory_data_eyelink_edf_subject;
            runexp_eyelink_save_v10(directory1);
        end
        
        if ismac
            ListenChar(0); % 1 turns the keyboard back on
        end
        
        % Hide or show cursor
        if expsetup.general.hidecursor == 2
            ShowCursor;
        end
        
%         % Show reward and number of trials completed
%         if expsetup.general.reward_on>0
%             reward_given = nansum(expsetup.stim.expmatrix(:, em_data_reward_size_ml));
%             fprintf ('\nAdministered reward %i milliliters\n', round(reward_given))
%         end
%         index1 = expsetup.stim.expmatrix(:, em_data_reject) == 1;
%         fprintf ('\nNumber of correct trials completed: %i \n', sum(index1))
        
        % Close the audio device
        if expsetup.general.psychaudio == 1 && ~isempty(expsetup.audio.handle)
            PsychPortAudio('Close', expsetup.audio.handle);
            fprintf ('\nClosed Psychtoolbox Sound device\n')
            expsetup.audio.handle = [];
        end 
        % This section is executed in case an error happens in the
        % experiment code implemented between try and catch...
        Screen('CloseAll')
        rethrow(lasterror);
        
    end
    
    
    %================
    %================
    %================
    
    % Check for keyboard whether to stop experiment
    if strcmp(char, expsetup.general.empty_key)
        [keyIsDown,timeSecs,keyCode] = KbCheck;
        char = KbName(keyCode);
        % Catch potential press of two buttons
        if iscell(char)
            char=char{1};
        end
    else
        char=char;
    end
    
    % Calibrate (upon button press)
    if strcmp(char,'c') && expsetup.general.recordeyes == 1
        fprintf ('\n\nUse "space"  to initialize calibration\n')
        instr_pic = 'image_calibration';
        char = runexp_instruction_v10(instr_pic, 1);
        if iscell(char)
            char=char{1};
        end
        if strcmp(char,'space')
            % Manual calibration
            runexp_eyelink_calib_primate_v14;
            %===========
            % Automated calibration
            % EyelinkDoTrackerSetup(expsetup.eyelink); % Calibrate the eye tracker before each block
        end
    end
    
    % Pause
    if tid>1
        if strcmp(char,'p') 
            fprintf ('\n\nPaused, press any key to continue\n')
            instr_pic = 'image_pause';
            runexp_instruction_v10(instr_pic, 1);
        end
    end
    
    % Forced experiment termination
    if strcmp(char, expsetup.general.quit_key)
        instr_pic = 'image_quit';
        fprintf('\nWant to terminate experiment? Type in letter "y" \n')
        char = runexp_instruction_v10(instr_pic, 1);
        if iscell(char)
            char=char{1};
        end
        if char == 'y'
            endexp1=1;
        end
    end
    
% % % % %     if tid==size(expsetup.stim.expmatrix,1)
% % % % %         endexp1=1;
% % % % %     end
    tid=tid+1;
    
    if ismac
        ListenChar(0); % 1 turns the keyboard back on
    end
    
end

%================
%================
%================


%% CLOSING OF SCREENS AND SOUNDS

% Show experiment over pic
if expsetup.general.code_instruction_on == 1
    instr_pic = 'image_over';
    runexp_instruction_v10(instr_pic);
end

% Close the audio device
if expsetup.general.psychaudio == 1 && ~isempty(expsetup.audio.handle)
    PsychPortAudio('Close', expsetup.audio.handle);
    fprintf ('\nClosed Psychtoolbox Sound device\n')
    expsetup.audio.handle = [];
end


% Hide or show cursor
if expsetup.general.hidecursor == 2
    ShowCursor;
end


% Save data structure if more than tno trials were run (does not save early terminations)
if tid>tno1
    dir1 = expsetup.general.directory_data_psychtoolbox_subject;
    f_name = sprintf( '%s%s', expsetup.general.subject_filename,  expsetup.general.data_structure_appendix);
    d1 = sprintf('%s%s', dir1, f_name);
    save (d1, 'expsetup');
end

% Save eyelink file
if tid>tno1 && expsetup.general.recordeyes == 1
    directory1 = expsetup.general.directory_data_eyelink_edf_subject;
        runexp_eyelink_save_v10(directory1);
end

       
fprintf('\nSettings successfully saved in directory %s\n', expsetup.general.directory_data_psychtoolbox_subject)
Screen('CloseAll');


%% CONVERSION OF EYETRACKER DATA

if expsetup.general.convert_edf == 1
    try
        if tid>tno1 && expsetup.general.recordeyes == 1
            
            fprintf ('\nConversion of edf2asc is going on, do not quit MALTAB\n')
            
            % Select folders for data saving
            cdDir_edf=expsetup.general.directory_data_eyelink_edf_subject;
            cdDir_asc=expsetup.general.directory_data_eyelink_asc_subject;
            cdDir_file = expsetup.general.subject_filename;
            %====================
            
            if ismac
                
                % Add edf2asc in /bin folder; use the file from
                % /Applications/EyeLink/EDF_Access_API/Example/
                
                % Change to edf directory
                cd(cdDir_edf)
                
                % create .dat file
                system(['edf2asc', ' ', sprintf('%s%s',cdDir_edf, cdDir_file),'.edf -s -miss -1.0 -y']);
                % Convert .asc file into .dat file and move it
                movefile(sprintf('%s.asc',cdDir_file), sprintf('%s%s.dat',cdDir_asc,cdDir_file))
                
                % Create .asc file
                system(['edf2asc', ' ', sprintf('%s%s',cdDir_edf, cdDir_file),'.edf -e -y']);
                % Move .asc file
                movefile(sprintf('%s.asc',cdDir_file), sprintf('%s%s.asc',cdDir_asc,cdDir_file))
                
            elseif ispc
                
                % Tell the path to edf2asc program file
                edf2asc_path = expsetup.general.edf2asc_path;
                edf2asc = edf2asc_path;
                
                % Change to edf directory
                cd(cdDir_edf)
                
                % create .dat file
                system(['edf2asc', ' ', sprintf('%s%s',cdDir_edf, cdDir_file),'.edf -s -miss -1.0 -y']);
                % Convert .asc file into .dat file and move it
                movefile(sprintf('%s.asc',cdDir_file), sprintf('%s%s.dat',cdDir_asc,cdDir_file))
                
                % Create .asc file
                system(['edf2asc', ' ', sprintf('%s%s',cdDir_edf, cdDir_file),'.edf -e -y']);
                % Move .asc file
                movefile(sprintf('%s.asc',cdDir_file), sprintf('%s%s.asc',cdDir_asc,cdDir_file))
                
            end
            fprintf ('\nDone with eyelink data conversion\n')
        end
    end
end


%% Print some statistics

% time_expend = GetSecs;
% fprintf ('\nExperiment duration %d minutes\n', ceil((time_expend-time_expstart)/60))
% 
% if expsetup.general.reward_on>0
%     reward_given = nansum(expsetup.stim.expmatrix(:, em_data_reward_size_ml));
%     fprintf ('\nAdministered reward %i milliliters\n', round(reward_given))
% end
% index1 = expsetup.stim.expmatrix(:, em_data_reject) == 1;
% fprintf ('\nNumber of correct trials completed: %i \n', sum(index1))

