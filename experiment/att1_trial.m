% Here all displays presented


%% Exp stage (either keep the same or change the task)

if tid==1
    expsetup.stim.exp_version_temp = 5; % Version to start with on the first trial
    expsetup.stim.exp_version_update_next_trial = 0;
    fprintf('Task level is %.2f\n', expsetup.stim.exp_version_temp)
elseif tid>1
    if expsetup.stim.exp_version_update_next_trial == 0 % Keep the same
        b = expsetup.stim.esetup_exp_version(tid-1,1);
        expsetup.stim.exp_version_temp = b;
    elseif expsetup.stim.exp_version_update_next_trial == 1 % Change the task
        a = expsetup.stim.esetup_exp_version(tid-1,1); % Take previous trial exp version
        b = expsetup.stim.training_stage_matrix (expsetup.stim.training_stage_matrix<a); % Other available exp versions
        b = b(end);
        expsetup.stim.exp_version_temp = b; % Take largest available number (smallest number is end of training)
    end
    fprintf('Task level is %.2f\n', b)
end


%% Determine whether task difficulty/level changes

% Find index of trials to check performance
% All indexes are up to trial tid-1 (as tid trial is not defined yet)
ind0 = tid-1 - expsetup.stim.trial_online_counter + 1; % Trial from which performance is measured
if ind0 > 0
    ind1 = ind0 : 1: tid-1; % Trials to check performance
elseif ind0<=0
    ind1 = [];
end

% How many correct/error trials are there
if ~isempty(ind1)
    total1 = sum (expsetup.stim.edata_trial_online_counter(ind1) == 1); % Correct
    total2 = sum (expsetup.stim.edata_trial_online_counter(ind1) == 2); % Error
    fprintf('Online stimulus updating: %d out of %d trials were correct\n', total1, expsetup.stim.trial_online_counter)
end

% Select variables to increase/decrease
% No performance updating if exp_version_temp==1;
if expsetup.stim.exp_version_temp~=1
    att1_trial_update_stimuli;
end


%% Change task difficulty/level

%===============
%===============
% A - if not enough trials collected
if isempty(ind1) && expsetup.stim.exp_version_temp~=1
    
    % Start of experiment uses default values
    for i = 1:numel(tv1)
        b = tv1(i).temp_var_ini;
        tv1(i).temp_var_current = b;
    end
    
    i=numel(tv1);
    fprintf('New task initialized: variable %s is %.2f \n', tv1(i).name, b)
    
    
    
    %===============
    %===============
    % B - If performance is good, update stimulus from previous trial to make task harder
elseif ~isempty(ind1) && total1 >= expsetup.stim.trial_correct_goal_up && expsetup.stim.exp_version_temp~=1
    
    % Select stim property and change it
    for i = 1:numel(tv1)
        % Select previous stim
        a = expsetup.stim.(tv1(i).name);
        a = a(tid-1,1);
        % Change stim
        b = a + tv1(i).temp_var_ini_step; %
        tv1(i).temp_var_current = b;
        % If stimulus reached the threshold, then stop updating it
        if tv1(i).temp_var_ini < tv1(i).temp_var_final && tv1(i).temp_var_current >= tv1(i).temp_var_final
            tv1(i).temp_var_current = tv1(i).temp_var_final;
        elseif tv1(i).temp_var_ini >= tv1(i).temp_var_final && tv1(i).temp_var_current <= tv1(i).temp_var_final
            tv1(i).temp_var_current = tv1(i).temp_var_final;
        end
    end
    
    % Print results
    i=numel(tv1);
    fprintf('Good performance: variable %s changed from %.2f to %.2f\n', tv1(i).name, a, tv1(i).temp_var_current)
    
    % Reset the counter after each update
    expsetup.stim.edata_trial_online_counter(ind1) = 99;
    
    
    
    %===============
    %===============
    % C - If performance is bad, update stimulus from previous to make task easier
elseif ~isempty(ind1) && total2 >= expsetup.stim.trial_correct_goal_down && expsetup.stim.exp_version_temp~=1
    
    % Select stim property and change it
    for i = 1:numel(tv1)
        % Select previous stim
        a = expsetup.stim.(tv1(i).name);
        a = a(tid-1,1);
        % Change stim
        b = a - tv1(i).temp_var_ini_step; %
        tv1(i).temp_var_current = b;
        % If stimulus reached the threshold, then stop updating it
        if tv1(i).temp_var_ini < tv1(i).temp_var_final && tv1(i).temp_var_current <= tv1(i).temp_var_ini
            tv1(i).temp_var_current = tv1(i).temp_var_ini;
        elseif tv1(i).temp_var_ini >= tv1(i).temp_var_final && tv1(i).temp_var_current >= tv1(i).temp_var_ini
            tv1(i).temp_var_current = tv1(i).temp_var_ini;
        end
    end
    
    % Print results
    i=numel(tv1);
    fprintf('Poor performance: variable %s changed from %.2f to %.2f\n', tv1(i).name, a, tv1(i).temp_var_current)
    
    % Reset the counter after each update
    expsetup.stim.edata_trial_online_counter(ind1) = 99;
    
    
    
    %===============
    %===============
    % D - If not enough of trials, copy values from earlier trial
elseif ~isempty(ind1) && total1 < expsetup.stim.trial_correct_goal_up && total2 < expsetup.stim.trial_correct_goal_down && expsetup.stim.exp_version_temp~=1
    
    % Select stim property and change it
    for i = 1:numel(tv1)
        % Select previous stim
        a = expsetup.stim.(tv1(i).name);
        a = a(tid-1,1);
        % Change stim
        b = a; % If not enough of trials, copy values from earlier trial
        tv1(i).temp_var_current = b;
    end
    
    i=numel(tv1);
    fprintf('Not enough trials to track performance: variable %s is %.2f \n', tv1(i).name, b)
    
end

%===================
% Make a decision whether to change the task level on next trial

% If stimulus reached the value selected, then stop updating it
if ~isempty(ind1) && expsetup.stim.exp_version_temp~=1
    i=numel(tv1);
    if tv1(i).temp_var_current==tv1(i).temp_var_final
        expsetup.stim.exp_version_update_next_trial = 1;
        % Print output onscreen
        a = expsetup.stim.esetup_exp_version(tid-1,1); % Take previous trial exp version
        b = expsetup.stim.training_stage_matrix (expsetup.stim.training_stage_matrix<a); % Other available exp versions
        b = b(end); % Take largest available number (smallest number is end of training)
        fprintf('Task criterion reached, on next trial will change task from level %.2f to level %.2f\n', a, b)
    elseif tv1(i).temp_var_current~=tv1(i).temp_var_final
        expsetup.stim.exp_version_update_next_trial = 0;
    end
elseif expsetup.stim.exp_version_temp==1 % Never change the task for final level
    expsetup.stim.exp_version_update_next_trial = 0;
end



%% PREPARE ALL OBJECTS AND FRAMES TO BE DRAWN:

window = expsetup.screen.window;

% On first trial, run randomization before trial starts
% From second trial track performance
if tid == 1
    att1_trial_randomize;
elseif tid>1
    att1_trial_randomize;
end

% Initialize all rectangles for the task
att1_trial_stimuli;

% Initialize all frames for the task
att1_trial_frames;


%% EYETRACKER INITIALIZE

% Start recording
if expsetup.general.recordeyes==1
    Eyelink('StartRecording');
    msg1=['TrialStart ', num2str(tid)];
    Eyelink('Message', msg1);
    WaitSecs(0.1);  % Record a few samples before we actually start displaying
end

% % SEND MESSAGE WITH TRIAL ID TO EYELINK
% if expsetup.general.recordeyes==1
%     a1 = em_blockno;
%     t1=length(find(expsetup.stim.expmatrix(:,a1)<expsetup.stim.expmatrix(tid,a1)));
%     trial_current=tid-t1; % Which trial of the current block it is?
%     trial_perblock=length(find(expsetup.stim.expmatrix(:,a1)==expsetup.stim.expmatrix(tid,a1))); % How many trials in this block?
%     msg1 = sprintf('Trial %i/%i in the block %i/%i', trial_current, trial_perblock, expsetup.stim.expmatrix(tid,a1), max(expsetup.stim.expmatrix(:,a1)) );
%     Eyelink('Command', 'record_status_message ''%s'' ', msg1);
% end


%% ================

% FIRST DISPLAY - BLANK

Screen('FillRect', window, expsetup.stim.background_color);
if expsetup.general.record_plexon==1
    Screen('FillRect', window, [255, 255, 255], ph_rect, 1); % Photodiode
end
[~, time_current, ~]=Screen('Flip', window);

% Save plexon event
if expsetup.general.record_plexon==1
    a1 = zeros(1,expsetup.ni_daq.digital_channel_total);
    a1_s = expsetup.general.plex_event_start; % Channel number used for events
    a1(a1_s)=1;
    outputSingleScan(ni.session_plexon_events,a1);
    a1 = zeros(1,expsetup.ni_daq.digital_channel_total);
    outputSingleScan(ni.session_plexon_events,a1);
end

% Save eyelink and psychtoolbox events
if expsetup.general.recordeyes==1
    Eyelink('Message', 'first_display');
end
expsetup.stim.edata_first_display(tid,1) = time_current;

% Save trialmat event
c1_frame_index1 = 1;
expsetup.stim.eframes_time{tid}(c1_frame_index1, 1) = time_current; % Save in the first row during presentation of first dislapy


%%  TRIAL LOOP

loop_over = 0;
while loop_over==0
    
    
    %=================
    % Initialize new frame index
    c1_frame_index1 = c1_frame_index1+1;
    
    
    %% Update frames dependent on acquiring fixation
    
    % Changes in fixation (stop blinking)
    if ~isnan(expsetup.stim.edata_fixation_acquired(tid,1))
        expsetup.stim.eframes_fix_blink{tid}(c1_frame_index1:end, 1) = 1;
    end
    
    % Determine whether to show attention cue (based on timing)
    t0 = expsetup.stim.edata_fixation_acquired(tid,1);
    t1 = expsetup.stim.esetup_att_cue_fix_soa(tid,1);
    if time_current - t0 >= t1
        % Update framesmat
        if nansum(expsetup.stim.eframes_att_cue_on{tid})==0
            % Copy data from trial mat
            a = expsetup.stim.eframes_att_cue_on_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_att_cue_on{tid}(ind, 1) = a(1:numel(ind));
            % Copy data from trial mat
            a = expsetup.stim.eframes_att_cue_off_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_att_cue_off{tid}(ind, 1) = a(1:numel(ind));
        end
    end
    
    
    % Determine whether to show gabor patches yet
    t0 = expsetup.stim.edata_att_cue_on(tid,1);
    if time_current - t0 >= expsetup.stim.probe_cue_isi
        % Update framesmat
        if nansum(expsetup.stim.eframes_att_stim_on{tid})==0
            % Copy data from trial mat
            a = expsetup.stim.eframes_att_stim_on_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_att_stim_on{tid}(ind, 1) = a(1:numel(ind));
            % Copy data from trial mat
            a = expsetup.stim.eframes_probe_on_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_probe_on{tid}(ind, 1) = a(1:numel(ind));
            % Copy data from trial mat
            a = expsetup.stim.eframes_probe_off_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_probe_off{tid}(ind, 1) = a(1:numel(ind));
        end
    end
    
    
    % Determine whether to show response rings yet
    t0 = expsetup.stim.edata_probe_off(tid,1);
    if time_current - t0 >= expsetup.stim.probe_to_response_isi
        % Update framesmat
        if nansum(expsetup.stim.eframes_response_ring_on{tid})==0
            % Copy data from trial mat
            a = expsetup.stim.eframes_response_ring_size_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_response_ring_size{tid}(ind, 1) = a(1:numel(ind));
            % Copy data from trial mat
            a = expsetup.stim.eframes_response_ring_on_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_response_ring_on{tid}(ind, 1) = a(1:numel(ind));
            % Copy data from trial mat
            a = expsetup.stim.eframes_response_ring_off_temp{tid};
            ind = c1_frame_index1 : size(a,1);
            expsetup.stim.eframes_response_ring_off{tid}(ind, 1) = a(1:numel(ind));
        end
        % Remove fixation
        expsetup.stim.eframes_fix_blink{tid}(c1_frame_index1:end, 1) = 0; % Fixation disappears
    end
    
    
    %% Plot displays
    
    %==================
    % Noise texture
    
    var1 = expsetup.stim.eframes_background_on{tid};
    if var1(c1_frame_index1,1)==1
        tex=texture_backgroud{1};
        Screen('DrawTexture', window, tex, [], background_rect, [], 0);
    end
    
    %==================
    % Fixation
    if expsetup.stim.eframes_fix_blink{tid}(c1_frame_index1,1)==1
        % Show stimulus
        fcolor1 = expsetup.stim.fixation_color_baseline;
        if strcmp(expsetup.stim.fixation_shape_baseline,'circle')
            Screen('FillArc', window,  fcolor1, fixation_rect, 0, 360);
        elseif strcmp(expsetup.stim.fixation_shape_baseline,'square')
            Screen('FillRect', window, fcolor1, fixation_rect, expsetup.stim.fixation_pen);
        end
    end
    
    %==================
    % Attention cue
    if expsetup.stim.eframes_att_cue_on{tid}(c1_frame_index1,1)==1
        % Show stimulus
        fcolor1 = expsetup.stim.fixation_color_baseline;
        Screen('DrawLine', window, fcolor1, att_cue_rect(1), att_cue_rect(2), att_cue_rect(3), att_cue_rect(4), expsetup.stim.att_cue_pen);
    end
    
    
    %==================
    % Plot gabor patches
    if expsetup.stim.eframes_att_stim_on{tid}(c1_frame_index1, 1) > 0
        for i = 1:size(att_stim_rect,2)
            j = expsetup.stim.eframes_att_stim_on{tid}(c1_frame_index1, 1);
            tex = texture_att_stim{1,i,j};
            Screen('DrawTexture', window, tex, [], att_stim_rect(:,i), [], 0);
        end
    end
    
    
    %=====================
    % Response object 1 or 2
    
    if expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1, 1) == 1 || expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1, 1) == 2
        
        % Response position (variable on each trial)
        pos1 = expsetup.stim.esetup_response_ring_position(tid,1);
        rad1 = expsetup.stim.esetup_response_ring_radius(tid,1);
        [xc, yc] = pol2cart(pos1*pi/180, rad1); % Convert to cartesian
        coord1=[];
        coord1(1)=xc; coord1(2)=yc;
        % Size
        a = expsetup.stim.eframes_response_ring_size{tid}(c1_frame_index1, 1);
        sz1 = [0, 0, a, a];
        % Rect
        response_ring_rect = runexp_convert_deg2pix_rect_v10(coord1, sz1); % One column - one object;
        
        % Color
        if expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1, 1) == 1
            fcolor1 = expsetup.stim.response_ring_color1;
        elseif expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1, 1) == 2
            fcolor1 = expsetup.stim.response_ring_color2;
        end
        
        % Plot
        Screen('FillArc', window,  fcolor1, response_ring_rect, 0, 360);
        
    end
    
    
    
    %% FLIP AND RECORD TIME
    
    [~, time_current, ~]=Screen('Flip', window);
    
    % Save flip time into trialmat
    expsetup.stim.eframes_time{tid}(c1_frame_index1, 1) = time_current; % Add row to each refresh
    
    % Record fixation onset
    if expsetup.stim.eframes_fix_blink{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_fixation_on(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'fixation_on');
        end
        expsetup.stim.edata_fixation_on(tid,1) = time_current;
    end
    
    % Record att cue onset
    if expsetup.stim.eframes_att_cue_on{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_att_cue_on(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'att_cue_on');
        end
        expsetup.stim.edata_att_cue_on(tid,1) = time_current;
    end
    
    % Record probe onset
    if expsetup.stim.eframes_probe_on{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_probe_on(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'probe_on');
        end
        expsetup.stim.edata_probe_on(tid,1) = time_current;
    end
    
    % Record probe offset
    if expsetup.stim.eframes_probe_off{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_probe_off(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'probe_off');
        end
        expsetup.stim.edata_probe_off(tid,1) = time_current;
    end
    
    % Record response ring 1 onset
    if expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_response_ring1_on(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'response_ring1_on');
        end
        expsetup.stim.edata_response_ring1_on(tid,1) = time_current;
    end
    
    % Record response ring 2 onset
    if expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)==2 && isnan(expsetup.stim.edata_response_ring2_on(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'response_ring2_on');
        end
        expsetup.stim.edata_response_ring2_on(tid,1) = time_current;
    end
    
    if expsetup.stim.eframes_response_ring_off{tid}(c1_frame_index1,1)==1 && isnan(expsetup.stim.edata_response_ring_off(tid,1))
        if expsetup.general.recordeyes==1
            Eyelink('Message', 'response_ring_off');
        end
        expsetup.stim.edata_response_ring_off(tid,1) = time_current;
    end
    
    %==================
    %==================
    
    
    %%  Get eyelink data sample
    
    try
        [mx,my] = runexp_eyelink_get_v10;
        expsetup.stim.eframes_eye_x{tid}(c1_frame_index1, 1)=mx;
        expsetup.stim.eframes_eye_y{tid}(c1_frame_index1, 1)=my;
    catch
        expsetup.stim.eframes_eye_x{tid}(c1_frame_index1, 1)=999999;
        expsetup.stim.eframes_eye_y{tid}(c1_frame_index1, 1)=999999;
    end
    
    
    %% Check button presses
    
    [keyIsDown, keyTime, keyCode] = KbCheck;
    char = KbName(keyCode);
    % Catch potential press of two buttons
    if iscell(char)
        char=char{1};
    end
    
    % Record what kind of button was pressed
    if strcmp(char,'c') || strcmp(char,'p') || strcmp(char,'r') || strcmp(char, expsetup.general.quit_key)
        expsetup.stim.edata_error_code{tid} = 'experimenter terminated the trial';
    end
    
    
    %%  Record whether lever/button is being pressed during the trial
    
    if expsetup.general.arduino_on == 0
        
        % Save message if lever/button was detected
        if isnan(expsetup.stim.edata_lever_on(tid,1)) && keyIsDown==1 && strcmp(char,'space')
            if expsetup.general.recordeyes==1
                Eyelink('Message', 'lever_onset');
            end
            expsetup.stim.edata_lever_on(tid,1) = keyTime;
        end
        
        % Save message if lever/button was released
        if ~isnan(expsetup.stim.edata_lever_on(tid,1)) && isnan(expsetup.stim.edata_lever_off(tid,1)) && keyIsDown==0
            if expsetup.general.recordeyes==1
                Eyelink('Message', 'lever_off');
            end
            expsetup.stim.edata_lever_off(tid,1) = keyTime;
        end
        
        % Save duration of lever/button holding
        if ~isnan(expsetup.stim.edata_lever_off(tid,1)) && ~isnan(expsetup.stim.edata_lever_on(tid,1))
            v1 = expsetup.stim.edata_lever_off(tid,1) -  expsetup.stim.edata_lever_on(tid,1);
            expsetup.stim.edata_lever_duration(tid,1) = v1;
        end
        
    end
    
    %% Monitor trial performance
    
    %===================
    % Check whether lever/fixation was acquired in time
    %===================
    
    if isnan(expsetup.stim.edata_fixation_acquired(tid,1))
        
        % Time
        timer1_now = expsetup.stim.eframes_time{tid}(c1_frame_index1, 1);
        %
        timer1_start = expsetup.stim.edata_fixation_on(tid,1);
        %
        timer1_duration = expsetup.stim.esetup_fixation_acquire_duration(tid,1);
        
        if timer1_now - timer1_start < timer1_duration % Record an error
            if ~isnan(expsetup.stim.edata_lever_on(tid,1))
                expsetup.stim.edata_fixation_acquired(tid,1) = GetSecs;
            elseif isnan(expsetup.stim.edata_lever_on(tid,1))
                % Proceed with trial
            end
        elseif timer1_now - timer1_start >= timer1_duration % Record an error
            if ~isnan(expsetup.stim.edata_lever_on(tid,1))
                expsetup.stim.edata_fixation_acquired(tid,1) = GetSecs;
            elseif isnan(expsetup.stim.edata_lever_on(tid,1))
                expsetup.stim.edata_error_code{tid} = 'fixation not acquired in time';
            end
        end
        
    end
    
    %===================
    % Check whether lever was released too early
    %===================
    
    if isnan(expsetup.stim.edata_response_ring1_on(tid,1)) && isnan(expsetup.stim.edata_response_ring2_on(tid,1)) && ~isnan(expsetup.stim.edata_lever_off(tid,1))
        expsetup.stim.edata_error_code{tid} = 'lever hold failure';
    end
    
    %===================
    % Save error if lever was not released
    %===================
    
    if ~isnan(expsetup.stim.edata_response_ring_off(tid,1)) && isnan(expsetup.stim.edata_lever_off(tid,1))
        expsetup.stim.edata_error_code{tid} = 'lever not released';
    end
    
    
    %===================
    % Check which lever response was given after response object appeared
    %===================
    
    if ~isnan(expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)) && ~isnan(expsetup.stim.edata_lever_off(tid,1)) && isnan(expsetup.stim.edata_lever_response(tid,1))
        
        if expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)==1 || expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)==2 % Save associated target
            expsetup.stim.edata_lever_response(tid,1) = expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1);
        elseif expsetup.stim.eframes_response_ring_on{tid}(c1_frame_index1,1)==0
            ind = expsetup.stim.eframes_response_ring_on{tid}(1:c1_frame_index1,1);
            ind = ind(ind>0);
            expsetup.stim.edata_lever_response(tid,1) = ind(end); % Save last associated target
        end
        
        % Determine whether correct target was selected
        if expsetup.stim.edata_lever_response(tid,1) == expsetup.stim.esetup_probe_change(tid,1)
            expsetup.stim.edata_error_code{tid} = 'correct';
        elseif expsetup.stim.edata_lever_response(tid,1) ~= expsetup.stim.esetup_probe_change(tid,1)
            expsetup.stim.edata_error_code{tid} = 'probe response error';
        end
        
    end
    
    
    %% If its the last frame, save few missing parameters & terminate
    
    % If run out of frames  - end trial (should never happen)
    if c1_frame_index1==size(expsetup.stim.eframes_time{tid},1)
        loop_over = 1;
    end
    
    % If error - end trial
    if ~isnan(expsetup.stim.edata_error_code{tid})
        loop_over = 1;
    end
    
    % If rings disappear - end trial
    if expsetup.stim.eframes_response_ring_off{tid}(c1_frame_index1,1) == 1
        loop_over = 1;
    end
    
end

% Reduce trialmat in size (save only frames that are used)
if c1_frame_index1+1<size(expsetup.stim.eframes_time{tid},1)
    f1 = fieldnames(expsetup.stim);
    ind = strncmp(f1,'eframes', 7);
    for i=1:numel(ind)
        if ind(i)==1
            if iscell(expsetup.stim.(f1{i}))
                expsetup.stim.(f1{i}){tid}(c1_frame_index1+1:end,:,:) = [];
            end
        end
    end
end

% Clear off all the screens
Screen('FillRect', window, expsetup.stim.background_color);
if expsetup.general.record_plexon==1
    Screen('FillRect', window, [0, 0, 0], ph_rect, 1);
end
[~, time_current, ~]=Screen('Flip', window);

% Close texture
if exist('tex')
    Screen('Close', tex);
end

% Plexon message that display is cleared
% Individual event mode (EVENT 2)
if expsetup.general.record_plexon==1
    a1 = zeros(1,expsetup.ni_daq.digital_channel_total);
    a1_s = expsetup.general.plex_event_end; % Channel number used for events
    a1(a1_s)=1;
    outputSingleScan(ni.session_plexon_events,a1);
    a1 = zeros(1,expsetup.ni_daq.digital_channel_total);
    outputSingleScan(ni.session_plexon_events,a1);
end


% Save eyelink and psychtoolbox events
if expsetup.general.recordeyes==1
    Eyelink('Message', 'loop_over');
end
expsetup.stim.edata_loop_over(tid,1) = time_current;


% Print trial duration
t1 = expsetup.stim.edata_loop_over(tid,1);
t0 = expsetup.stim.edata_first_display(tid,1);
fprintf('Trial duration (from first display to reward) was %i ms \n', round((t1-t0)*1000))
fprintf('Trial evaluation: %s\n', expsetup.stim.edata_error_code{tid})


% Clear eyelink screen
if expsetup.general.recordeyes==1
    Eyelink('command','clear_screen 0');
end

% Clear off all the screens again
Screen('FillRect', window, expsetup.stim.background_color);
[~, time_current, ~]=Screen('Flip', window);



%% Online performance tracking

% Check whether trial is counted towards online performance tracking. In
% some cases correct trials can be discounted.

if expsetup.stim.esetup_exp_version(tid,1) <=5
    if strcmp(expsetup.stim.edata_error_code{tid}, 'correct')
        % If previous trial was an error and this is a repeat - discount this trial from tracking
        if strcmp(expsetup.stim.edata_error_code{tid}, 'probe response error') && expsetup.stim.trial_error_repeat==1
            expsetup.stim.edata_trial_online_counter(tid,1) = 99;
        else
            expsetup.stim.edata_trial_online_counter(tid,1) = 1;
        end
    elseif strcmp(expsetup.stim.edata_error_code{tid}, 'probe response error')
        expsetup.stim.edata_trial_online_counter(tid,1) = 2;
    end
end



% % Plot reward image onscreen
% if expsetup.general.reward_on==1
%     if expsetup.stim.reward_feedback==1
%
%         if loop_error==1 || strcmp(char, 'space') || strcmp(char, 'r')
%             Screen('DrawTexture',window, tex_positive, [], reward_rect, [], 0);
%             [~, time_current, ~]=Screen('Flip', window);
%         else
%             Screen('DrawTexture',window, tex_negative, [], reward_rect, [], 0);
%             [~, time_current, ~]=Screen('Flip', window);
%         end
%
%         % Record reward image is onscreen
%         if loop_reward_image_on==0
%             if expsetup.general.recordeyes==1
%                 Eyelink('Message', 'reward_image_on');
%             end
%             time12=time_current;
%             loop_reward_image_on=1;
%         end
%     elseif expsetup.stim.reward_feedback==2  %Auditory feedback
%
%         % Record reward image is onscreen
%         if loop_reward_image_on==0
%             if expsetup.general.recordeyes==1
%                 Eyelink('Message', 'reward_image_on');
%             end
%             time12=time_current;
%             loop_reward_image_on=1;
%         end
%
%         if loop_error==1 || strcmp(char, 'space') || strcmp(char, 'r')
%             b1 = MakeBeep(600, 0.05);
%             b2 = MakeBeep(800, 0.05);
%             beep = [b1, b2];
%             Snd('Play', beep);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         elseif loop_error==5 % Wrong target selected
%             b1 = MakeBeep(600, 0.05);
%             b2 = MakeBeep(200, 0.05);
%             beep = [b1, b2];
%             Snd('Play', beep);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         else
%             b1 = sin(0:2000);
%             beep = [b1, b1];
%             Snd('Play', beep);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         end
%
%     elseif expsetup.stim.reward_feedback==3 && isfield (expsetup.general, 'arduino_session')
%
%         % Record reward image is onscreen
%         if loop_reward_image_on==0
%             if expsetup.general.recordeyes==1
%                 Eyelink('Message', 'reward_image_on');
%             end
%             time12=time_current;
%             loop_reward_image_on=1;
%         end
%
%         if loop_error==1 || strcmp(char, 'space') || strcmp(char, 'r')
%             playTone(expsetup.general.arduino_session, 'D10', 600, 0.05);
%             playTone(expsetup.general.arduino_session, 'D10', 800, 0.05);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         elseif loop_error==5 % Wrong target selected
%             playTone(expsetup.general.arduino_session, 'D10', 600, 0.05);
%             playTone(expsetup.general.arduino_session, 'D10', 200, 0.1);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         else
%             playTone(expsetup.general.arduino_session, 'D10', 100, 0.2);
%             WaitSecs(expsetup.stim.reward_feedback_audio_dur);
%         end
%
%     end
% end
%
%
% % Prepare reward signal
% if expsetup.general.reward_on==1
%     if loop_error==1 || strcmp(char, 'r')
% %         % Continous reward
% %         reward_duration = expsetup.stim.expmatrix(tid,em_reward_size_ms);
% %         signal1 = linspace(expsetup.ni_daq.reward_voltage, expsetup.ni_daq.reward_voltage, reward_duration)';
% %         signal1 = [0; signal1; 0; 0; 0; 0; 0];
% %         queueOutputData(ni.session_reward, signal1);
%     end
% end

% if expsetup.general.reward_on == 1
%     if loop_error==1 || strcmp(char, 'space') || strcmp(char, 'r')
%         if loop_reward_on==0 % && time_current - expsetup.stim.expmatrix(tid,em_data_fixation_maintained) >= expsetup.stim.expmatrix(tid,em_reward_soa)
%             ni.session_reward.startForeground;
%             loop_reward_on=1;
%         end
%     end
% end
%
% % Record reward on
% if loop_reward_on==1
%     if expsetup.general.recordeyes==1
%         Eyelink('Message', 'reward_on');
%     end
%     time9=time_current;
%     loop_reward_on=2;
% end
%
%
% %% Inter-trial interval & possibility to add extra reward
%
% if loop_error == 1 % On correct trials OR reward trials
%     trial_duration = expsetup.stim.trial_dur_intertrial;
% else % Error trials
%     trial_duration = expsetup.stim.trial_dur_intertrial_error;
% end
% timer1_now = GetSecs;
% time_start = GetSecs;
%
% try
%     endloop_skip = 0;
%     if strcmp(char, expsetup.general.quit_key) || strcmp(char, 'p') || strcmp(char, 'c')
%         endloop_skip = 1;
%     end
% end
% while endloop_skip == 0
%
%     % Record what kind of button was pressed
%     [keyIsDown,timeSecs,keyCode] = KbCheck;
%     char = KbName(keyCode);
%     % Catch potential press of two buttons
%     if iscell(char)
%         char=char{1};
%     end
%
%     % Give reward
%     if (strcmp(char,'space') || strcmp(char,'r')) && loop_reward_on==0 && expsetup.general.reward_on == 1
%
%         % Prepare reward signal
%         if expsetup.general.reward_on==1
%             % Continous reward
%             reward_duration = expsetup.stim.expmatrix(tid,em_reward_size_ms);
%             signal1 = linspace(expsetup.ni_daq.reward_voltage, expsetup.ni_daq.reward_voltage, reward_duration)';
%             signal1 = [0; signal1; 0; 0; 0; 0; 0];
%             queueOutputData(ni.session_reward, signal1);
%         end
%
%         if expsetup.general.reward_on == 1
%             if loop_reward_on==0 % && time_current - expsetup.stim.expmatrix(tid,em_data_fixation_maintained) >= expsetup.stim.expmatrix(tid,em_reward_soa)
%                 ni.session_reward.startForeground;
%                 loop_reward_on=1;
%             end
%         end
%
%         % Record reward on
%         if loop_reward_on==1
%             if expsetup.general.recordeyes==1
%                 Eyelink('Message', 'reward_on');
%             end
%             time9=time_current;
%             loop_reward_on=2;
%         end
%
%         % End loop
%         endloop_skip=1;
%     end
%
%     % Check time
%     timer1_now = GetSecs;
%     if timer1_now - time_start >= trial_duration
%         endloop_skip=1;
%     end
% end
%


%
% % Save error done in the trial
% expsetup.stim.expmatrix(tid, em_data_reject) = loop_error; % Save error
% if loop_error==0
%     expsetup.stim.expmatrix(tid, em_data_reject) = 99; % Code for unknown errors
% end
%



%% Trigger new trial

Screen('FillRect', window, expsetup.stim.background_color);
Screen('Flip', window);


% %% Plot online data
%
% % If plexon recording exists, get spikes
% if expsetup.general.record_plexon == 1 && expsetup.general.plexon_online_spikes == 1
%     look2_online_spikes;
% end


if expsetup.general.record_plexon == 0
    if ~strcmp(expsetup.stim.edata_error_code{tid}, 'correct') || sum(tid==1:expsetup.stim.plot_every_x_trial:10000)==1
        att1_online_plot;
    end
end

%% Stop experiment if too many errors in a row

if strcmp(expsetup.stim.edata_error_code{tid}, 'fixation not acquired in time') % Trial failed
    expsetup.stim.edata_trial_abort_counter(tid,1) = 1;
end

% Add pause if trials are not accurate
if tid>=expsetup.stim.trial_abort_counter
    ind1 = tid-expsetup.stim.trial_abort_counter+1:tid;
    s1 = expsetup.stim.edata_trial_abort_counter(ind1, 1)==1;
    if sum(s1) == expsetup.stim.trial_abort_counter
        if ~strcmp(char,expsetup.general.quit_key')
            char='x';
            % Over-write older trials
            expsetup.stim.edata_trial_abort_counter(ind1, q) = 2000;
        end
    end
end


%% STOP EYELINK RECORDING

if expsetup.general.recordeyes==1
    msg1=['TrialEnd ',num2str(tid)];
    Eyelink('Message', msg1);
    Eyelink('StopRecording');
end

fprintf('  \n')

