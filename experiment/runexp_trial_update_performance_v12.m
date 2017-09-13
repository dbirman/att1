% Determine whether task difficulty/level changes
%
% Variables:
% trial_online_counter = 1 - correct trial, 2 - error trial; NaN - not
% counted trial
% exp_version_temp - current version
% trial_correct_goal_down, trial_correct_goal_up - variables for monitoring
% performance
% expname_trial_update_stimuli - this code must exist
%
% v1.0 August 16, 2017. Initial code.
% v1.1 August 19, 2017. Recoded task version into string variable.
% v1.2 September 12, 2017. Added automatic task updating without changes in
% task parameters


%% Detect whether task is changing

if tid==1
    if ~isfield(expsetup.stim, 'exp_version_temp')
        expsetup.stim.exp_version_temp = expsetup.stim.training_stage_matrix{end}; % Version to start with on the first trial
    end
    expsetup.stim.exp_version_update_next_trial = 0;
    fprintf('Running task version: %s\n', expsetup.stim.exp_version_temp)
elseif tid>1
    if expsetup.stim.exp_version_update_next_trial == 0 % Keep the same
        b = expsetup.stim.esetup_exp_version{tid-1};
        expsetup.stim.exp_version_temp = b;
    elseif expsetup.stim.exp_version_update_next_trial == 1 % Change the task
        a = expsetup.stim.esetup_exp_version{tid-1};
        ind1 = strcmp(expsetup.stim.training_stage_matrix, a);
        ind1 = find(ind1==1);
        if ind1+1<=numel(expsetup.stim.training_stage_matrix)
            b = expsetup.stim.training_stage_matrix{ind1+1};
        else
            b = expsetup.stim.training_stage_matrix{ind1};
        end
        expsetup.stim.exp_version_temp = b;
    end
    fprintf('Running task version: %s\n', expsetup.stim.exp_version_temp)
end


%% Determine how to update the task to next stage

% Initialize tv1 structure
u1 = sprintf('%s_trial_update_stimuli', expsetup.general.expname); % Path to file containing trial settings
eval (u1);

% Select which update is being done
if isfield(tv1(1), 'update')
    if strcmp(tv1(1).update, 'gradual') % Gradual update
        trial_online_counter = expsetup.stim.trial_online_counter_gradual;
        update_var = 1;
    elseif strcmp(tv1(1).update, 'step') % Step update
        trial_online_counter = expsetup.stim.trial_online_counter_single_step;
        update_var = 2;
    end
else
    % No updating
    fprintf('This task does not update\n')
    update_var = 0;
    tv1 = struct; % Over-write structure empty;
    trial_online_counter = 1000;
end


% Find index of trials to check performance
% All indexes are up to trial tid-1 (as tid trial is not defined yet)
ind0 = tid-1 - trial_online_counter + 1; % Trial from which performance is measured
if ind0 > 0
    ind1 = ind0 : 1: tid-1; % Trials to check performance
elseif ind0<=0
    ind1 = [];
end

% How many correct/error trials are there
if ~isempty(ind1)
    total1 = sum (expsetup.stim.edata_trial_online_counter(ind1) == 1); % Correct
    total2 = sum (expsetup.stim.edata_trial_online_counter(ind1) == 2); % Error
    fprintf('Online stimulus updating: %d correct; %d error; out of %d total\n', total1, total2, trial_online_counter)
end


%% Update task gradually

if update_var==1
        
    %===============
    %===============
    % A - if not enough trials collected
    if isempty(ind1) && ~isempty(fieldnames(tv1))
        
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
    
    elseif ~isempty(ind1) && total1 >= expsetup.stim.trial_correct_goal_up && ~isempty(fieldnames(tv1))
        
        % Select stim property and change it
        for i = 1:numel(tv1)
            % Select previous stim
            a = expsetup.stim.(tv1(i).name);
            a = a(tid-1,1);
            % Change stim
            b = a + tv1(i).temp_var_ini_step;
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
    
    elseif ~isempty(ind1) && total2 >= expsetup.stim.trial_correct_goal_down && ~isempty(fieldnames(tv1))
        
        % Select stim property and change it
        for i = 1:numel(tv1)
            % Select previous stim
            a = expsetup.stim.(tv1(i).name);
            a = a(tid-1,1);
            % Change stim
            b = a - tv1(i).temp_var_ini_step;
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
    
    elseif ~isempty(ind1) && total1 < expsetup.stim.trial_correct_goal_up && total2 < expsetup.stim.trial_correct_goal_down && ~isempty(fieldnames(tv1))
        
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
    
    %=============
    % Make a decision whether to change the task level on next trial
    
    % If stimulus reached the value selected, then stop updating it
    if ~isempty(ind1) && ~isempty(fieldnames(tv1))
        i=numel(tv1);
        if tv1(i).temp_var_current==tv1(i).temp_var_final
            expsetup.stim.exp_version_update_next_trial = 1;
            fprintf('Task criterion reached, on next trial will change task\n')
        elseif tv1(i).temp_var_current~=tv1(i).temp_var_final
            expsetup.stim.exp_version_update_next_trial = 0;
        end
    elseif isempty(fieldnames(tv1)) % Never change the task for final level
        expsetup.stim.exp_version_update_next_trial = 0;
    end
    
end


%% Update task in a one step

if update_var==2
    
    if isempty(ind1)

        fprintf('No task updating, waiting to acquire %.0f trials\n', trial_online_counter)
        expsetup.stim.exp_version_update_next_trial = 0;
       
    elseif ~isempty(ind1)

        ind_task = strcmp(expsetup.stim.esetup_exp_version, expsetup.stim.exp_version_temp);
        goal_up = expsetup.stim.trial_online_counter_single_step_goal_up;
        prop_trials = expsetup.stim.trial_online_counter_single_step_tn;
        perf_correct = total1/(total1+total2); % Proportion correct
        
        % If enough trials in current training stage acquired
        if sum(ind_task) >= trial_online_counter
            
            if (total1+total2)/numel(ind1) < prop_trials % If not enough trials completed
                fprintf('No task updating, waiting to acquire %.2f completed trials\n', prop_trials)
                expsetup.stim.exp_version_update_next_trial = 0;
            elseif (total1+total2)/numel(ind1) >= prop_trials % If enough trials completed
                if perf_correct>=goal_up
                    fprintf('Task criterion reached, on next trial will change task\n')
                    expsetup.stim.exp_version_update_next_trial = 1;
                    % Reset the counter after each update
                    expsetup.stim.edata_trial_online_counter(ind1) = 99;
                else
                    fprintf('Waiting for task criterion of %.2f; Tracked performance is %.2f \n', goal_up, perf_correct)
                    expsetup.stim.exp_version_update_next_trial = 0;
                end
            end

            % If not enough trials acquired, do nothing
        elseif sum(ind_task) < trial_online_counter
            expsetup.stim.exp_version_update_next_trial = 0;
            fprintf('No task updating, waiting to acquire %.0f trials\n', trial_online_counter)
        end
        
    end
    
end

