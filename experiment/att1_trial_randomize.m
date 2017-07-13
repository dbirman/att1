% Randomized all parameters for the trial


%% Initialize NaN fields of all settings

% New trial initialized
if tid == 1
    % Do nothing
else
    f1 = fieldnames(expsetup.stim);
    ind = strncmp(f1,'esetup', 6) |...
        strncmp(f1,'edata', 5);
    for i=1:numel(ind)
        if ind(i)==1
            if ~iscell(expsetup.stim.(f1{i}))
                [m,n,o]=size(expsetup.stim.(f1{i}));
                expsetup.stim.(f1{i})(tid,1:n,1:o) = NaN;
            elseif iscell(expsetup.stim.(f1{i}))
                expsetup.stim.(f1{i}){tid} = NaN;
            end
        end
    end
end

%% Which exp version is running?

expsetup.stim.esetup_exp_version(tid,1) = expsetup.stim.exp_version_temp;

%%  Fix

% Fixation position
expsetup.stim.esetup_fix_coord1(tid,1) = expsetup.stim.fixation_position(1);
expsetup.stim.esetup_fix_coord2(tid,1) = expsetup.stim.fixation_position(2);

% Fixation acquire duration
temp1=Shuffle(expsetup.stim.fixation_acquire_duration);
expsetup.stim.esetup_fixation_acquire_duration(tid,1) = temp1(1);

% Fixation maintain duration
temp1=Shuffle(expsetup.stim.fixation_maintain_duration);
expsetup.stim.esetup_fixation_maintain_duration(tid,1) = temp1(1);


%% Att

% How many attention stimuli
if expsetup.stim.esetup_exp_version(tid, 1) > 4
    t1 = expsetup.stim.att_stim_number_ini;
    expsetup.stim.esetup_att_stim_number(tid,1) = t1;
elseif expsetup.stim.esetup_exp_version(tid, 1) <= 4
    t1 = expsetup.stim.att_stim_number_max;
    expsetup.stim.esetup_att_stim_number(tid,1) = t1;
end

% How many attention repetitions
t2 = expsetup.stim.att_stim_reps;

% Select locations
a = Shuffle(expsetup.stim.att_stim_position);
temp1 = a(1:expsetup.stim.esetup_att_stim_number(tid,1)); % Select as many objects as there are

% Determine the coordinates/size for each att stimulus repetition
for j = 1:t2
    expsetup.stim.esetup_att_stim_position(tid, 1:t1, j) = temp1;
    expsetup.stim.esetup_att_stim_radius(tid, 1:t1, j) = expsetup.stim.att_stim_radius;
    expsetup.stim.esetup_att_stim_size(tid, 1:t1, j) = expsetup.stim.att_stim_size;
end

% Probe location and radius is location no1
expsetup.stim.esetup_probe_position(tid,1) = expsetup.stim.esetup_att_stim_position(tid,1,1);
expsetup.stim.esetup_probe_radius(tid,1) = expsetup.stim.esetup_att_stim_radius(tid,1,1);

% Determine the tilt angle for each att stimulus
temp1=Shuffle(expsetup.stim.att_stim_angle);
for j = 1:t2
    if numel(expsetup.stim.att_stim_angle) >= expsetup.stim.esetup_att_stim_number(tid,1)
        expsetup.stim.esetup_att_stim_angle(tid,1:t1,j) = temp1(1:t1);
    else
        for i = 1:t1
            temp1=Shuffle(expsetup.stim.att_stim_angle);
            expsetup.stim.esetup_att_stim_angle(tid,i,j) = temp1(1);
        end
    end
end

% Probe tilt angle
c1=Shuffle(expsetup.stim.probe_change);
if c1(1)==1 % No probe angle change
    expsetup.stim.esetup_probe_change(tid,1) = c1(1); % Save whether it is probe change or no change
elseif c1(1)==2 % Probe angle change
    a = Shuffle(expsetup.stim.probe_angle_diff); % Difference from probe angle
    b = expsetup.stim.esetup_att_stim_angle(tid,1,1); % First element is probe angle
    temp1 = a(1)+b;
    if temp1<0
        temp1 = temp1+180;
    elseif temp1>=180
        temp1 = temp1-180;
    end
    expsetup.stim.esetup_probe_angle(tid,1) = temp1; % Save probe angle
    j = expsetup.stim.att_stim_reps_probe;
    expsetup.stim.esetup_att_stim_angle(tid,1,j) = temp1; % Over-write with the new probe angle
    expsetup.stim.esetup_probe_change(tid,1) = c1(1); % Save whether it is probe change or no change
end


% Determine phase of the gabor patch
for i = 1:t1
    for j = 1:t2
        expsetup.stim.esetup_att_stim_phase(tid,i,j) = randn;
    end
end

%==========
% Determine contrast of the probe
expsetup.stim.esetup_probe_contrast(tid,1) = expsetup.stim.probe_contrast;

% Distractgor contrast varies as a function of training
if expsetup.stim.esetup_exp_version(tid, 1) < 4
    temp1 = Shuffle(expsetup.stim.distractor_contrast);
elseif expsetup.stim.esetup_exp_version(tid, 1) == 4
    temp1 = Shuffle(tv1(1).temp_var_current);
elseif expsetup.stim.esetup_exp_version(tid, 1) > 4
    temp1 = 0; % No distractors
end
expsetup.stim.esetup_distractor_contrast(tid,1) = temp1(1);


for i = 1:t1
    for j = 1:t2
        if i==1 % Probe contrast
            expsetup.stim.esetup_att_stim_contrast(tid,i,j) = expsetup.stim.esetup_probe_contrast(tid,1);
        else  % Distractor contrast
            expsetup.stim.esetup_att_stim_contrast(tid,i,j) = expsetup.stim.esetup_distractor_contrast(tid,1);
        end
    end
end


%% Probe duration

% Probe duration
temp1=Shuffle(expsetup.stim.probe_duration);
expsetup.stim.esetup_probe_duration(tid,1) = temp1(1);

% Probe isi varies as a function of training
if expsetup.stim.esetup_exp_version(tid, 1) < 2
    temp1 = Shuffle(expsetup.stim.probe_isi);
elseif expsetup.stim.esetup_exp_version(tid, 1) == 2
    temp1 = Shuffle(tv1(2).temp_var_current);
elseif expsetup.stim.esetup_exp_version(tid, 1) > 2
    temp1 = Shuffle(expsetup.stim.probe_isi_ini);
end
expsetup.stim.esetup_probe_isi(tid,1) = temp1(1);


%% Attention cue

% Fixation - att cue soa varies as a function of training
if expsetup.stim.esetup_exp_version(tid, 1) < 2
    temp1 = Shuffle(expsetup.stim.att_cue_fix_soa);
elseif expsetup.stim.esetup_exp_version(tid, 1) == 2
    temp1 = Shuffle(tv1(1).temp_var_current);
elseif expsetup.stim.esetup_exp_version(tid, 1) > 2
    temp1 = Shuffle(expsetup.stim.att_cue_fix_soa_ini);
end
expsetup.stim.esetup_att_cue_fix_soa(tid,1) = temp1(1);

%=======
% Attention cue length - varies as a function of training
if expsetup.stim.esetup_exp_version(tid, 1) < 3
    temp1 = Shuffle(expsetup.stim.att_cue_length);
elseif expsetup.stim.esetup_exp_version(tid, 1) == 3
    temp1 = Shuffle(tv1(1).temp_var_current);
elseif expsetup.stim.esetup_exp_version(tid, 1) > 2
    temp1 = Shuffle(expsetup.stim.att_cue_length_ini);
end
expsetup.stim.esetup_att_cue_length(tid,1) = temp1(1);

% Attention cue position (angle on a circle)
% Same as probe position
expsetup.stim.esetup_att_cue_position(tid,1) = expsetup.stim.esetup_probe_position(tid,1);


%% Response rings

% Response position (angle on a circle)
temp1=Shuffle(expsetup.stim.response_ring_position);
expsetup.stim.esetup_response_ring_position(tid,1) = temp1(1);

% Response radius (distance from center)
temp1=Shuffle(expsetup.stim.response_ring_radius);
expsetup.stim.esetup_response_ring_radius(tid,1) = temp1(1);

% Response ring sequence on the trial

temp1=Shuffle(expsetup.stim.response_ring_sequence);
expsetup.stim.esetup_response_ring_sequence(tid,:) = temp1;


%% Stage 5 - practice of the task progressing
% Only monitored for stage 5

if expsetup.stim.esetup_exp_version(tid, 1) == 5
    temp1 = Shuffle(tv1(1).temp_var_current);
end
expsetup.stim.esetup_stage_5(tid,1) = temp1(1);


%% If previous trial was an error, then copy settings of the previous trial

if tid>1
    if expsetup.stim.trial_error_repeat == 1 % Repeat error trial immediately
        if  ~strcmp(expsetup.stim.edata_error_code{tid-1}, 'correct')
            f1 = fieldnames(expsetup.stim);
            ind = strncmp(f1,'esetup', 6);
            for i=1:numel(ind)
                if ind(i)==1
                    if ~iscell(expsetup.stim.(f1{i}))
                        [m,n,o]=size(expsetup.stim.(f1{i}));
                        expsetup.stim.(f1{i})(tid,1:n,1:o) = expsetup.stim.(f1{i})(tid-1,1:n,1:o);
                    elseif iscell(expsetup.stim.(f1{i}))
                        expsetup.stim.(f1{i}){tid} = NaN;
                    end
                end
            end
        end
    end
end

