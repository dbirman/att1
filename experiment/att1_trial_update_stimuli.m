% Create a structure TV which will be used to update stimulus values

%============

tv1 = struct; % Temporary Variable (TV)
tv1(1).single_step_update = 2;


% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'increase probe isi')
    tv1(1).temp_var_final = nanmean(expsetup.stim.att_cue_fix_soa);
    tv1(1).temp_var_ini = expsetup.stim.att_cue_fix_soa_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.att_cue_fix_soa_ini_step;
    tv1(1).name = 'esetup_att_cue_fix_soa';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(2).temp_var_final = nanmean(expsetup.stim.probe_isi);
    tv1(2).temp_var_ini = expsetup.stim.probe_isi_ini;
    tv1(2).temp_var_ini_step = expsetup.stim.probe_isi_ini_step;
    tv1(2).temp_var_current = NaN; % This value will be filled up
    tv1(2).name = 'esetup_probe_isi';
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'decrease att cue length')
    tv1(1).temp_var_final = nanmean(expsetup.stim.att_cue_length);
    tv1(1).temp_var_ini = expsetup.stim.att_cue_length_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.att_cue_length_ini_step;
    tv1(1).name = 'esetup_att_cue_length';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'introduce distractors')
    tv1(1).temp_var_final = nanmean(expsetup.stim.distractor_contrast);
    tv1(1).temp_var_ini = expsetup.stim.distractor_contrast_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.distractor_contrast_ini_step;
    tv1(1).name = 'esetup_distractor_contrast';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'lever hold training')
    tv1(1).temp_var_final = nanmean(expsetup.stim.train_trial_duration);
    tv1(1).temp_var_ini = expsetup.stim.train_trial_duration_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.train_trial_duration_ini_step;
    tv1(1).name = 'esetup_train_trial_duration';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'release lever on long ring')
    tv1(1).temp_var_final = nanmean(expsetup.stim.response_ring_duration);
    tv1(1).temp_var_ini = expsetup.stim.response_ring_duration_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.response_ring_duration_ini_step;
    tv1(1).name = 'esetup_response_ring_duration';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'release lever on big ring')
    tv1(1).temp_var_final = nanmean(expsetup.stim.response_ring_size_start);
    tv1(1).temp_var_ini = expsetup.stim.response_ring_size_start_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.response_ring_size_start_ini_step;
    tv1(1).name = 'esetup_response_ring_size_start';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

% Select variables to be modified
if strcmp(expsetup.stim.exp_version_temp, 'reduce target size')
    tv1(1).temp_var_final = nanmean(expsetup.stim.att_stim_size);
    tv1(1).temp_var_ini = expsetup.stim.att_stim_size_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.att_stim_size_ini_step;
    tv1(1).name = 'esetup_att_stim_size';
    tv1(1).temp_var_current = NaN; % This value will be filed up
    tv1(1).single_step_update = 1;
end

