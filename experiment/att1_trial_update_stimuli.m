% Create a structure TV which will be used to update stimulus values

%============

tv1 = struct; % Temporary Variable (TV)

% Select variables to be modified
if expsetup.stim.epx_version_temp == 1
    % No more variables to modify
end

% Select variables to be modified
if expsetup.stim.epx_version_temp == 2
    tv1(1).temp_var_final = expsetup.stim.att_cue_fix_soa;
    tv1(1).temp_var_ini = expsetup.stim.att_cue_fix_soa_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.att_cue_fix_soa_ini_step;
    tv1(1).temp_var_current = NaN;
    tv1(1).name = 'esetup_att_cue_fix_soa';
    tv1(2).temp_var_final = expsetup.stim.probe_isi;
    tv1(2).temp_var_ini = expsetup.stim.probe_isi_ini;
    tv1(2).temp_var_ini_step = expsetup.stim.probe_isi_ini_step;
    tv1(2).name = 'esetup_probe_isi';
    tv1(2).temp_var_current = NaN;
end

% Select variables to be modified
if expsetup.stim.epx_version_temp == 3
    tv1(1).temp_var_final = expsetup.stim.att_cue_length;
    tv1(1).temp_var_ini = expsetup.stim.att_cue_length_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.att_cue_length_ini_step;
    tv1(1).name = 'esetup_att_cue_length';
    tv1(1).temp_var_current = NaN;
end

% Select variables to be modified
if expsetup.stim.epx_version_temp == 4
    tv1(1).temp_var_final = expsetup.stim.distractor_contrast;
    tv1(1).temp_var_ini = expsetup.stim.distractor_contrast_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.distractor_contrast_ini_step;
    tv1(1).name = 'esetup_distractor_contrast';
    tv1(1).temp_var_current = NaN;
end

% Select variables to be modified
if expsetup.stim.epx_version_temp == 5
    tv1(1).temp_var_final = expsetup.stim.stage_5_final;
    tv1(1).temp_var_ini = expsetup.stim.stage_5_ini;
    tv1(1).temp_var_ini_step = expsetup.stim.stage_5_ini_step;
    tv1(1).name = 'esetup_stage_5';
    tv1(1).temp_var_current = NaN;
end
