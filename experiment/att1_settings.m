% All possible experimental settings within this file;
%
% Produces stim structure which contains all stimuli settings and trial
% definitions

%% Different training stages have different stim durations

stim.training_stage_matrix = [1,2,3,4,5]; % Different levels of task difficulty. Select to use one.

% Training stage 1.0
% Is final stage. No changes to the code

% Training stage 2.0
% Increase probe isi (it becomes a memory task)
stim.att_cue_fix_soa_ini = 1.5;
stim.att_cue_fix_soa_ini_step = -0.1; % Decrease fix soa
stim.probe_isi_ini = 0.01;
stim.probe_isi_ini_step = 0.1; % Increase probe isi

% Training stage 3.0
% Shorten central cue
stim.att_cue_length_ini = 7; % dva
stim.att_cue_length_ini_step = -3; % decrease

% Training stage 4.0
% Add distractors of varying contrast
stim.distractor_contrast_ini = 0.05;
stim.distractor_contrast_ini_step = 0.5;

% Training stage 5.0
% Just collect a bunch of single probe responses
stim.stage_5_ini = 1;
stim.stage_5_ini_step = 5;
stim.stage_5_final = 10;


%% Stimuli

%==============
% Noise stimulus
stim.noise_background_position = [0,0];
stim.noise_background_size = [0,0,20,15]; 

%==============
% Fixation

stim.fixation_size = [0,0,0.5,0.5]; % Size of fixation (degrees)
stim.fixation_position = [0,0]; % Degrees, position on a circle
stim.fixation_color_baseline = [20,20,200]; % Color of fixation or text on the screen
stim.fixation_shape_baseline = 'circle';
stim.fixation_pen = 4; % Fixation outline thickness (pixels)
stim.fixation_blink_frequency = 2; % How many time blinks per second;

% Fixation duration
stim.fixation_acquire_duration = [30]; % How long to show fixation before it is acquired
stim.fixation_maintain_duration = [0.5]; % Time to maintain target before stuff starts happening

%==============
% Attention cue
stim.att_cue_fix_soa = 0.5; % Soa between fix acquired and att cue
stim.att_cue_duration = 0.3;
stim.att_cue_length = 0.75; % dva
stim.att_cue_pen = 10; % pixels


%===================
% Attention stimuli

stim.att_stim_number_ini = 1; 
stim.att_stim_number_max = 4; % Max number of Gabor patches in the setup
stim.att_stim_position = [45:90:360];
stim.att_stim_reps = 2; % How many att stim repetitions
stim.att_stim_reps_probe = 2; % On which repetition probe appears
stim.att_stim_radius = 7;
stim.att_stim_size = 5; 
stim.att_stim_angle = [0:10:170];

%====================
% Probes

stim.probe_cue_isi = [0.3]; % Time from cue onset to probe onset
stim.probe_duration = [0.5]; % Duration
stim.probe_isi = 1; % Time between two probe repetitions
stim.probe_to_response_isi = [0.5];

stim.probe_size = stim.att_stim_size;
stim.probe_radius = stim.att_stim_radius;
stim.probe_angle_diff = [-50, -40, -30, 30, 40, 50];
stim.probe_change = [1, 2]; % 1 is no change; 2 is change
stim.probe_change_meaning = {'same', 'change'};

% Contrast of probes and distractors
stim.probe_contrast = 1;
stim.distractor_contrast = 1;

%==============
% Gabors

stim.gabor_frequency = 1;
% stim.gabor_contrast = 1; % range is 0-1;
stim.gabor_phase = []; % Phase of the grating (0:1)
stim.gabor_bgluminance = 0.5; % Background luminace (as stimuli are shown on gray)
stim.gabor_sigma_period = (stim.att_stim_size*stim.gabor_frequency)/5; % Periods covered by one STD of gausian


%==============
% Response ring

stim.response_ring_duration = 0.9; % Seconds to give response
stim.response_ring_isi = 0.1; % Time interval between two rings
stim.response_ring_size_start = 4; % How many degrees is ring size
stim.response_ring_size_end = 0; % How many degrees is ring size
stim.response_ring_pen = 4; % How wide the ring is (pixels)

stim.response_ring_position = [0]; % Angle, on a circle
stim.response_ring_radius = [0]; % Radius
stim.response_ring_sequence = [1,2]; % Different ring colors

% Response ring color
% 1 is no change
% 2 is change
c1 = [255, 255, 255];
c2 = [20, 255, 20];

if strcmp(expsetup.general.subject_id, 'aq')
    stim.response_ring_color1 = c2;
    stim.response_ring_color2 = c1;
elseif strcmp(expsetup.general.subject_id, 'hb')
    stim.response_ring_color1 = c1;
    stim.response_ring_color2 = c2;
elseif strcmp(expsetup.general.subject_id, 'jw')
    stim.response_ring_color1 = c2;
    stim.response_ring_color2 = c1;
else
    stim.response_ring_color1 = c1;
    stim.response_ring_color2 = c2;
end

%==============
% Screen colors
stim.background_color = [127, 127, 127];

%===============
% Duration of trials
stim.trial_dur_intertrial = 0.5; % Blank screen at the end
stim.trial_dur_intertrial_error = 2; % Blank screen at the end

%===============
% Other

% Staircase
stim.trial_online_counter = 3; % How many trials to count for updating task difficulty
stim.trial_correct_goal_up = 3; % What is accuracy to make task harder
stim.trial_correct_goal_down = 2; % What is accuracy to make task harder

% Other
stim.trial_error_repeat = 1; % 1 - repeats same trial if error occured immediatelly; 0 - no repeat
stim.trial_abort_counter = 20; % Quit experiment if trials in a row are aborted
stim.plot_every_x_trial = 1; % Every which trial to plot (every 1, 2nd, 10th etc trial)


%% Do not forget to update for each epxeriment: 

% Specify column names for expmatrix

stim.esetup_exp_version = NaN; % Which task participant is doing

stim.esetup_fix_coord1 = NaN;  % Fixation x position
stim.esetup_fix_coord2 = NaN;  % Fixation y position
stim.esetup_fixation_acquire_duration = NaN;
stim.esetup_fixation_maintain_duration = NaN;

stim.esetup_probe_position = NaN; % deg of arc
stim.esetup_probe_radius = NaN;
stim.esetup_probe_angle = NaN; 
stim.esetup_probe_change = NaN; % Angle changes or not
stim.esetup_probe_duration = NaN; % Angle changes or not
stim.esetup_probe_isi = NaN; % Angle changes or not

stim.esetup_probe_contrast = NaN;
stim.esetup_distractor_contrast = NaN;

stim.esetup_att_stim_number = NaN; 

stim.esetup_att_cue_fix_soa = NaN;
stim.esetup_att_cue_position = NaN; 
stim.esetup_att_cue_length = NaN; 

stim.esetup_response_ring_position = NaN;  
stim.esetup_response_ring_radius = NaN;  
stim.esetup_response_ring_sequence(1,1:2) = NaN; % Which color of the ring shown first

stim.esetup_lever_maintain_duration = NaN;  % Fixation size for eye-tracking

stim.esetup_stage_5 = NaN; % Practice of the task

stim.edata_first_display = NaN; 
stim.edata_lever_on = NaN;
stim.edata_lever_off = NaN;
stim.edata_lever_duration = NaN;
stim.edata_lever_response = NaN;

stim.edata_fixation_on = NaN; % Stimulus time
stim.edata_fixation_acquired = NaN; % Stimulus time

stim.edata_probe_on = NaN; % Stimulus confirmation
stim.edata_probe_off = NaN; % Stimulus time

stim.edata_att_cue_on = NaN; % Stimulus confirmation

stim.edata_response_ring1_on = NaN;
stim.edata_response_ring2_on = NaN;
stim.edata_response_ring_off = NaN;

stim.edata_loop_over = NaN; % Stimulus time

% Monitoring performance
stim.edata_error_code = cell(1); % Error codes
stim.edata_error_code{1} = NaN;

stim.edata_trial_abort_counter = NaN;
stim.edata_trial_online_counter = NaN; % Error code

% Initialize gabor patch properties
t1 = stim.att_stim_number_max;
t2 = stim.att_stim_reps; 
stim.esetup_att_stim_position(1, 1:t1, 1:t2) = NaN;
stim.esetup_att_stim_radius(1, 1:t1, 1:t2) = NaN;
stim.esetup_att_stim_size(1, 1:t1, 1:t2) = NaN;
stim.esetup_att_stim_angle(1, 1:t1, 1:t2) = NaN;
stim.esetup_att_stim_phase(1, 1:t1, 1:t2) = NaN;
stim.esetup_att_stim_contrast(1, 1:t1, 1:t2) = NaN;


%%  Initialize frames matrices (one trial - one cell; one row - one frame onscreen)

% Timingn and eye position
stim.eframes_time{1}(1) = NaN;
stim.eframes_eye_x{1}(1) = NaN;
stim.eframes_eye_y{1}(1) = NaN;

% Background & fixation
stim.eframes_background_on{1}(1) = NaN;
stim.eframes_fix_blink{1}(1) = NaN;

% Central cue
stim.eframes_att_cue_on_temp{1}(1) = NaN;
stim.eframes_att_cue_on{1}(1) = NaN;
stim.eframes_att_cue_off_temp{1}(1) = NaN;
stim.eframes_att_cue_off{1}(1) = NaN;

% Gabor patches
stim.eframes_att_stim_on_temp{1}(1) = NaN;
stim.eframes_att_stim_on{1}(1) = NaN;

% Probe on/off
stim.eframes_probe_on_temp{1}(1) = NaN;
stim.eframes_probe_on{1}(1) = NaN;
stim.eframes_probe_off_temp{1}(1) = NaN;
stim.eframes_probe_off{1}(1) = NaN;

% Response ring on/off
stim.eframes_response_ring_size_temp{1}(1) = NaN;
stim.eframes_response_ring_size{1}(1) = NaN;
stim.eframes_response_ring_on_temp{1}(1) = NaN;
stim.eframes_response_ring_on{1}(1) = NaN;
stim.eframes_response_ring_off_temp{1}(1) = NaN;
stim.eframes_response_ring_off{1}(1) = NaN;


%% Save into expsetup

expsetup.stim=stim;


