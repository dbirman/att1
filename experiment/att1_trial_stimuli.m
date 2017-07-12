% Prepare all the rectangles for the trial

%% Fixation converted to pixels

coord1=[];
coord1(1,1) = expsetup.stim.esetup_fix_coord1(tid,1);
coord1(2,1) = expsetup.stim.esetup_fix_coord2(tid,1);
sz1 = expsetup.stim.fixation_size;

fixation_rect = runexp_convert_deg2pix_rect_v10(coord1, sz1);


%% Central cue for attention

% Att cue position start (same as fixation position)
coord1=[];
coord1(1,1) = expsetup.stim.esetup_fix_coord1(tid,1);
coord1(2,1) = expsetup.stim.esetup_fix_coord2(tid,1);
rect1 = runexp_convert_deg2pix_coord_v10(coord1); % One column - one object;

% Att cue position endpoint (variable on each trial)
pos1 = expsetup.stim.esetup_att_cue_position(tid,1);
rad1 = expsetup.stim.esetup_att_cue_length(tid,1);
[xc, yc] = pol2cart(pos1*pi/180, rad1); % Convert to cartesian
coord1=[];
coord1(1)=xc; coord1(2)=yc;
rect2 = runexp_convert_deg2pix_coord_v10(coord1); % One column - one object;

att_cue_rect = [rect1; rect2];


%% All Gabor locations converted to pix

t1 = expsetup.stim.esetup_att_stim_number(tid,1);
pos1 = expsetup.stim.esetup_att_stim_position(tid,1:t1);
rad1 = expsetup.stim.esetup_att_stim_radius(tid,1:t1);
[xc, yc] = pol2cart(pos1*pi/180, rad1); % Convert to cartesian
coord1=[];
coord1(1,:)=xc; coord1(2,:)=yc; % One column, one object

% Probe size

a = expsetup.stim.esetup_att_stim_size(tid,1:t1);
sz1 = [];
sz1(1:t1,1) = 0; % One row - one set of coordinates (psychtoolbox requirement)
sz1(1:t1,2) = 0;
sz1(1:t1,3) = a;
sz1(1:t1,4) = a;

att_stim_rect = runexp_convert_deg2pix_rect_v10(coord1, sz1); % One column - one object;


%% Gabor patch texture

% Target gabors
t1 = expsetup.stim.esetup_att_stim_number(tid,1);
angle1 = expsetup.stim.esetup_att_stim_angle(tid,1:t1,:);
phase1 = expsetup.stim.esetup_att_stim_phase(tid,1:t1,:);
contrast1 = expsetup.stim.esetup_att_stim_contrast(tid,1:t1,:);

texture_att_stim = cell(1);

for i = 1:size(angle1,2)
    for j = 1:size(angle1,3)
        
        % Settings
        gabor_contrast = contrast1(1,i,j);
        gabor_angle = angle1(1,i,j);
        gabor_phase = phase1(1,i,j);
        gabor_size = [att_stim_rect(3,i)-att_stim_rect(1,i), att_stim_rect(4,i)-att_stim_rect(2,i)];
        gabor_size = round(gabor_size);
        pixelsPerPeriod = round(expsetup.screen.deg2pix/expsetup.stim.gabor_frequency); % How many pixels will each period/cycle occupy?
        gabor_sigma_period = expsetup.stim.gabor_sigma_period;
        gabor_bgluminance = expsetup.stim.gabor_bgluminance;
        
        % Grating
        TargetGrating = GenerateGrating(gabor_size(1), gabor_size(2), gabor_angle, pixelsPerPeriod, gabor_phase, gabor_contrast);
        
        sigma=pixelsPerPeriod*gabor_sigma_period;
        aperture = GenerateGaussian(gabor_size(1), gabor_size(2), sigma, sigma, 0, 0, 0); % Make Gaussian aperture = Gabors
        TargetShift = TargetGrating + gabor_bgluminance; % Shifts values to around mean luminance
        TargetShift(TargetShift>1)=1; TargetShift(TargetShift<0)=0;
        TargetCOR = TargetShift.*255; % Convert to correct range
        
        Target=[];
        Target(:,:,1) = (TargetCOR.*1);
        Target(:,:,2) = (TargetCOR.*1);
        Target(:,:,3) = (TargetCOR.*1);
        Target(:,:,4) = aperture.*255; % Set alpha values for aperture within a circular window
        
        texture_att_stim {1,i,j} = Screen('MakeTexture', window, Target); % Make texture
        
    end
end


%% Background texture

coord1=[];
coord1 = expsetup.stim.noise_background_position;
sz1 = expsetup.stim.noise_background_size;

background_rect = runexp_convert_deg2pix_rect_v10(coord1, sz1);

noiseimg=(128*randn(round(background_rect(4)), round(background_rect(3))) + 128); % DDD noise color and distribution
texture_backgroud = cell(1);
texture_backgroud{1} = Screen('MakeTexture', window, noiseimg);


%% Flash for photodiode

if expsetup.general.record_plexon==1
    sz1 = 110;
    d1_rect = [expsetup.screen.screen_rect(3)-sz1, 1, expsetup.screen.screen_rect(3), sz1]';
    % Rename
    ph_rect=d1_rect;
end
