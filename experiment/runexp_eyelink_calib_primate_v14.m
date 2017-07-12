% Run manual calibration
%
% target positions: 1 column - position on the cirlce, degrees; 
% 2 columns - X,Y coordinates in degrees
%
% v1.1 DJ: January 31, 2016
% v1.2 DJ: February 3, 2016. Fixes reward, adds blinking stimuli, rings
% v1.3 DJ: June 9, 2016. Uses only first channel to administer reward
% v1.4 Dj July 5, 2016. Now reward has a separate channel

function eyelink = runexp_eyelink_calib_primate_v14


%% Settings

global expsetup;
global ni;


%% Read screen settings

if exist ('expsetup') && isfield (expsetup, 'screen')
    try
        window = expsetup.screen.window;
        deg2pix = expsetup.screen.deg2pix;
        dispcenter = expsetup.screen.dispcenter;
        w_width = expsetup.screen.screen_rect(3);
        w_height = expsetup.screen.screen_rect(4);
        frame_ifi = expsetup.screen.ifi;
    catch
        error ('expsetup.screen matrix is used, but not all params are specified')
    end
else
    Screen('CloseAll');
    Eyelink('Shutdown');
    sprintf('\n\nProbably DJ will fix it in the next version\n\n')
    error ('Screen defaults not initialized, likely screen not open')
end



%% Read eyelink settings

if exist ('expsetup') && isfield (expsetup, 'eyelink')
    el = expsetup.eyelink; % Load current eyelink defaults
else   
    % Default display size
    disp_width = 520; % mm
    disp_height = 295; % mm
    view_dist = 600; % mm
    
    warning (sprintf('\n\neyelink_ini code was not loaded, initializing default settings'))
    warning (sprintf('default disp_width=%i, disp_height=%i and view_dist=%i are used', disp_width, disp_height, view_dist))
    
    % Initialize eyelink
    status = Eyelink('IsConnected');
    if status==0
        if EyelinkInit(0)
            fprintf('Eyelink successfully initialized.\n');
        elseif ~EyelinkInit(0)
            fprintf('Eyelink initialization was aborted.\n');
            Eyelink('ShutDown');
            Screen('CloseAll'); % Make sure to close all screens if eyelink initialization failed
            error('Eyelink did not initialize. Refer to runexp_eyelink_ini_vXX code');
        end;
    else
        fprintf('Eyelink already initialized.\n');
    end
    
    % Set defaults;
    el=EyelinkInitDefaults(window);
    % distance in mm from the center of the screen to  edge of screen [left top right bottom]
    Eyelink('Command','screen_phys_coords = %ld %ld %ld %ld ', round(-disp_width/2), round(disp_height/2), round(disp_width/2), round(-disp_height/2) );
    % Distance from center of eye to the center of the screen
    Eyelink('Command', 'screen_distance = %ld', round(view_dist));
    % sets  'screen_pixel_coords' field in physical.ini file on host computer
    Eyelink('Command','screen_pixel_coords = %ld %ld %ld %ld', 0, 0, w_width-1, w_height-1);
end




%% Read calibration settings

if exist ('expsetup') && isfield (expsetup, 'eyecalib')
    try
        backgroundcolor = expsetup.eyecalib.backgroundcolor;
        foregroundcolor = expsetup.eyecalib.foregroundcolor;
        target_shape = expsetup.eyecalib.target_shape;
        target_size_deg = expsetup.eyecalib.target_size_deg;
        target_positions = expsetup.eyecalib.target_positions;
        if size(target_positions,1)==1 && size(target_positions,2)>1
            target_positions=target_positions'; % Flip to format required
        end
        if size(target_positions,2)==1
            target_radius = expsetup.eyecalib.target_radius;
        end
        fixation_cond_blink = expsetup.eyecalib.fixation_cond_blink; % Is it blinkig target?
        fixation_cond_ring = expsetup.eyecalib.fixation_cond_ring; % is there ring plotted?
    catch
        Screen('CloseAll');
        Eyelink('Shutdown');
        error ('expsetup.eyecalib matrix is used, but not all params are specified')
    end
else
    sprintf('\n\nLoading default calibration settings, as matrix expsetup.eyecalib is not specified\n\n')
    backgroundcolor = [127, 127, 127];
    foregroundcolor = [200, 20, 20];
    target_shape = 'oval';
    target_size_deg = [0, 0, 1, 1];
    target_positions = [0:90:350]';
    target_radius = 0.8; % In screen size (80% screen size)
    fixation_cond_blink = 0;
    fixation_cond_ring = 1;
end

if isstr(foregroundcolor)
    foregroundcolor=[];
    foregroundcolor(1,:)=[200, 20, 20];
    foregroundcolor(2,:)=[200, 150, 20];
    foregroundcolor(3,:)=[70, 70, 200];
    foregroundcolor(4,:)=[200, 20, 200];
    foregroundcolor(5,:)=[20, 200, 20];
    foregroundcolor(6,:)=[255, 255, 200];
end

% Calculate target coordinates (in pixels)
if size(target_positions,2)==2
    Screen('CloseAll');
    Eyelink('Shutdown');
    error ('At the moment DJ has not written code to specify XY coordinates, use arc instead')
elseif size(target_positions,2)==1
    [cx1, cy1] = pol2cart(target_positions*pi/180, target_radius);
    cx = round(dispcenter(1) + w_height/2*[0, cx1']);
    cy = round(dispcenter(2) + w_height/2*[0, cy1']);
    % Coordinates of calibration targets: [center; randomized targets; center]
    coord1=[];
    coord1(1) = cx(1); % Center of the screen x
    coord1(2) = cy(1); % Center of the screen y
    crp = randperm(length(target_positions))+1; % Randomize and add 1 (so that not use [0,0] coordinate)
    coord1(3:2:length(target_positions)*2+1)=cx(crp);
    coord1(4:2:length(target_positions)*2+2)=cy(crp);
    coord1(length(target_positions)*2+3) = cx(1);
    coord1(length(target_positions)*2+4) = cy(1);
end



%% Send eyelink commands

Eyelink('command', 'clear_screen=0');
Eyelink('command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA,PUPIL');
Eyelink('command', 'generate_default_targets=NO');
Eyelink('command', 'enable_automatic_calibration','NO');
% a1 = ['HV' num2str(size(target_positions,1)+1)];
% Eyelink('Command', ['calibration_type=' a1]);
Eyelink('command', 'calibration_type=USER');
Eyelink('command', 'randomize_calibration_order 0');
Eyelink('command', 'randomize_validation_order 0');
Eyelink('command', 'cal_repeat_first_target 0');
Eyelink('command', 'val_repeat_first_target 0');
Eyelink('command', sprintf('calibration_samples=%i', size(target_positions,1)+2));
Eyelink('command', sprintf('validation_samples=%i', size(target_positions,1)+2));

% Send command stating sequence of the targets presented
c_s = [0:size(target_positions,1)+1]; % Range is [0 (first fixation), target_positions, +1 (last fixation)]
temp = 'calibration_sequence=';
for i=1:length(c_s)
    temp=sprintf('%s%i ', temp, c_s(i));
end
Eyelink('command', temp);

% Send target coordinates to eyetracker
temp = 'calibration_targets=';
for i=1:length(coord1)
    temp=sprintf('%s%i ', temp, coord1(i));
end
Eyelink('command', temp);

% Update defaults
EyelinkUpdateDefaults(el);

% Start setup and send the keypress 'c' to select "Calibrate"
Eyelink('StartSetup')
test1=1;
while test1~=2
    test1=Eyelink('CurrentMode');
end
Eyelink('SendKeyButton',double('c'), 0, el.KB_PRESS);



%% Initialize matrix with blinks/radius changes

% Plotting settings
fixation_blink_frequency = 12; % How many time blinks per second;
fixation_ring_size = 2.5; % How many times fixation size is the contracting ring
fixation_ring_frequency = 3;
ring_pen = 4; % How wide the ring is (pixels)
fixation_cond_blink = 0; % Is it blinkig target?
fixation_cond_ring = 1; % To show the ring?

% trialmat
cond1 = 7; % Columns
dur = 5; % How long the trial is/seconds
time_unit = frame_ifi;
b1 = ceil(dur/time_unit); % How many frames to initalize
trialmat = NaN(b1, cond1);

% Columns of trialmat
c1_time = 1; % What recorded time is it?
c1_eye_x = 2;
c1_eye_y = 3;
c1_eye_target = 4; % Which target was selected

% Stable fixation
c1_fix_on = 5;
trialmat(:,c1_fix_on)=1;

% Stable fixation
c1_fix_off = 8;
trialmat(:,c1_fix_off)=0;

% Blinking fixation
c1_fix_blink = 6;
b1 = 1/fixation_blink_frequency; % How many ms blink lasts
b1=round(b1/time_unit); % How many frames blink lasts
m1 = [ones(b1,1); zeros(b1,1)];
ans1=floor(size(trialmat,1)/length(m1));
m1 = repmat(m1,ans1,1);
trialmat(1:length(m1),c1_fix_blink)=m1;
if length(m1)<size(trialmat,1)
    trialmat(length(m1)+1:end,c1_fix_blink)=m1(end);
end

% Ring fixation
c1_ring_radius=7;
b1 = 1/fixation_ring_frequency; % How many ms blink lasts
b1=round(b1/time_unit); % How many frames blink lasts
if target_size_deg(3)==target_size_deg(4)
    sz=target_size_deg(3);
else
    sz=(target_size_deg(3)+target_size_deg(4))/2;
end
s1 = sz*fixation_ring_size;
m1 = linspace(s1, sz, b1);
ans1=floor(size(trialmat,1)/length(m1));
m1 = repmat(m1',ans1,1);
trialmat(1:length(m1),c1_ring_radius)=m1;
if length(m1)<size(trialmat,1)
    trialmat(length(m1)+1:end,c1_ring_radius)=0;
end

trialmat_copy = trialmat;


disp ('Button responses accepted: ')
disp ('space - accept fixation position')
disp ('r - give reward')
disp ('b - make target blink')
disp ('s - make target stationary')
disp ('h - hide target and  ring')
disp ('x - quit')
disp ('escape - quit')

%% Show stimuli onscreen

Screen('FillRect', window, backgroundcolor);
[~, time_current, ~]=Screen('Flip', window);

calibration_stop=0; target_number=0; fixation_color=[-1,-1,-1];

while calibration_stop==0
    
    target_number=target_number+1; % Which target is shown
    
    %===============
    % Retrieve expected eye coordinates
    % Loop is used to compensate for some unknown bug that does not
    % retrieve position immediatelly
    %==============
    
    result=0;
    tstart=GetSecs;
    while(result==0)
        [result, targ_x, targ_y] = Eyelink('TargetCheck');
        tend=GetSecs;
        if tend-tstart>0.5
            result=1; calibration_stop=1;
        end
    end
    
    if calibration_stop==0
        disp(sprintf(' \nTarget number %i at [%i,%i] \n', target_number, targ_x, targ_y));
    end
    
    %=======================
    % Present stimuli onscreen
    %=======================
    
    if calibration_stop==0
        
        % Prepare fixation coordinates
        sizepix = round(deg2pix * target_size_deg);
        pospix = [targ_x, targ_y];
        fixation_rect = CenterRectOnPoint(sizepix, pospix(1), pospix(2));
        
        % Initialize variables
        loop_over = 0;
        frame_index1 = 0;
        char = 'q'; 
        
        % Which conditions is it
        if fixation_cond_blink==1
            col1=c1_fix_blink;
        else
            col1=c1_fix_on;
        end
        
        % Select target color (allows randomization)
        l1=0;
        while l1==0
            a=randperm(length(foregroundcolor));
            a=foregroundcolor(a(1),:);
            if fixation_color~=a
                fixation_color=a;
                l1=1; 
            end
        end
        
        % Stimulus and response loop
        while loop_over==0
            
            % ================
            % Initialize new frame index
            frame_index1 = frame_index1+1;
            
            
            %=================
            % Fixation
            if trialmat(frame_index1,col1)==1
                if strcmp(target_shape,'circle')
                    Screen('FillArc', window,  fixation_color, fixation_rect, 0, 360);
                elseif strcmp(target_shape,'square')
                    Screen('FillRect', window,  fixation_color, fixation_rect, ring_pen);
                end
            end
            
            %================
            % Ring
            if fixation_cond_ring==1 && col1~=c1_fix_off
                
                xc=targ_x;
                yc=targ_y; % Revert y values
                
                % Center objects to be drawn on the grid cells.
                % 1 column - 1 object
                sizedeg = [0, 0, trialmat(frame_index1,c1_ring_radius), trialmat(frame_index1,c1_ring_radius)];
                sizepix = round(sizedeg*deg2pix);
                ring_rect=CenterRectOnPoint(sizepix, targ_x, targ_y);
                ring_rect = ring_rect'; % One column - one object
                
                % Plot
                Screen('FrameArc', window,  fixation_color, ring_rect, 0, 360, ring_pen);
            end
            
            %===================
            [~, time_current, ~]=Screen('Flip', window);

            
            %=====================
            % Check button presses
            
            [keyIsDown,timeSecs,keyCode] = KbCheck;
            % Record what kind of button was pressed
            char = KbName(keyCode);
            % Catch potential press of two buttons
            if iscell(char)
                char=char{1};
            end
            
            %=======================
            % Do something with button presses
            if strcmp(char,'space')
                Eyelink('AcceptTrigger');
                % Send reward signal
                if expsetup.general.reward_on==1
                    signal1 = linspace(expsetup.ni_daq.reward_voltage, expsetup.ni_daq.reward_voltage, expsetup.eyecalib.reward_ms)';
                    signal1 = [0; signal1; 0; 0; 0; 0; 0];
                    queueOutputData(ni.session_reward, signal1);
                    ni.session_reward.startForeground;
                end
                loop_over = 1;
            elseif strcmp(char,'r')
                % Send reward signal
                if expsetup.general.reward_on==1
                    signal1 = linspace(expsetup.ni_daq.reward_voltage, expsetup.ni_daq.reward_voltage, expsetup.eyecalib.reward_ms)';
                    signal1 = [0; signal1; 0; 0; 0; 0; 0];
                    queueOutputData(ni.session_reward, signal1);
                    ni.session_reward.startForeground;
                end
            elseif strcmp(char,'b')
                col1=c1_fix_blink;
            elseif strcmp(char,'s')
                col1=c1_fix_on;
            elseif strcmp(char,'h')
                col1=c1_fix_off;
            elseif strcmp(char,'ESCAPE') || strcmp(char,'x');
                loop_over = 1;
                calibration_stop=1;
            end
            
            %  Restart trial if no button was pressed
            if frame_index1==size(trialmat,1)
                frame_index1 = 0;
            end
            
        end
        
        % Show blank screen & wait a bit
        Screen('FillRect', window, backgroundcolor);
        [~, time_current, ~]=Screen('Flip', window);
        WaitSecs(1);
        
    end
    
end



