% Function creates a subject folder. Automatically checks which repetition
% of the experiment it is and suggests to save session number
%
% Prespecified screen size and distance are necessary! Also open
% psychtoolbox window is necessary
% Multiple outputs as structure 'eyelink' containing all initalized eyelink
% properties
%
% v1.0 DJ: February 11, 2015
% v1.1 DJ: January 26, 2016 small improvements

function eyelink = runexp_eyelink_ini_v11


%% Settings

global expsetup;

% edf
edfname1 = expsetup.general.edfname_temporary;

% Display settings
window = expsetup.screen.window; % Screen handle
screen_rect = expsetup.screen.screen_rect; % pixels

% Display settings
disp_width = expsetup.general.monwidth_cm*10; % mm
disp_height = expsetup.general.monheight_cm*10; % mm
view_dist = expsetup.general.viewingdist_cm*10; % mm

% Colors
white = expsetup.screen.white;
black = expsetup.screen.black;
gray = expsetup.screen.gray;


%% Initialize connection

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

%% Initialize eyelink defaults

el=EyelinkInitDefaults(window);


%% Change some default settings

el.targetbeep=0;
el.feedbackbeep=0;
el.backgroundcolour = gray;
el.foregroundcolour = black;

EyelinkUpdateDefaults(el);


%%  Open file to record data to

d1=Eyelink('Openfile', edfname1);
if d1~=0
    fprintf('Cannot create EDF file ''%s'' ', edfname1);
    Eyelink('ShutDown');
    Screen('CloseAll');
    error('Not possible to start experiment as creating edf file failed');
    return;
end

%% Provide eyelink with physcial screen size parameters

% distance in mm from the center of the screen to  edge of screen [left top right bottom]
Eyelink('Command','screen_phys_coords = %ld %ld %ld %ld ', round(-disp_width/2), round(disp_height/2), round(disp_width/2), round(-disp_height/2) );

% Distance from center of eye to the center of the screen
Eyelink('Command', 'screen_distance = %ld', round(view_dist));

% sets  'screen_pixel_coords' field in physical.ini file on host computer
Eyelink('Command','screen_pixel_coords = %ld %ld %ld %ld', 0, 0, screen_rect(3)-1, screen_rect(4)-1);


%% Output

expsetup.eyelink = el;


end