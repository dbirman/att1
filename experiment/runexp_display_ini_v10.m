% Function creates a subject folder. Automatically checks which repetition
% of the experiment it is and suggests to save session number
%
% Prespecified screen size and distance are necessary!
% Multiple outputs as structure 'screen' containing all initalized display
% properties
%
% v1 DJ: February 11, 2015


function runexp_display_ini_v10

%% Settings

global expsetup
try
    monwidth_cm = expsetup.general.monwidth_cm;
    viewingdist_cm = expsetup.general.viewingdist_cm;
    fps_expected = expsetup.general.fps_expected;
catch
    error ('Not all variables are provided for function runexp_display_ini, function terminated')
end


%% Initialize fullscreen display 

% On iMac disable the synctest (as it fails)
if ismac
    Screen('Preference','SkipSyncTests', 1)
end

screen_number=max(Screen('Screens'));
% screen_number=0;
[window, screen_rect] = Screen('OpenWindow', screen_number, 0,[],32,2);


%% Setup blend function

Screen(window,'BlendFunction',GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


%% Determine display center

[dispcenter(1), dispcenter(2)] = RectCenter(screen_rect); % Display center


%% Find deg2pix

deg2pix = (screen_rect(3)-screen_rect(1)) / (2*atand(monwidth_cm/(viewingdist_cm*2))); % Pixels per degree for that monitor


%% Setup screen colors

white=WhiteIndex(window); % White
black=BlackIndex(window); % Black
gray=(white+black)/2; % Gray


%% Determine display speed

fps = Screen('FrameRate',window); % Frames per second
ifi = Screen('GetFlipInterval', window); % Inter-frame interval
if fps == 0
    fps = 1/ifi;
end


%% Quit experiment if refresh rate is wrong

if fps~=0 && ~isempty(fps_expected)
    if round(fps)==fps_expected
        fprintf('\nRunning experiment at expected %i Hz refresh rate \n', fps_expected);
    else
        Screen ('CloseAll');
        fprintf ('\nExpected refresh rate is %i Hz\n', fps_expected);
        error ('Expected refresh rate does not match refresh rate of the display')
    end
else
    fprintf('\nScreen refresh rate can not be determined. Check your setup settings file \n');
end


%% Outputs

expsetup.screen.screen_number = screen_number;
expsetup.screen.window = window;
expsetup.screen.screen_rect = screen_rect;
expsetup.screen.dispcenter(1)=dispcenter(1);
expsetup.screen.dispcenter(2)=dispcenter(2);
expsetup.screen.deg2pix = deg2pix;
expsetup.screen.white = white;
expsetup.screen.black = black;
expsetup.screen.gray = gray;
expsetup.screen.fps = fps;
expsetup.screen.ifi = ifi;


end