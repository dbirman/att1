% Shows picture on the screen and waits for button press. 
%
% Input - picture matrix contained within exp directory
% Input - wait for keyborad (1) or not wait for keyboard (0)
% Input - initialized PsychToolbox screen properties
% Output - button press code
%
% v1.0 DJ: February 18, 2015


function y = runexp_instruction_v10 (instr_pic, wait_keyboard)


%% Settings

global expsetup
window = expsetup.screen.window;
dispcenter = expsetup.screen.dispcenter;
char='q'; % Make sure no letter is typed in


%% Load image & make texture

a1 = load (instr_pic); % Load image specified in isntrpic
a1 = struct2cell(a1); % Convert to cell for easines of use (no need to access structure fields)
instrsize = size(a1{1});
instrobj = CenterRectOnPoint([0, 0, instrsize(2), instrsize(1)], dispcenter(1), dispcenter(2)); % Instrsize(2) refers to x-number of columns!!!
tex=Screen('MakeTexture', window, a1{1});


%%  Select background color (same as top left corner color of the picture)

color1(1)=a1{1}(1,1,1);
color1(2)=a1{1}(1,1,2);
color1(3)=a1{1}(1,1,3);
Screen('FillRect', window, color1); % DDD - color of background, based on instrpic


%% Draw texture

Screen('DrawTexture',window, tex, [], instrobj, [], 0);
Screen('Flip', window);
Screen('Close', tex);


%% Wait for keyboard input to proceed

if nargin==2 && wait_keyboard==1
    InitChoicexx=0;
    WaitSecs(0.2);
    [~,keyCode, ~] = KbWait;
    char = KbName(keyCode);
    WaitSecs(0.1);
end


%% Output

if nargout == 1
    y = char;
end


end