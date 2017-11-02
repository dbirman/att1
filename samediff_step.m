function ret = samediff_step( correct )
%SAMEDIFFERENT_STEP Step a same different simulation code forward
%   Returns a 32x32x3x140 trial representing a view of the screen as a same
%   different psychophysics task runs. The task runs in the following way:
%       20 frames: blank
%       10 frames: color1
%       20 frames: blank
%       10 frames: color2
%       20 frames: blank
%       10 frames: response1
%       20 frames: blank
%       10 frames: response2
%       20 frames: blank
%
%   Color 1 and color 2 are either same (both red/yellow) or different (red and
%   yellow)
%   Blank periods are grey
%   Response 1 and 2 are either white (same) or green (different)
%   
%   INPUT:
%       Correct - whether the agent got the last trial correct (use for
%                 tracking but not actually relevant)
%   OUTPUT:
%       RET - A cell array with:
%           frames - a 32x32x3x140 x * y * time
%           value - a 1x140 array with the value for lifting the spacebar
%             at any particular time (the trial ends immediately after the
%             spacebar is lifted no matter what)

%% Setup
global SD

if isempty(SD)
    initSD();
else
    SD.correct(end+1) = correct;
end

%% Build a trial
frames = zeros(32,32,3,140);
rr = 14:18;

value = -ones(1,140);

intervals = [81:90;111:120];
interval = 1+(rand<.5); % 0 = first interval, 1 = second interval

value(intervals(interval,:))=1;
    
sintervals = [31:40;61:70];

if rand<.5
    % same trial
    if rand<.5
        % both red
        s1_color = SD.colors.red;
        s2_color = SD.colors.red;
    else
        s1_color = SD.colors.yellow;
        s2_color = SD.colors.yellow;
    end
    r1_color = SD.colors.white;
    r2_color = SD.colors.green;
else
    % different trial
    if rand<.5
        % both red
        s1_color = SD.colors.red;
        s2_color = SD.colors.yellow;
    else
        s1_color = SD.colors.yellow;
        s2_color = SD.colors.red;
    end
    r1_color = SD.colors.green;
    r2_color = SD.colors.white;
end

flip = [2 1];
frames(rr,rr,:,sintervals(1,:)) = repmat(reshape(s1_color,1,1,3),length(rr),length(rr),1,length(sintervals(interval,:)));
frames(rr,rr,:,sintervals(2,:)) = repmat(reshape(s2_color,1,1,3),length(rr),length(rr),1,length(sintervals(interval,:)));
frames(rr,rr,:,intervals(interval,:)) = repmat(reshape(r1_color,1,1,3),length(rr),length(rr),1,length(intervals(interval,:)));
frames(rr,rr,:,intervals(flip(interval),:)) = repmat(reshape(r2_color,1,1,3),length(rr),length(rr),1,length(intervals(interval,:)));

SD.frames{end+1} = frames;

ret.frames = frames;
ret.value = value;

%% test
% h = figure;
% for i = 1:140
%     title(sprintf('Frame %i',i));
%     imagesc(squeeze(frames(:,:,:,i)));
%     pause(.05);
% end

function initSD()

global SD

SD.trial = 1;
SD.colors.red = [255 0 0];
SD.colors.yellow = [255 255 0];
SD.colors.grey = [127 127 127];
SD.colors.white = [255 255 255];
SD.colors.green = [0 255 0];
SD.frames = {};
SD.correct = [];
SD.initialized = true;
