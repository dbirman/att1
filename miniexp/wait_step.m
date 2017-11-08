function ret = wait_step( correct )
%WAIT_STEP Step a same different simulation code forward
%   Returns a 32x32x3x140 trial representing a view of the screen as a same
%   different psychophysics task runs. The task runs in the following way:
%       40 frames: blank
%       10 frames: response1
%
%   The agent is supposed to learn to lift the spacebar during response1
%   
%   INPUT:
%       Correct - whether the agent got the last trial correct (use for
%                 tracking but not actually relevant)
%   OUTPUT:
%       RET - A cell array with:
%           frames - a 32x32x3x140 x * y * time
%           value - a 1x50 array with the value for lifting the spacebar
%             at any particular time (the trial ends immediately after the
%             spacebar is lifted no matter what)

%% Setup
global SD

if isempty(SD)
    initSD();
else
    SD.trials(end+1) = SD.trial;
    SD.correct(end+1) = correct;
    SD.trial = SD.trial + 1;
    %% save
    if mod(SD.trial,100)==0
        disp(sprintf('Saving data'));
        data = SD;
        save(fullfile(SD.datafolder,'waitstep',sprintf('data%i.mat',SD.trial)),'data');
        disp(sprintf('Clearing frames'));
        SD.trials = [];
        SD.correct = [];
        SD.frames = {};
    end
end

%% Build a trial
frames = zeros(32,32,3,50,'uint8');

value = -ones(1,50,'int8');

value(41:50) = 1;

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
SD.trials = [];
SD.correct = [];
SD.initialized = true;
SD.datafolder = '~/data/att1/dqn';
