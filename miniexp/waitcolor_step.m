function ret = waitcolor_step( correct )
%WAITCOLOR_STEP Step a wait forc olor simulation code forward
%   Returns a 32x32x3x60 trial representing a view of the screen as a same
%   different psychophysics task runs. The task runs in the following way:
%       20 frames: blank
%       X frames: color1
%       10 frames: color2
%       20 frames: blank
%
%   Color 1 and color 2 are two different colors. The agent is supposed to
%   learn to press when color1 switches to color 2. X takes a random value
%   from the options 10/20/30/40/50
%   
%   INPUT:
%       Correct - whether the agent got the last trial correct (use for
%                 tracking but not actually relevant)
%   OUTPUT:
%       RET - A cell array with:
%           frames - a 32x32x3x140 x * y * time
%           value - a 1x60 array with the value for lifting the spacebar
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
        save(fullfile(SD.datafolder,'waitcolor',sprintf('data%i.mat',SD.trial)),'data');
        disp(sprintf('Clearing frames'));
        SD.trials = [];
        SD.correct = [];
        SD.frames = {};
    end
end

%% Build a trial
numWait = randsample(10:10:50,1);

frames = zeros(32,32,3,50+numWait,'uint8');

value = -ones(1,50+numWait,'int8');

color1 = randi(5);
color2 = randi(5);
colors = fields(SD.colors);

intervals = {21:(20+numWait),(20+numWait+1):(20+numWait+10)};

frames(:,:,:,intervals{1}) = repmat(reshape(SD.colors.(colors{color1}),1,1,3),32,32,1,length(intervals{1}));
frames(:,:,:,intervals{2}) = repmat(reshape(SD.colors.(colors{color2}),1,1,3),32,32,1,length(intervals{2}));
value(intervals{2})=1;

SD.frames{end+1} = frames;

ret.frames = frames;
ret.value = value;

%% test
% h = figure;
% for i = 1:size(frames,4)
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
