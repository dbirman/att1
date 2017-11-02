% Initialize frames matrix which will contain every frame info for the trial

%% Blank variables

t1 = expsetup.stim.esetup_lever_acquire_duration(tid,1);
dur1 = t1 + 10; % How long the trial is
time_unit = expsetup.screen.ifi;
b1 = ceil(dur1/time_unit) + 1; % How many frames to initalize (always add extra frame at the end)

% Create all fields filled with NaNs
f1 = fieldnames(expsetup.stim);
ind = strncmp(f1,'eframes', 7);
for i=1:numel(ind)
    if ind(i)==1
        if iscell(expsetup.stim.(f1{i}))
            [~,n,o]=size(expsetup.stim.(f1{i}){1});
            frames_mat = NaN(b1, n, o); 
            expsetup.stim.(f1{i}){tid} = frames_mat;
        end
    end
end


%% Background texture

temp1 = frames_mat(:,1);
temp1(:, 1) = 1;
% Save data
expsetup.stim.eframes_texture_on{tid}=temp1;


%% Blinking fixation

b1 = 1/expsetup.stim.fixation_blink_frequency/2; % How many ms blink lasts; /2 because: 1 blink = 1 stim on + 1 blank
b1=round(b1/time_unit); % How many frames blink lasts
m1 = [ones(b1,1); zeros(b1,1)]; % Stim on + blank
% Fill in trialmat with blinks
temp1 = frames_mat;
ans1 = floor(size(temp1,1)/length(m1)); % How many reps per trial
m1 = repmat(m1,ans1,1); % How many reps per trial
temp1(1:length(m1),1) = m1;
if length(m1)<size(temp1,1)
    ind = length(m1)+1:size(temp1,1);
    temp1(ind,1) = m1(1:numel(ind));
end
% Save data
expsetup.stim.eframes_fixation{tid}=temp1;


%% Attention cue

m1 = ceil((expsetup.stim.att_cue_duration)/time_unit);
% Att cue on
temp1 = frames_mat;
temp1(1:m1,1)=1; 
expsetup.stim.eframes_att_cue_on_temp{tid}=temp1;
% Att cue off
temp1 = frames_mat;
temp1(m1+1,1)=1;
expsetup.stim.eframes_att_cue_off_temp{tid}=temp1;


%% Gabor patch

% Probe + blank duration
a = expsetup.stim.esetup_probe_duration(tid, 1);
p_dur = ceil(a/time_unit); % Gabor
a = expsetup.stim.esetup_probe_isi(tid, 1);
p_isi = ceil(a/time_unit); % Blank

m1 = [ones(p_dur,1); zeros(p_isi,1)];
for i = 1:expsetup.stim.att_stim_reps
    a = m1*i; % Save att stimulus number
    if i == 1
        b = a;
    else
        b = [b;a];
    end
end
m1 = b; % gabors + blanks

% Fill in trialmat with probe + blank
if expsetup.stim.att_stim_reps>0
    if numel(m1)<=size(frames_mat,1)
        expsetup.stim.eframes_att_stim_on_temp{tid}(1:length(m1),1)=m1;
    else
        expsetup.stim.eframes_att_stim_on_temp{tid}(1:end,1)=m1(1:size(frames_mat,1));
    end
end

% Fill in trialmat with probe onset time
a = find(diff(m1) == expsetup.stim.att_stim_reps_probe);
a = a+1;
expsetup.stim.eframes_probe_on_temp{tid}(a,1) = 1;

% Fill in trialmat with probe offset time
a = find(diff(m1) == -expsetup.stim.att_stim_reps_probe);
a = a+1;
expsetup.stim.eframes_probe_off_temp{tid}(a,1) = 1;


%% Response ring size

b1 = expsetup.stim.esetup_response_ring_duration(tid,1); % How many ms ring lasts
b1 = round(b1/time_unit); % How many frames ring lasts
a1 = round(expsetup.stim.esetup_response_ring_size_start(tid,1)); % Ring size max
a2 = round(expsetup.stim.esetup_response_ring_size_end(tid,1)); % Ring size min

% Ring size for each frame
m1_stim = linspace(a1, a2, b1); 
if size(m1_stim,2)>1 % Rotate if necessary
    m1_stim = m1_stim';
end

% Blank duration for each frame
b1 = ceil((expsetup.stim.response_ring_isi)/time_unit);
m1_isi = NaN(b1,1);

%=================
% Save ring size into matrix
a = expsetup.stim.esetup_response_ring_sequence(tid,1:2);
ind = ~isnan(a);

if sum(ind)==0 % If no rings are shown
    m1 = NaN;

elseif sum(ind)>=1 % If only one ring is shown
    
    for i = 1:numel(ind)
        if i==1
            if ind(i)==1
                m1 = [m1_stim; m1_isi];
            else
                m1 = [NaN((numel(m1_stim)),1); NaN(numel(m1_isi),1)];
            end
        elseif i>1
            if ind(i)==1
                m1 = [m1; m1_stim; m1_isi];
            else
                m1 = [m1; NaN((numel(m1_stim)),1); NaN(numel(m1_isi),1)];
            end
        end
    end
end

% Save ring sizes into matrix
if numel(m1)<=size(frames_mat,1)
    expsetup.stim.eframes_response_ring_size_temp{tid}(1:length(m1),1)=m1;
else
    expsetup.stim.eframes_response_ring_size_temp{tid}(1:end,1)=m1(1:size(frames_mat,1));
end

%% Response ring durations

a = expsetup.stim.esetup_response_ring_sequence(tid,1:2);
ind = ~isnan(a);

if sum(ind)==0 % If no rings are shown
    m1 = NaN;
    m2 = NaN;

elseif sum(ind)>=1 % If 1 or more rings are shown
    for i = 1:numel(ind)
        if i==1
            if ind(i)==1
                m1 = [ones((numel(m1_stim)),1)*a(i); zeros(numel(m1_isi),1)];
            else
                m1 = [NaN((numel(m1_stim)),1); NaN(numel(m1_isi),1)];
            end
        elseif i>1
            if ind(i)==1
                m1 = [m1; ones((numel(m1_stim)),1)*a(i); zeros(numel(m1_isi),1)];
            else
                m1 = [m1; NaN((numel(m1_stim)),1); NaN(numel(m1_isi),1)];
            end
        end
    end
    m1(end+1)=NaN; % Add an extra element
    
    % Determine when last ring is shown
    m2 = [NaN(numel(m1),1)];
    a = find (m1>=0);
    m2(a(end)+1) = 1; % Rings off
    
end

% Save rings into frames matrix
if numel(m1)<=size(frames_mat,1)
    expsetup.stim.eframes_response_ring_on_temp{tid}(1:length(m1),1) = m1;
    expsetup.stim.eframes_response_ring_off_temp{tid}(1:length(m1),1) = m2;
else
    expsetup.stim.eframes_response_ring_on_temp{tid}(1:end,1) = m1(1:size(frames_mat,1));
    expsetup.stim.eframes_response_ring_off_temp{tid}(end,1) = 1; % Save last frame as response ring offset
end



