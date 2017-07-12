% ONLINE data figure
% Recalculates from pixels back to degrees - however it's easy to plot
% everythng in raw data (pixels).


%================
% Plot fixation BEFORE drift correction
%================

% Radius & window
x1=fix_xc-dispcenter(1); y1=fix_yc-dispcenter(2);
x1=x1/deg2pix; y1=y1/deg2pix; % Convert to degrees
eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
radius1 = eyecoord1;
if expsetup.stim.fixation_drift_correction_on==1
    saccwindow = expsetup.stim.fixation_accuracy_drift;
else
    saccwindow = expsetup.stim.expmatrix(tid,em_eye_window);
end

% Plot fixation before it was acquired
starttime1 = expsetup.stim.expmatrix(tid,em_data_fixation_on) - expsetup.stim.expmatrix(tid,em_data_first_display);
starttime1 = starttime1*1000;
if ~isnan(expsetup.stim.expmatrix(tid,em_data_fixation_acquired))
    trialdur = expsetup.stim.expmatrix(tid,em_data_fixation_acquired) - expsetup.stim.expmatrix(tid,em_data_fixation_on);
    trialdur = trialdur*1000;
else
    trialdur = expsetup.stim.expmatrix(tid,em_data_fixation_off)-expsetup.stim.expmatrix(tid,em_data_fixation_on); % Convert to seconds
    trialdur = trialdur*1000;
end
h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
    'EdgeColor', [0.85, 0.85, 0.85],  'FaceColor', [0.85, 0.85, 0.85], 'Curvature', 0, 'LineWidth', 1);

% Plot fixation after it was acquired, during drift correction
if ~isnan(expsetup.stim.expmatrix(tid,em_data_drift_maintained))
    starttime1 = expsetup.stim.expmatrix(tid,em_data_fixation_acquired) - expsetup.stim.expmatrix(tid,em_data_first_display);
    starttime1 = starttime1*1000;
    trialdur = expsetup.stim.expmatrix(tid,em_data_drift_maintained) - expsetup.stim.expmatrix(tid,em_data_fixation_acquired);
    trialdur = trialdur*1000;
    h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
        'EdgeColor', [0.65, 0.65, 0.65],  'FaceColor', [0.65, 0.65, 0.65], 'Curvature', 0, 'LineWidth', 1);
end

%================
% Plot fixation AFTER drift correction
%===============

% Radius & window
x1=fix_xc; y1=fix_yc;
x1=x1-dispcenter(1); y1=y1-dispcenter(2);
x1=x1/deg2pix; y1=y1/deg2pix;
eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
radius1 = eyecoord1;
saccwindow = expsetup.stim.expmatrix(tid,em_fixation_window);

% Plot fixation
if ~isnan(expsetup.stim.expmatrix(tid,em_data_drift_maintained)) && ~isnan(expsetup.stim.expmatrix(tid,em_data_fixation_off))
    starttime1 = expsetup.stim.expmatrix(tid,em_data_drift_maintained) - expsetup.stim.expmatrix(tid,em_data_first_display);
    starttime1 = starttime1*1000;
    trialdur = expsetup.stim.expmatrix(tid,em_data_fixation_off) - expsetup.stim.expmatrix(tid,em_data_drift_maintained);
    trialdur = trialdur*1000;
    h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
        'EdgeColor', [1, 0.8, 0.8],  'FaceColor', [1, 0.8, 0.8], 'Curvature', 0, 'LineWidth', 1);
end

%==========================
% Plot saccade target onset
%=========================

if ~isnan(expsetup.stim.expmatrix(tid,em_data_response_on))
    
    % Radius & window
    if expsetup.stim.expmatrix(tid,em_blockcond)==1 && expsetup.stim.expmatrix(tid,em_target_number)==2
        y1 = t1_yc; x1 = t1_xc;
    elseif expsetup.stim.expmatrix(tid,em_blockcond)==2 && expsetup.stim.expmatrix(tid,em_target_number)==2
        y1 = t2_yc; x1 = t2_xc;
    elseif expsetup.stim.expmatrix(tid,em_blockcond)==3 || expsetup.stim.expmatrix(tid,em_target_number)==1
        y1 = t3_yc; x1 = t3_xc;
    end
    x1=x1-dispcenter(1); y1=y1-dispcenter(2);
    x1=x1/deg2pix; y1=y1/deg2pix;
    eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
    radius1 = eyecoord1;
    saccwindow = expsetup.stim.expmatrix(tid,em_eye_window);
    
    % Plot saccade target
    starttime1 = expsetup.stim.expmatrix(tid,em_data_response_on) - expsetup.stim.expmatrix(tid,em_data_first_display);
    starttime1 = starttime1*1000;
    trialdur = expsetup.stim.expmatrix(tid,em_data_response_off) - expsetup.stim.expmatrix(tid,em_data_response_on);
    trialdur = trialdur*1000;
    h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
        'EdgeColor', [1, 0.5, 0.5], 'FaceColor', [1, 0.5, 0.5], 'Curvature', 0, 'LineWidth', 1);
    text(starttime1, -3, 'ST1', 'Color', [1, 0.5, 0.5],  'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
end


%=================
% Plot distractor
%=================

if ~isnan(expsetup.stim.expmatrix(tid,em_data_response2_on)) &&  expsetup.stim.expmatrix(tid,em_target_number)==2
    
    % Radius & window
    if expsetup.stim.expmatrix(tid,em_blockcond)==1
        y1 = t2_yc; x1 = t2_xc;
    elseif expsetup.stim.expmatrix(tid,em_blockcond)==2
        y1 = t1_yc; x1 = t1_xc;
    end
    x1=x1-dispcenter(1); y1=y1-dispcenter(2);
    x1=x1/deg2pix; y1=y1/deg2pix;
    eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
    radius1 = eyecoord1;
    saccwindow = expsetup.stim.expmatrix(tid,em_eye_window);
    
    % Plot distractor
    starttime1 = (expsetup.stim.expmatrix(tid,em_data_response2_on) - expsetup.stim.expmatrix(tid,em_data_first_display));
    starttime1 = starttime1*1000;
    if ~isnan(expsetup.stim.expmatrix(tid,em_data_response2_off))
        trialdur = (expsetup.stim.expmatrix(tid,em_data_response2_off) - expsetup.stim.expmatrix(tid,em_data_response2_on));
        trialdur = trialdur*1000;
    elseif isnan(expsetup.stim.expmatrix(tid,em_data_response2_off))
        trialdur = expsetup.stim.response_duration;
        trialdur = trialdur*1000;
    end
    h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
        'EdgeColor', [0.7, 0.1, 0.1], 'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
    text(starttime1, -5, 'ST2', 'Color', [0.7, 0.1, 0.1],  'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
end

%==========================
% Plot memory object
%=========================

% Radius & window
eyecoord1=t1_yc;
radius1 = eyecoord1;
saccwindow = expsetup.stim.expmatrix(tid,em_eye_window);

% Plot memory target
if  ~isnan(expsetup.stim.expmatrix(tid,em_data_memory_on))
    starttime1 = (expsetup.stim.expmatrix(tid,em_data_memory_on) - expsetup.stim.expmatrix(tid,em_data_first_display));
    starttime1 = starttime1*1000;
    if ~isnan(expsetup.stim.expmatrix(tid,em_data_memory_off))
        trialdur = (expsetup.stim.expmatrix(tid,em_data_memory_off) - expsetup.stim.expmatrix(tid,em_data_memory_on));
        trialdur = trialdur*1000;
    elseif isnan(expsetup.stim.expmatrix(tid,em_data_memory_off))
        trialdur = expsetup.stim.expmatrix(tid,em_memory_duration);
        trialdur = trialdur*1000;
    end
    h=rectangle('Position', [starttime1, radius1-saccwindow, trialdur, saccwindow*2],...
        'EdgeColor', [0.4, 0.4, 1],  'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
    text(starttime1, -1, 'M', 'Color', [0.4, 0.4, 1],  'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
end

%====================
% Plot reward timing
if ~isnan(expsetup.stim.expmatrix(tid,em_data_reward_on))
    yc = 0;
    saccwindow = 1;
    starttime1 = (expsetup.stim.expmatrix(tid,em_data_reward_on) - expsetup.stim.expmatrix(tid,em_data_first_display));
    starttime1 = starttime1*1000;
    trialdur = expsetup.stim.expmatrix(tid,em_reward_size_ms);
    h=rectangle('Position', [starttime1, yc-saccwindow, trialdur, saccwindow],...
        'EdgeColor', [0.2, 0.2, 0.5], 'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
    text(starttime1, 2, 'Reward', 'Color', [0.2, 0.2, 0.5],  'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
end

%==============
% Plot eye position data
%==============

% Eye position
sx2 = trialmat(:, 1:3); % Data to be used
sx2(:,1) = sx2(:,1) - sx2(1,1); % Reset time to 0
sx2(:,1) = sx2(:,1) * 1000; % Convert to miliseconds
ind=sx2(:,2)==0; % Replace zeros (no recording)
sx2(ind,2)=NaN; sx2(ind,3)=NaN; % Replace zeros
% Convert to degrees
x1=sx2(:,2)-dispcenter(1); y1=sx2(:,3)-dispcenter(2);
x1=x1./deg2pix; y1=y1./deg2pix;

% Plot eye position
if size(sx2,1)>1
    eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
    h=plot(sx2(:,1), eyecoord1, 'k', 'LineWidth', 1); % Plot eye position in space and time
end

%=================
% Fig settings

set (gca,'FontSize', fontsz);

% Set the legend
if expsetup.stim.expmatrix(tid, em_data_reject)==1
    text(0, 10, 'Correct', 'Color', color1(1,:), 'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
elseif  expsetup.stim.expmatrix(tid, em_data_reject)==99
    text(0, 10, 'Unknown', 'Color', color1(8,:), 'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
else
    try
        text(0, 10, error_message, 'Color', color1(5,:), 'FontSize', fontszlabel, 'HorizontalAlignment', 'left')
    end
end

% Y axis
x1=t1_xc-dispcenter(1); y1=t1_yc-dispcenter(2);
x1=x1./deg2pix; y1=y1./deg2pix;
eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
eyecoord1=round(eyecoord1);
if eyecoord1~=0 % For peripheral targets
    set(gca,'YTick',[0, eyecoord1]);
else % For central targets
    % Nothing at the moment
end
set(gca,'YLim',[-6.5, 12]);
ylabel ('Gaze position', 'FontSize', fontszlabel)

% X axis
if ~isnan(expsetup.stim.expmatrix(tid,em_data_memory_on)) && ~isnan(expsetup.stim.expmatrix(tid,em_data_response_on))
    set(gca,'XTick', [0, ...
        round((expsetup.stim.expmatrix(tid,em_data_memory_on) - expsetup.stim.expmatrix(tid,em_data_first_display))*1000),...
        round((expsetup.stim.expmatrix(tid,em_data_response_on) - expsetup.stim.expmatrix(tid,em_data_first_display))*1000),...
        round((expsetup.stim.expmatrix(tid,em_data_last_display) - expsetup.stim.expmatrix(tid,em_data_first_display))*1000)])
else
    set(gca,'XTick', [0, round((expsetup.stim.expmatrix(tid,em_data_last_display) - expsetup.stim.expmatrix(tid,em_data_first_display))*1000)])
end
a1 = (expsetup.stim.expmatrix(tid,em_data_last_display) - expsetup.stim.expmatrix(tid,em_data_first_display))*1000;
set(gca,'XLim',[-250,a1+250]);
xlabel ('Time', 'FontSize', fontszlabel)

