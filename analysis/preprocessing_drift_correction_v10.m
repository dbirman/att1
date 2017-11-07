% V1.0 October 26, 2017. Initial version

function [var1, var2] = preprocessing_drift_correction_v10(settings, var1, var2)

fprintf('Doing drift correction\n')

% Eye position for drift correction
sacc1 = var1.eye_data.saccades_EK;
saccraw1 = var2.eye_processed;

% Time for drift calculations
time1 = var1.eyelink_events.(settings.drift_correction_time); % Time relative to which drift correction is done
t_start = time1 + settings.drift_correction_tstart; % Relative to time1 start checking for drift;
t_end = time1 + settings.drift_correction_tend; % Relative to time1 end checking for drift;
saccade_amp_threshold = settings.drift_correction_sacc_amp; % How large saccades are allowed

% Outputs
dist_mat = NaN(numel(sacc1),1);
coord_mat = NaN(numel(sacc1),2);
sacc_amp_index = NaN(numel(sacc1),1);
drift_output = cell(numel(sacc1),1);


%% Window size for the drift


a = var1.stim.(settings.drift_correction_window_max);
if iscell (a)
    drift_threshold = cell2mat(a);
elseif size(a,2)>1
    drift_threshold = a(:,4);
elseif size(a,2)==1
    drift_threshold = a;
end


%% Average gaze position in the trial

for tid = 1:length(saccraw1)
    
    sx1 = saccraw1{tid};
    
    if ~isnan(t_start(tid)) && ~isnan(t_start(tid)) && length(sx1)>1
        
        t1 = t_start(tid);
        t2 = t_end(tid);
        
        % Select data samples within given time
        index1=sx1(:,1)>=t1 & sx1(:,1)<=t2;
        x1=sx1(index1,2);
        y1=sx1(index1,3);
        eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate distance from the center
        
        % Distance & eye position
        dist_mat(tid)=nanmean(eyecoord1);
        coord_mat(tid,1)=nanmean(x1);
        coord_mat(tid,2)=nanmean(y1);
        
    end
end

coord_mat_copy = coord_mat;


%% Determine number of saccades


for tid = 1:length(sacc1)
    
    sx2 = sacc1{tid};
    
    if ~isnan(t_start(tid)) && ~isnan(t_start(tid)) && length(sx2)>1
        
        t1 = t_start(tid);
        t2 = t_end(tid);
        
        % Saccade amplitude
        x=sx2(:,5)-sx2(:,3);
        y=sx2(:,6)-sx2(:,4);
        sacc_amp = sqrt((x.^2)+(y.^2));
        starttimes=sx2(:,1);
        
        % sacc_amp_index is large saccades
        ind = sacc_amp>=saccade_amp_threshold & starttimes>=t_start(tid) & starttimes<=t_end(tid);
        sacc_amp_index(tid) = sum(ind);
        
    end
end


%%  Find average eye position over number of trials

% Remove trials that deviate from fixation too much
coord_mat(dist_mat>=drift_threshold,:) = NaN;

% Remove trials with saccades from means calculation
coord_mat(sacc_amp_index>0) = NaN;

% Initialize avg_mat in case it's needed
avg_mat = NaN(size(coord_mat));

% Calculate mean/median distance from fixation
if strcmp(settings.drift_correction_method, 'median')
    step1 = settings.drift_correction_trials;
    try
        % movmedian function
        for i=1:size(coord_mat,2)
            avg_mat(:,i) = movmedian(coord_mat(:,i), step1, 'omitnan'); % How much to reset
        end
    catch
        % Manual code (older matlab)
        step1 = settings.drift_correction_trials;
        for i=1:size(coord_mat,2)
            for tid=1:size(coord_mat,1)
                ind = round(tid-(step1-1)/2):1:round(tid+(step1-1)/2);
                ind(ind<1)=[];
                ind(ind>size(coord_mat,1))=[];
                temp1 = coord_mat(ind,i);
                temp1 = temp1(~isnan(temp1));
                if ~isempty(temp1)
                    avg_mat(tid,i) = median(temp1);
                end
            end
        end
    end
elseif strcmp(settings.drift_correction_method, 'mean')
    step1 = settings.drift_correction_trials;
    try
        % movmean functions
        for i=1:size(coord_mat,2)
            avg_mat(:,i) = movmean(coord_mat(:,i), step1, 'omitnan'); % How much to reset
        end
    catch
        % Manual code (older matlab)
        for i=1:size(coord_mat,2)
            for tid=1:size(coord_mat,1)
                ind = round(tid-(step1-1)/2):1:round(tid+(step1-1)/2);
                ind(ind<1)=[];
                ind(ind>size(coord_mat,1))=[];
                temp1 = coord_mat(ind,i);
                temp1 = temp1(~isnan(temp1));
                if ~isempty(temp1)
                    avg_mat(tid,i) = mean(temp1);
                end
            end
        end
    end
elseif strcmp(settings.drift_correction_method, 'each trial')
    avg_mat = coord_mat;
else
    avg_mat = coord_mat;
end


%% Do drift correction

for tid = 1:length(sacc1)
    
    if ~isnan(avg_mat(tid,1))
                
        % Change saccraw data
        saccraw1{tid}(:,2) = saccraw1{tid}(:,2) - avg_mat(tid,1);
        saccraw1{tid}(:,3) = saccraw1{tid}(:,3) - avg_mat(tid,2);
        
        % Correct individual saccades
        if length(sacc1{tid})>1
            sacc1{tid}(:,3) = sacc1{tid}(:,3) - avg_mat(tid,1);
            sacc1{tid}(:,5) = sacc1{tid}(:,5) - avg_mat(tid,1);
            sacc1{tid}(:,4) = sacc1{tid}(:,4) - avg_mat(tid,2);
            sacc1{tid}(:,6) = sacc1{tid}(:,6) - avg_mat(tid,2);
        end
        
        drift_output{tid} = 'drift on';
    else
        drift_output{tid} = 'drift off';
    end
end


%%  Output

var1.eye_data.drift_output = drift_output;
var1.eye_data.predrift_xy = coord_mat_copy;
var1.eye_data.predrift_xy_average = avg_mat;

var1.eye_data.saccades_EK = sacc1; % Over-write the field
var2.eye_processed = saccraw1; % Over-wrtie the field

