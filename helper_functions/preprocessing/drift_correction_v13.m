% Offline drift correction for eye position
%
% Latest revision history:
% v1.0 30 January 2015 Initial script written
% v1.1 13 February 2015 Added 4th column; now this code is as it's own
% script
% v1.2 July 29, 2015: adapted to new format to use S structure everyhwere
% v1.3 October 10, 2016: made it into a function
%
% Output:
% column 1          0 - no drift correct, 1 - drift correct
% columns 2 & 3     x & y of drift correct
% column 4          distance from fixation
%
% Donatas Jonikaitis


%% Settings

function [y, y_sacc, y_raw] = drift_correction_v13 (sacc1, saccraw1, acc1, time1)


% Inputs
sacc1; % Raw saccades matrix
saccraw1; % Raw eyetraces matrix
acc1; % Contains allowed window for accuracy
time1; % Relative to which time to track drift correction

% Some settings, quite basic
% Assumes that data is in degrees
amp_threshold = 1; % Max sacc amplitude allowed during period of drift correct
min_threshold = 0.5; % Radius out of which drift correct is considered
int_start = -90; % Relative to time1 start checking for drift;
int_end = -10; % Relative to time1 end checking for drift;

% For drift correction initialize empty matrix
drift_correct=zeros(length(sacc1),4); % Initialize matrix


%% Do drift correction

for tid=1:length(saccraw1)
    
    max_threshold = acc1(tid); % Radius within which drift correct is considered
    
    sx1 = saccraw1{tid};
    sx2 = sacc1{tid};
    if ~isnan(time1(tid))
        t1 = time1(tid) + int_start; % Change this value
        t2 = time1(tid) + int_end; % Change this value
        
        % Convert raw data into coordinates
        if length(sx1)>1
            index1=sx1(:,1)>=t1 & sx1(:,1)<=t2;
            x1=sx1(index1,2);
            y1=sx1(index1,3);
            eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
        end
        
        % Check whethere there were some large saccades during period of
        % interest
        if length(sx2)>1
            
            % Find saccade length
            xsacc=sx2(:,5)-sx2(:,3);
            ysacc=sx2(:,6)-sx2(:,4);
            sacclength=sqrt((xsacc.^2)+(ysacc.^2));
            
            % index1 is correct saccades
            starttimes=sx2(:,1);
            index1=sacclength>amp_threshold & starttimes>t1 & starttimes<t2;
            amp_index1=+index1;
        else
            amp_index1=0;
        end
        
        % Do drift correction
        if length(sx1)>1
            if nanmean(eyecoord1) >= min_threshold && nanmean(eyecoord1) <= max_threshold && sum(amp_index1)==0
                % Save factor of correction
                drift_correct(tid,1) = 1;
                drift_correct(tid,2) = nanmean(x1);
                drift_correct(tid,3) = nanmean(y1);
                drift_correct(tid,4) = nanmean(eyecoord1);
                % Change saccraw data
                saccraw1{tid}(:,2) = saccraw1{tid}(:,2) - nanmean(x1);
                saccraw1{tid}(:,3) = saccraw1{tid}(:,3) - nanmean(y1);
                % Reset all saccades
                if length(sx2)>1
                    sacc1{tid}(:,3) = sacc1{tid}(:,3) - nanmean(x1);
                    sacc1{tid}(:,5) = sacc1{tid}(:,5) - nanmean(x1);
                    sacc1{tid}(:,4) = sacc1{tid}(:,4) - nanmean(y1);
                    sacc1{tid}(:,6) = sacc1{tid}(:,6) - nanmean(y1);
                end
            else
                % Save factor of correction
                drift_correct(tid,1) = 0;
                drift_correct(tid,2) = nanmean(x1);
                drift_correct(tid,3) = nanmean(y1);
                drift_correct(tid,4) = nanmean(eyecoord1);
            end
        end
    end
end

y = drift_correct;
y_raw = saccraw1;
y_sacc = sacc1;

