% Extract saccades using Engbert & Kliegl (2003) code
%
% path 1: path in which .dat file is stored and output is saved
% file_name: basic filename (typically initials + date). 
% const : struct containing info needed for analysis
%
% output is a matrix containing all saccades, as many rows as there are saccades;
% (:,1) - saccade onset
% (:,2) - saccade offset
% (:,3) - saccade start x
% (:,4) - saccade start y
% (:,5) - saccade end x
% (:,6) - saccade end y
% (:,7) - peak velocity
%
% Function created by Donatas Jonikaitis, inspired by Martin Szinte
% V1 : 05.07.2011 Basic version
% V2 : 18.03.2014 Removing eye blink related data & replacing it with
% interpolated data; changing indexing from find to logical indexing to
% speed up calculations; removing trials with NaN values; adding column
% to the raw data containing removed data due to blinks;
% V3: 13.04.2014 Further debugging of the code (minor changes)
% V4: 11.06.2014 Small changes to blink detection algorithm
% v11: July 10, 2015 Changes how the files are loaded
% v12: March 28, 2016 Now removes ... message notation at the 5th column of the .dat file
% v13: August 26, 2016 Rewrote the file completely. Simplified desing. 
% v14: September 5, 2016 Removes blink detection into separate code.
% v15: February 23, 2017 Changed blink threshold (5 samples instead of 10) to match human psychophysics better


function preprocessing_eye_saccades_EK2003_v15 (path1, file_name, const)

% % For debugging purposes only, use only once
% path1 = path_out;

% Load the files
path_in = sprintf('%s%s.dat',path1, file_name); % Raw file containing eye traces
path_out = sprintf('%s%s_saccades_EK.mat',path1, file_name); % Output file containing detected saccades in this code


% Check whetHer data needs to be cleaned (removing ... notation from dat file)
try
    dat = load(path_in);
catch
    fileID = fopen(path_in);
    data = textscan(fileID, '%f %f %f %f %s');
    data{5} = [];
    data_cleaned = cell2mat(data);
    save(path_in,'data_cleaned', '-ascii', '-double')
    fclose(fileID);
    dat = data_cleaned;
    clear data_cleaned;
end

warning off MATLAB:divideByZero  % Warning could be encountered during calcullation

S = struct;

trial_start = const.trial_start;
trial_end = const.trial_end;

for i_trial = 1:(size(trial_start,1))
        
    % If pupil size is not recorded, add a default missing value
    if size(dat,2)==3
        dat2(:,4)=-1;
    end
    
    % Blink during trial. Mark it as [pupil size = 0]
    index1  = dat(:,1)>=trial_start(i_trial) & dat(:,1)<=trial_end(i_trial) & dat(:,2)==-1; % -1 is standard missing eye position marker
    if sum(index1)>0;
        dat(index1,4)=0; % Save blink data as pupil size 0 (default eye-link format)
    end
    
    % Missing time stamps trial. Mark as [x, y, pupil size = NaN]
    missing_data = 0; % In case data is missing
    index1 = dat(:,1)>=trial_start(i_trial) & dat(:,1)<=trial_end(i_trial);
    if sum(dat(index1,1))==0;
        missing_data = 1;
    end
    
    %============
    % Select data for saccade detection
    
    index1  = dat(:,1)>=trial_start(i_trial) & dat(:,1)<=trial_end(i_trial);
    
    % Time in given trial
    t_time=[];
    t_time = dat(index1,1);
    % Pupil size in given trial
    t_pupil=[];
    t_pupil = dat(index1,4);
    % X & Y coordinates
    t_pos=[];
    t_pos=[dat(index1,2),dat(index1,3)];
    
    % Save pre-blink detection data
    t_pos_old = t_pos;
    t_pupil_old = t_pupil;
    
    %=============
    
    % Get saccades in good trials or trials where blink detection went
    % allright
    matrix1=[];
    
    if missing_data==0 && ~isempty(t_time)
        
        % Remove blinks
        [t_pupil, t_pos] = blink_detect_v11(t_pupil_old, t_pos_old, 5);

        % Gety saccades, in case there is no data, just return empty ms
        % matrix
        try 
            v = preprocessing_eye_vecvel_v10(t_pos, const.SAMPRATE, const.VELTYPE);  % Get velocity of the eye
            ms = preprocessing_eye_microsaccmerge_v10(t_pos,v,const.velSD,const.minDur,const.mergeInt);  % Get saccades
            ms = preprocessing_eye_saccpar_v10(ms);
        catch
            % This catches trial if saccades can not be
            % calculated (such as blink through whole trial)
            ms = [];
        end
        
        % Single saccades saved into matrix
        i_sacc = size(ms,1);
        if i_sacc > 0 % If saccades detected
            for j=1:i_sacc
                matrix1(j,1)=t_time(ms(j,1)); % Start time
                matrix1(j,2)=t_time(ms(j,2)); % End time
                matrix1(j,3)=t_pos(ms(j,1),1); % Startx
                matrix1(j,4)=t_pos(ms(j,1),2); % Starty
                matrix1(j,5)=t_pos(ms(j,2),1); % Endx
                matrix1(j,6)=t_pos(ms(j,2),2); % Endy
                matrix1(j,7)=ms(j,4); % Peak velocity
            end
            S.saccades_EK{i_trial,1} = matrix1;
        else % If no saccades detected
            S.saccades_EK{i_trial,1} = matrix1;
        end
        
    elseif missing_data==0
        S.saccades_EK{i_trial,1} = matrix1;
    end
    
    % Raw data saved in this matrix
    S.eye_raw{i_trial, 1} = [t_time, t_pos, t_pupil]; % Raw data

    % Save pre-blink removal data
    S.eye_preblink{i_trial,1} = [t_time, t_pos_old, t_pupil_old];
    
end

saccades = S;
save (eval('path_out'), 'saccades')


y=saccades; % Output of the analysis


