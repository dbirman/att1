% Combine files, extract saccades
% V1.0 - October 10, 2016. Neurophysiology setup.
% V1.1 - November 4, 2016. Neurophys setup. Small bug fixes.
% V2.0 - February 2, 2017. Psychophysics setup. Creates one file per session
% V1.2 - September 14, 2017. Neurophys setup. Re-written experiment design.
% Converted into function.
% V1.3 - October 25, 2017. Minor adjustments to code. Uses psychtoolbox
% folder to detect files instead of eye-tracking
%
% Output - saved structures in 'temp2' folder:
% xx_settings
% xx_eye_traces
%
% Donatas Jonikaitis

function  preprocessing_import_files_v13(settings)

% Show file you are running
p1 = mfilename;
fprintf('\n=========\n')
fprintf('\n Current file:  %s\n', p1)
fprintf('\n=========\n')

% Loading the files needed
if ~exist('settings', 'var')
    settings = struct;
end
settings = get_settings_ini_v10(settings);

% Run pre-processing for each subject
for i_subj = 1:length(settings.subjects)
    
    % Select curent subject
    settings.subject_current = settings.subjects{i_subj};
    
    % Get subject folder paths and dates to analyze
    settings = get_settings_path_and_dates_ini_v11(settings, 'path_data_psychtoolbox_subject');
    dates_used = settings.data_sessions_to_analyze;
    
    % Analysis for each separate day
    for i_date = 1:length(dates_used)
        
        % Current folder to be analysed (raw date, with session index)
        date_current = dates_used(i_date);
        settings.date_current = date_current; % Variable needed for data import
        ind = date_current==settings.index_dates;
        folder_name = [settings.subject_current, num2str(settings.date_current)];
        
        path1 = [settings.path_data_temp_2_subject, folder_name, '/' folder_name, '_settings.mat'];
        
        if ~exist(path1, 'file') || settings.overwrite==1
            fprintf('\nStarting file conversion and combining for the folder name %s\n', folder_name)
            %===========
            % Main preprocessing function
            preprocessing_psych_eye_combine_v24 (settings, 'stim');
            %===========
        else
            fprintf('\nFolder name %s already exists, skipping file import\n', folder_name)
            % Do nothing
        end
        
    end
     
end
% End of analysis for each subject










