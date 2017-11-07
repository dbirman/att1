% Analysis code
% V1.0 August 29, 2016. Initial version.
% V1.1 November 1, 2016. Made script more modular.

clear all;
clc;
close all;

% Which subject to run?
settings.subjects = 'all'; % 'all' to run all subjects
settings.exp_name = 'att1';
settings.exp_setup = 'dj office';

% Which sessions to run?
% 'all', 'last', 'before', 'after', 'interval', 'selected'
settings.data_sessions = 'all'; 

eval(sprintf('%s_analysis_settings', settings.exp_name)); % Load general settings


%% Preprocessing: import data into usable format

% Import raw data files of psychtoolbox & eyelink
% This step should be default for most experiments

% Connect to server and import data from it
settings.data_import_from_server = 0;
if settings.data_import_from_server == 1
    % Optional variables
%     settings.import_folders_include{1} = 'data_eyelink_edf';
%     settings.import_folders_include{2} = 'data_psychtoolbox';
%     settings.import_folders_exclude{1} = 'data_plexon_raw';
%     settings.import_folders_exclude{2} = 'data_plexon_mat';
    % Run code
    preprocessing_data_import_from_server_v21(settings);
end

% Modify raw settings for compatibility between experiments
settings.overwrite_raw_settings = 1;
settings.overwrite = 0;
if settings.overwrite_raw_settings == 1
    preprocessing_overwrite_raw_settings_v10(settings);
end

% Import .mat and .edf files into one folder
settings.preprocessing_import_files = 1;
settings.overwrite = 0; % If 1, runs analysis again even if it was done 
if settings.preprocessing_import_files == 1
    preprocessing_import_files_v13(settings);
end

% Modify eyelink messages for compatibility between experiments
settings.overwrite_eyelink_settings = 1;
settings.overwrite=0;
if settings.overwrite_eyelink_settings == 1
    preprocessing_overwrite_eyelink_settings_v10(settings);
end

% Combine settings and saccades files into one file; 
% reset saccades to degrees of visual angle; do drift correction
settings.preprocessing_eyelink_conversion = 1;
settings.overwrite = 0; % If 1, runs analysis again even if it was done 
if settings.preprocessing_eyelink_conversion == 1
    preprocessing_eyelink_conversion_v13(settings);
end

% Remove intermediate pre-processing folders
settings.preprocessing_remove_folders = 1;
if settings.preprocessing_remove_folders == 1
    preprocessing_remove_folders_v10(settings, 'path_data_temp_1_subject');
    preprocessing_remove_folders_v10(settings, 'path_data_temp_2_subject');
end



%% Preprocessing: detect and plot saccades

% Detect saccades
settings.overwrite = 0;
settings.preprocessing_saccade_detection = 0;
if settings.preprocessing_saccade_detection == 1
    att1_preprocessing_saccade_detection;
end

% Plot eye traces for manual inspection
settings.plot_saccades_raw = 0;
settings.overwrite = 0;
if settings.plot_saccades_raw == 1
    att1_preprocessing_plot_saccades_raw;
end


%% Behavioural data analysis

% % Plot day to day trials accepted/rejected
% settings.analysis_plot_training_performance = 1;
% settings.overwrite = 1;
% if settings.analysis_plot_training_performance==1
%     att1_analysis_plot_training_performance;
% end

% Plot day to day trials accepted/rejected
settings.analysis_errors_timecourse = 1;
settings.overwrite = 1;
if settings.analysis_errors_timecourse==1
    att1_analysis_plot_last_day_performance;
end

