% Change raw data files to be compliant with current analysis.
% Changes are renaming columns or fixing bugs in the code that are well
% established.
% No data changes in this code.
%
% V1.0 October 30, 2017

function  preprocessing_overwrite_raw_settings_v10(settings)

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
    try
        settings = get_settings_path_and_dates_ini_v11(settings, 'path_data_psychtoolbox_subject');
        dates_used = settings.data_sessions_to_analyze;
    catch
        dates_used = [];
    end
    
    %===========
    % Find an index of dates to be modified
    settings.overwrite_temp_switch = 0;
    a1 = sprintf('%s_overwrite_raw_settings', settings.exp_name); % Path to file containing trial settings
    eval(a1)
    
    % Analysis for each separate day
    for i_date = 1:length(dates_used)
        
        % Current folder to be analysed (raw date, with session index)
        date_current = dates_used(i_date);
        settings.date_current = date_current; % Variable needed for data import
        ind_folders = find (date_current==settings.index_dates);
        
        % By default, it will not over-write the data
        settings.overwrite_temp_switch = 0;
        if exist('overwrite_temp_index', 'var')
            for i = 1:numel (overwrite_temp_index)
                a = overwrite_temp_index{i};
                b(i) = sum(date_current == a);
            end
            if sum(b)>0
                settings.overwrite_temp_switch = 1;
            else
                fprintf('\nCurrent date is not in the list to be modified as the raw data file\n')
            end
        else
            fprintf('\nScript for over-writing raw data has no dates specified, no files will be modified\n')
        end
        
        if settings.overwrite_temp_switch == 1;
            for i_folder = 1:numel(ind_folders)
                
                folder_name = settings.index_directory{ind_folders(i_folder)};
                path1 = [settings.path_data_psychtoolbox_subject, folder_name, '/' folder_name, '_data_structure.mat'];
                path1_copy = [settings.path_data_psychtoolbox_subject, folder_name, '/' folder_name, '_data_structure_original.mat'];
                
                %================
                % Analysis
                %================
                
                if ((exist(path1, 'file') && ~exist(path1_copy, 'file')) || settings.overwrite==1) && exist(path1, 'file')
                    
                    fprintf('\nChanges to experiment settings file with modifications and bug fixes %s\n', folder_name)
                    
                    %==========
                    % Save original file before doing changes
                    % Original data is saved only once
                    if ~exist(path1_copy, 'file')
                        copyfile(path1, path1_copy)
                        fprintf('\nWill save original settings file as a backup %s_data_structure_original\n', folder_name)
                    else
                        fprintf('\nCopy of the settings file %s_data_structure_original exists, no changes to that file\n', folder_name)
                    end
                    
                    % Now open the original data file (always)
                    var1 = get_struct_v10(path1_copy);
                    
                    %===========
                    % Main preprocessing function
                    a1 = sprintf('%s_overwrite_raw_settings', settings.exp_name); % Path to file containing trial settings
                    eval(a1)
                    
                    %===========
                    % Save data
                    expsetup = var1;
                    save (path1, 'expsetup')
                    
                elseif ~exist(path1, 'file')
                    fprintf('\nData file %s_data_structure does not exist, skipping analysis\n', folder_name)
                elseif exist(path1, 'file') && settings.overwrite==0
                    fprintf('\nCopy of the settings file %s_data_structure_original already exists, skipping analysis\n', folder_name)
                else
                    fprintf('\nUnknown file localisation error \n')
                end
                % End of analysis
                
            end
            % End of each folder
        end
        % End of settings.overwrite_temp_switch
    end
    % End of each day
end
% End of analysis for each subject
