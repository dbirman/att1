% Remove some intermediate pre-processing folders if needed
%
% V1.0 October 30, 2017


function preprocessing_remove_folders_v10(settings, varargin)


% Show file you are running
p1 = mfilename;
fprintf('\n=========\n')
fprintf('\n Current file:  %s\n', p1)
fprintf('\n=========\n')

if length(varargin)>=1
    varx1 = varargin{1};
else
    varx1 = [];
end

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
    settings = get_settings_path_and_dates_ini_v11(settings, varx1);
    dates_used = settings.data_sessions_to_analyze;
    
    % Analysis for each separate day
    for i_date = 1:length(dates_used)
        
        % Current folder to be analysed (raw date, with session index)
        date_current = dates_used(i_date);
        settings.date_current = date_current; % Variable needed for data import
        ind_dates = find (date_current==settings.index_dates);
        
        for i_folder = 1:numel(ind_dates)
            
            folder_name = settings.index_directory{ind_dates(i_folder)};
            path1 = sprintf('%s%s/', settings.(varx1), folder_name);
            
            % Remove given directory
            if isdir(path1)
                fprintf('Will remove following directory %s\n', path1);
                rmdir(path1, 's')
            end
            
        end
        % End of each folder
    end
    % End of each day
end
% End of analysis for each subject

