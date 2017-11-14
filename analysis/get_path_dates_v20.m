% Reads multiple folders recorded on same/different days and produces
% matrix with all sessions and folders recorded
%
% Revision history:
%
% primate 1.0 - July 15, 2015. Introducing individual session analysis. Combine
% single day recordings into one daily log file. 
% primate 1.2 - March 8, 2016. Data format is now regular. Removes all
% dashes from the date (2016_12_31 becomes 20161231)
% primate 1.3 - March 28, 2016 - Fixes bug which did not allow to record more than
% one participant
% primate 1.4 - July 29, 2016 - Makes analysis modular, for the purpose of public
% archiving.
% primate 1.5 - September 1, 2016 - Rewrote the code: compratmentalized into
% tiny blocks parsing parts of file name for easier adaptability in future.
% Assumes files are saved as 'subject_name'_'date'_('session_number').
% Output is a list of dates and folder names detected, sorted in
% chronological order.
% v2.0 - September 7, 2016. Renamed the function.
%
% Donatas Jonikaitis


function y = get_path_dates_v20 (path1, subject_name)

% path1 is directory you want to check for data (f.e. ~/Experiment_data/Experiment_name/data_psychtoolbox/subject_name/)
% subject_name is subject initials

%% Setup

% Determine how many files with the subject name are there
index_cur_dir = dir(path1);

% Initialize output variables
temp_ind_subj = cell(length(index_cur_dir), 1);
temp_ind_sessions = NaN(length(index_cur_dir), 1);
temp_ind_dates = NaN(length(index_cur_dir), 1);


%% Parse the file name

% Temporary variables initiated
fnames = cell (length(index_cur_dir), 1); % Temporary file name used
fdates = cell (length(index_cur_dir), 1); % Dates in string format will be stored here

% Extract file names & their sizes
for i=1:length(index_cur_dir)
    fnames{i} = index_cur_dir(i).name;
end

%============
% Extract subject names
for i=1:length(fnames)
    if length(fnames{i}) > length(subject_name)
        if strcmp(fnames{i}(1:length(subject_name)), subject_name)
            temp_ind_subj{i} = subject_name; % Save subject name
            fnames{i}(1:length(subject_name)) = [];
        else
            temp_ind_subj{i} = NaN; % Save subject name
        end
    else
        temp_ind_subj{i} = NaN;
        fnames{i}=[]; % Discard if folder doesnt start with subject name
    end
end

% If there is a dash, extract it
for i=1:length(fnames)
    if length(fnames{i}) > 1
        if strcmp(fnames{i}(1), '_')
            fnames{i}(1) = [];
        end
    end
end

%===============
% Detect if next is year notation
for i=1:length(fnames)
    if length(fnames{i}) > 4
        if str2num(fnames{i}(1:4))
            fdates{i}(1:4) = fnames{i}(1:4);
            fnames{i}(1:4) = [];
        else 
            fdates{i} = NaN;
        end
    else
        fdates{i} = NaN;
    end
end

% If there is a dash, extract it
for i=1:length(fnames)
    if length(fnames{i}) > 1
        if strcmp(fnames{i}(1), '_')
            fnames{i}(1) = [];
        end
    end
end

%================
% Detect if next is month notation
for i=1:length(fnames)
    if length(fnames{i}) > 2
        if str2num(fnames{i}(1:2))
            fdates{i}(5:6) = fnames{i}(1:2);
            fnames{i}(1:2) = [];
        else
            fdates{i} = NaN;
        end
    else
        fdates{i} = NaN;
    end
end

% If there is a dash, extract it
for i=1:length(fnames)
    if length(fnames{i}) > 1
        if strcmp(fnames{i}(1), '_')
            fnames{i}(1) = [];
        end
    end
end

%=================
% Detect if next is day notation
for i=1:length(fnames)
    if length(fnames{i}) >= 2
        if str2num(fnames{i}(1:2))
            fdates{i}(7:8) = fnames{i}(1:2);
            fnames{i}(1:2) = [];
        else
            fdates{i} = NaN;
        end
    else
        fdates{i} = NaN;
    end
end


% If there is a dash, extract it
for i=1:length(fnames)
    if length(fnames{i}) > 1
        if strcmp(fnames{i}(1), '_')
            fnames{i}(1) = [];
        end
    end
end

%=================
% If there is session number afterwards, extract it
% CODE ASSUMES THAT FILES END WITH SESSION NUMBER
for i=1:length(fnames)
    if ~isempty(fnames{i})
        if str2num(fnames{i})
            temp_ind_sessions(i) = str2num(fnames{i});
            fnames{i} = [];
        else
            temp_ind_sessions(i)=NaN;
        end
    else temp_ind_sessions(i)=NaN;
    end
end


%% Transform date matrix from string to number

% Convert dates into numbers
for i=1:length(fdates)
    if ~isempty(fdates{i})
        if ~isnan(fdates{i})
            temp_ind_dates(i)=str2num(fdates{i});
        end
    end
end

% Find unique days used
unique_dates = unique(temp_ind_dates);
ind=isnan(unique_dates); unique_dates(ind)=[];



%% For each date and session prepare a list of paths

index_cur_dir; % This is necessary input. Contains all file names and file ordering

% Create output matrix
folder_name1 = cell(1);
session1 = [];
dates1 = [];
subject1 = cell(1);

for i=1:length(unique_dates)
        
    % If there are more than 10 sessions recorded, re-order the indexes of
    % the sessions into increasing sequence (from 1, 11, 2, 3) into (1, 2, 3, 11)
    index1 = find(temp_ind_dates == unique_dates(i));
    [~, t1] = sort(temp_ind_sessions(index1));
    index1 = index1(t1);
    
    for j=1:length(index1)
        if length(folder_name1)==1 && isempty(folder_name1{1})
            ind = 1;
        else
            ind = ind+1;
        end
        folder_name1{ind,1} = index_cur_dir(index1(j)).name;
        session1(ind,1) = temp_ind_sessions(index1(j));
        dates1(ind,1) = temp_ind_dates(index1(j));
        subject1{ind,1} = temp_ind_subj{index1(j)};
    end
    
end


%% Output

y.index_dates=dates1;
y.index_unique_dates = unique_dates;
y.index_sessions = session1;
y.index_subjects = subject1;
y.index_directory = folder_name1;

