% Reads multiple folders recorded on same/different days and produces
% matrix with all sessions and folders recorded
%
% Revision history:
%
% v1.0 September 1, 2016. Basic script prepared.
% Data file is format:
% [subjectName_Date_chChannelNumber_uUnitNumber_unitType.mat]
% (for example: aq_20161231_ch1_u1_m.mat);


function [y] = get_path_spikes_v10 (path1, subject_name)

% Determine how many files with the subject name are there
index_cur_dir = dir(path1);

% Initialize output variables
temp_ind_subj = cell(length(index_cur_dir), 1);
temp_ind_dates = NaN(length(index_cur_dir), 1);
temp_ind_channels = NaN (length(index_cur_dir), 1);
temp_ind_units = NaN (length(index_cur_dir), 1);
temp_ind_unit_class = cell (length(index_cur_dir), 1);

%% Parse the file name

% Temporary variables initiated
fnames = cell (length(index_cur_dir), 1); % Temporary file name used
fdates = cell (length(index_cur_dir), 1); % Dates in string format will be stored here


% Extract file names & their sizes
for i=1:length(index_cur_dir)
    fnames{i} = index_cur_dir(i).name;
end

%============
% Subject names
%============

% Extract subject names
for i=1:length(fnames)
    if length(fnames{i}) > length(subject_name)
        if strcmp(fnames{i}(1:length(subject_name)), subject_name)
            temp_ind_subj{i} = subject_name; % Save subject name
            fnames{i}(1:length(subject_name)) = [];
        else
            temp_ind_subj{i} = NaN; % Save subject name
            fnames{i} = NaN;
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
% Date detection
%===============

% Detect chanel number
for i=1:length(fnames)
    if length(fnames{i}) >= 2
        [a,b] = strsplit(fnames{i}, '_'); % Find delimiter
        if ~isempty(b) % If delimiter exists
            if str2num(a{1})
                temp_ind_dates(i) = str2num(a{1});
                fnames{i}(1:length(a{1})) = [];
            else
                temp_ind_dates(i)=NaN;
                fnames{i}=NaN;
            end
        else
            temp_ind_dates(i)=NaN;
            fnames{i} = NaN;
        end
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
% Channel detection
%=================

% Detect if next is channel notation
for i=1:length(fnames)
    if length(fnames{i}) >= 2
        if strcmp(fnames{i}(1:2), 'ch')
            fnames{i}(1:2) = [];
        else
            fnames{i}=NaN;
        end
    else
        fnames{i} = NaN;
    end
end


% Detect chanel number
for i=1:length(fnames)
    if length(fnames{i}) >= 2
        [a,b] = strsplit(fnames{i}, '_'); % Find delimiter
        if ~isempty(b) % If delimiter exists
            if str2num(a{1})
                temp_ind_channels(i) = str2num(a{1});
                fnames{i}(1:length(a{1})) = [];
            else
                temp_ind_channels(i) = NaN;
                fnames{i}=NaN;
            end
        else
            temp_ind_channels(i) = NaN;
            fnames{i} = NaN;
        end
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
% Unit detection
%=================

% Detect if next is unit number notation
for i=1:length(fnames)
    if length(fnames{i}) >= 1
        if strcmp(fnames{i}(1), 'u')
            fnames{i}(1) = [];
        else
            fnames{i}=NaN;
        end
    else
        fnames{i} = NaN;
    end
end

% Detect unit number
for i=1:length(fnames)
    if length(fnames{i}) >= 2
        [a,b] = strsplit(fnames{i}, '_'); % Find delimiter
        if ~isempty(b) % If delimiter exists
            if str2num(a{1})
                temp_ind_units(i) = str2num(a{1});
                fnames{i}(1:length(a{1})) = [];
            else
                temp_ind_units(i) = NaN;
                fnames{i}=NaN;
            end
        else
            temp_ind_units(i) = NaN;
            fnames{i} = NaN;
        end
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
% Unit classification detection
%=================

for i=1:length(fnames)
    if length(fnames{i}) >= 1
        if strcmp(fnames{i}(1), 'u') || strcmp(fnames{i}(1), 's') || strcmp(fnames{i}(1), 'm')
            temp_ind_unit_class{i}(1) = fnames{i}(1);
            fnames{i}(1) = [];
        else
            temp_ind_unit_class{i}(1) = NaN;
            fnames{i}=NaN;
        end
    else
        temp_ind_unit_class{i}(1) = NaN;
        fnames{i} = NaN;
    end
end


%% For each date and session prepare a list of paths

% Find unique days used
unique_dates = unique(temp_ind_dates);
ind=isnan(unique_dates); unique_dates(ind)=[];

index_cur_dir; % This is necessary input. Contains all file names and file ordering

% Create output matrix
file_name1 = cell(1);
dates1 = [];
subject1 = cell(1);
unit1 = [];
channel1 = [];
unit_class1 = cell(1);
path_temp1 = cell(1);

for i=1:length(unique_dates)
    
    index1 = find(temp_ind_dates == unique_dates(i));
    
    for j=1:length(index1);
        if length(file_name1)==1 && isempty(file_name1{1})
            ind = 1;
        else
            ind = ind+1;
        end
        file_name1{ind,1} = index_cur_dir(index1(j)).name;
        dates1(ind,1) = temp_ind_dates(index1(j));
        subject1{ind,1} = temp_ind_subj{index1(j)};
        unit1(ind,1) = temp_ind_units(index1(j));
        channels1(ind,1) = temp_ind_channels(index1(j));
        unit_class1{ind,1} = temp_ind_unit_class{index1(j)};
        path_temp1{ind,1} = [path1, index_cur_dir(index1(j)).name];
    end
    
end

%% Output

y.index_dates=dates1;
y.index_unique_dates = unique_dates;
y.index_subjects = subject1;
y.index_file_name = file_name1;
y.index_unit = unit1;
y.index_channel = channels1;
y.index_unit_type = unit_class1;
y.index_path = path_temp1;
