% Preprocessing of raw data files - combining them, adding eye movements
% Make sure to provide folders for data access
% Typically, this file does not have to be modified, unless recorded data
% formats change
%
% Donatas Jonikaitis
%
% V2.0: 6 September 2016. Combines psycthoolbox, eyelink and plexon files and
% saves them. It does not do any data analysis.
% Only prerequisite is that START message exists in eyelink
% messages file (to parse trials, can be replaces with any other
% message).
% V2.0 24 September 2016. Combines matrices only if all data is present
% V2.1 27 October 2016. Separated spike detection from event detection due
% to bugs in pl2 file export.
% V2.2 31 October 2016. Removed spike detection as it made too many
% complications given plexon data formats. Now it deals with eyelink and
% psychtoolbox data only.
%
% Format:
% preprocessing_psychtoolbox_v20 (settings, i_date, varargin)
% settins: settings containing path to data folders;
% i_date: current date to be analysed (f.e. 20160101). Takes only one date
% at a time.
% varargin text input: names of fields to be treated as special inputs to
% be matched to eyelink input; This is due to differences in how expmatrix
% and how eyelink data are saved (trial counts do not match on them).
%
% Associated codes necessary for this to work:
% preprocessing_eye_edf2asc_v10
% preprocessing_eye_msg2tab_v20
% preprocessing_eye_saccades_EK2003_v14
% get_path_dates_v20 output must be stored in settings file: fields
% settings.index_dates, settings.index_directory, settings.subject_name
% are necessary for the code to work


function preprocessing_psychtoolbox_v22(settings, i_date, varargin)

% %==============
% % For debugging purposes initialize few variables

% varargin{1}='stim.expmatrix';
% varargin{2}='stim.trialmatrix';
% varargin{3}='stim.refresh_rate_mat';
% clear S;
% try
%     i_date = date_index(i_date);
% end

%==============

% Pre-specified folder names imported from settings
% Right hand side of equation can be easily substituted with other folder
% names wihtout rewriting the rest of this code

settings.path_data_psychtoolbox = settings.path_data_psychtoolbox; % Psychtoolbox settings
settings.path_data_eyelink_edf = settings.path_data_eyelink_edf; % Eyelink input
settings.path_data_eyelink_asc = settings.path_data_eyelink_asc; % Eyelink output
settings.path_data_output = settings.path_data_combined; % To which folder the data is saved

overwrite_saccades_EK2006 = 0; % 1 - will over-write saccade detection for EK algorithm


%% Do analysis

ind_session = find(i_date==settings.index_dates);

% Do analysis for each desired session
% No changes needed for this section
for i_session = 1:length(ind_session)
    
    
    %% Folder name to be used
    
    folder_name = settings.index_directory{ind_session(i_session)};
    
    
    %% Load psychtoolbox data (no processing needed)
    
    file_name = [folder_name, '_data_structure.mat']; % How file is called
    path_psych = [settings.path_data_psychtoolbox, folder_name, '/', file_name];

    if exist (path_psych, 'file')
        varx = load(path_psych);
    else
        varx = struct;
    end
    if ~isempty(fieldnames(varx))
        fprintf('\nPsychtoolbox file %s successfully loaded\n', file_name)
    else
        fprintf('\nPsychtoolbox file %s does not exist; it was not loaded\n', file_name)
    end
    
    % Extract a structure which is in the var1
    var1 = struct; % Initialize empty structure
    f1 = fieldnames(varx);
    if length(f1)==1
        var1 = varx.(f1{1});
    end

    
    %  Save the path of loaded files
    var3 = struct;
    if isempty(fieldnames(var1))
        var3 = struct;
    elseif ~isempty(fieldnames(var1))
        var3.settings_preprocessing.path_psychtoolbox{i_session} = path_psych;
    end
    
    % var1 is psychtoolbox file
    % var3 contains paths to the files loaded
    
    
    
    
    %% Load/preprocess eye-tracker data
    
    
    % General path setup
    path_in=[settings.path_data_eyelink_edf, folder_name, '/']; % Load edf file
    path_out=[settings.path_data_eyelink_asc, folder_name, '/']; % Save into edf folder
    file_name = folder_name;
    
    if ~isdir(path_out)
        mkdir (path_out)
    end
    
    %============
    % Convert .edf into .asc .dat
    
    path_asc = sprintf('%s%s.asc', path_out, file_name); % Full path to .asc file
    path_edf = sprintf('%s%s.edf', path_in, file_name); % Full path to .asc file
    if ~exist (path_asc,'file') && exist (path_edf,'file') % If asc file doesn't exist - do the conversion
        preprocessing_eye_edf2asc_v10(path_in, path_out, file_name) % Script doing conversion
        if exist (path_asc,'file')
            fprintf ('\n\nFile %s.edf was converted to .asc and .dat files \n', file_name)
        end
    elseif ~exist (path_asc,'file') && ~exist (path_edf,'file')
        fprintf ('Eyelink files %s.edf does not exist, skipping edf conversion  \n', file_name)
    else
        fprintf ('Eyelink files  .asc and .dat in the folder %s exist, no conversion done  \n', file_name)
    end
    
    
    %============
    % Get eyelink messages from .asc file
    
    % Check whether message file exist. If it does, skip analysis
    path_events = sprintf('%s%s_events.mat',path_out, file_name); % Full path to events file
    if ~exist (path_events,'file')
        path_asc = sprintf('%s%s.asc', path_out, file_name); % Full path to .asc file
        if exist (path_asc,'file')
            preprocessing_eye_msg2tab_v20(path_out, file_name);
        else
            fprintf ('.asc file does not exist. Can not proceed with event extraction\n')
        end
        if exist(path_events,'file')
            fprintf ('Extracted all messages into file %s_events successfully.\n', file_name)
        end
    else
        fprintf ('Messages file %s_events already exist, no conversion done  \n', file_name)
    end
    
    % Load messages file
    if exist (path_events, 'file')
        varx = load(path_events);
    else
        varx = struct;
    end
    
    % Extract a structure which is in the var1
    var4 = struct;
    f1 = fieldnames(varx);
    if length(f1)==1
        var4.eyelink_events = varx.(f1{1});
    end
    
    
    %==============
    % Find saccades using Engbert & Kliegl (2003) algorithm
    
    
    % Extract saccades
    path_saccades_EK = sprintf('%s%s_saccades_EK.mat', path_out, file_name); % Full path to saccades_EK file
    
    
    % IF converted file does not exist, then do conversion
    if ~exist (path_saccades_EK, 'file')  || overwrite_saccades_EK2006==1
        
        path_dat = sprintf('%s%s.dat', path_out, file_name);  % Full path to .asc file
        
        if exist (path_dat,'file')
            
            % Saccade detection settings
            const.SAMPRATE  = []; % Sampling rate of Eye Tracker
            const.velSD     = 6; % Lambda threshold for saccade/microsaccade detection
            const.minDur    = 6; % Duration threshold for saccade detection
            const.VELTYPE   = 2; % Velocity type for saccade detection (using default)
            const.mergeInt  = 20; % Interval within which saccades/microsaccades will be merged; ms
            const.SAMPRATE = var4.eyelink_events.sampling_frequency(1); % Assumes sampling frequency does not change
            const.trial_start = var4.eyelink_events.START;
            const.trial_end = var4.eyelink_events.END;
            
            fprintf('Processing %s eye data based on Engbert & Kliegl (2006) algorithm \n', file_name);
            preprocessing_eye_saccades_EK2003_v14(path_out, file_name, const); % Eye movement analysis
        else
            fprintf ('.dat file does not exist. Can not proceed with saccade detection\n')
        end
        if exist(path_saccades_EK,'file')
            fprintf ('Prepared saccades file %s_saccades_EK based on Engbert & Kliegl (2006); conversion complete \n', file_name)
        end
    else
        fprintf ('Saccades file %s_saccades_EK already exist, no conversion done  \n', file_name)
    end
    
    % Load saccades file
    if exist (path_saccades_EK, 'file')
        varx = load(path_saccades_EK);
    else
        varx = struct;
    end
    
    % Extract a structure which is in the var1
    var2 = struct;
    f1 = fieldnames(varx);
    if length(f1)==1
        var2.eye_data = varx.(f1{1});
    end
    
    % Save the path of loaded files
    if ~isempty(fieldnames(var3))
        var3.settings_preprocessing.path_eyelink_in{i_session} = path_in;
        var3.settings_preprocessing.path_eyelink_out{i_session} = path_out;
    end
    
    % var2 is eye data - saccaddes using Engbert & Kliegl (2003) algorithm
    % var4 contains all eyelink detected messages
    % var3 contains paths to the files loaded
    
    
    %% Combine psychtoolbox & eyelink into one structure
    
    comb_mat = struct;
    
    % Only if all data was successfully loaded
    a = [~isempty(fieldnames(var1)); ~isempty(fieldnames(var2)); ...
        ~isempty(fieldnames(var3)); ~isempty(fieldnames(var4))];
    
    if sum(a)==4
        for rep1 = 1:4
            
            % Select variable for combining
            % Particular order is just for convenience of displaying structure
            % in command window
            if rep1==1
                temp1 = var3; % Path to files
            elseif rep1==2
                temp1 = var1; % Psychtoolbox matrix
            elseif rep1==3
                temp1 = var4; % Eye messages
            elseif rep1==4
                temp1 = var2; % Saccades
            end
            
            % Generic code to combine fields and subfields
            f1 = fieldnames(temp1);
            for i=1:length(f1)
                if isstruct(temp1.(f1{i}))
                    f2 = fieldnames(temp1.(f1{i}));
                    for j=1:length(f2)
                        n1 = [f2{j}];
                        comb_mat.(f1{i}).(n1){1}=temp1.(f1{i}).(f2{j});
                    end
                else
                    comb_mat.(f1{i}){1}=temp1.(f1{i});
                end
            end
            
        end
    else
        fprintf('Not all files (eyelink, psychtoolbox) could be loaded, omitting this recording\n')
    end
    
    
    %% Combine multiple sessions into one strucutre called S
    
    
    if ~isempty(fieldnames(comb_mat))
        
        temp0 = comb_mat; % Matrix with data
        temp0.session{1} = ones(size(temp0.eyelink_events.START{1}, 1), 1); % Save session number
        temp0.date{1} = ones(length(temp0.eyelink_events.START{1}), 1) * i_date; % Save current date
        
        if  ~exist('S', 'var') % First repretition
            S = temp0;
            repetition1 = 0; % Repetition number
            
        else % Second and later repetitions
            
            repetition1 = repetition1+1; % Repetition number
            temp0.session{1} = temp0.session{1} + max(S.session{end}); % Update session number
            f1 = fieldnames(temp0); % Fields of a new matrix
            for i=1:length(f1)
                
                %===============
                % If a field of structure is another structure, then combine
                % fields in structure.structure.fieldname
                
                if isstruct(temp0.(f1{i}))
                    
                    % Read out fields of newly loaded structure and combine
                    f2 = fieldnames(temp0.(f1{i})); % New structure fields
                    for j=1:length(f2)
                        % If such field already exists, concatenate
                        if isfield(S.(f1{i}), f2{j})
                            S.(f1{i}).(f2{j}) = [S.(f1{i}).(f2{j}); temp0.(f1{i}).(f2{j})];
                            % If field does not exist, create an empty one
                        else
                            % Create empty matrix and save it
                            a1 = cell(repetition1,1);
                            S.(f1{i}).(f2{j}) = [a1; temp0.(f1{i}).(f2{j})];
                        end
                    end
                    
                    % Readout fields of the older structure and see if new
                    % structure lacks any
                    f3 = fieldnames(S.(f1{i})); % Old structure fields
                    if length(f2)==length(f3)
                        % If fields of both structures match, you are good
                    else % If field in a new recording does not exist, add empty cell
                        [~, ~, if3] = intersect(f2,f3);
                        f3(if3) = []; % Remove intersecting fields
                        for i_f3=1:length(f3)
                            a1 = cell(1);
                            S.(f1{i}).(f3{i_f3}) = [S.(f1{i}).(f3{i_f3}); a1];
                        end
                        
                    end
                    
                    %==================
                    % If it is structure, combine it's fields in
                    % structure.fieldname
                else
                    % If such field already exists, concatenate
                    if isfield(S, f1{i})
                        S.(f1{i}) = [S.(f1{i}); temp0.(f1{i})];
                        % If field does not exist, create an empty one
                    else
                        % Create empty matrix and save it
                        a1 = cell(repetition1,1);
                        S.(f1{i}) = [a1; temp0.(f1{i})];
                    end
                end
            end
        end
        
    end

end
% End of analysis for separate sessions; From now on all data is
% combined


%% Combine some variables into matrixes
% This section can be disabled if needed

%=============
% Concatenate the fields of eyelink data (done by default)
% One row - one trial
if exist('S', 'var')
    temp1 = S.eye_data;
    f1 = fieldnames(temp1);
    
    for j=1:length(f1)
        
        % Extract dimensions from each field
        dim1=NaN(size(S.session,1),1); 
        dim2=NaN(size(S.session,1),1); 
        dim3=NaN(size(S.session,1),1);
        for rep1=1:size(S.session,1)
            if ndims(temp1.(f1{j}){rep1})==2 % Get the size
                [dim1(rep1), dim2(rep1)] = size(temp1.(f1{j}){rep1});
            else
                error('Combining sessions not written for 3D matrix')
            end
            dim3(rep1) = size(S.session{rep1},1);
        end
        
        % If the field size matches session size, concatenate it
        if dim1==dim3
            temp1.(f1{j})= cat(1, temp1.(f1{j}){:});
        end
        
    end
    S.eye_data = temp1;
end


%=============
% Concatenate the fields of eyelink messages (done by default)
% One row - one trial.
% Also control for fact that sometimes messages might not be recorded on
% some blocks

if exist('S', 'var')
    
    temp1 = S.eyelink_events;
    f1 = fieldnames(temp1);
    
    for j=1:length(f1)
        
        % Extract dimensions from each field
        dim1=NaN(size(S.session,1),1); 
        dim2=NaN(size(S.session,1),1); 
        dim3=NaN(size(S.session,1),1);
        for rep1=1:size(S.session,1)
            if ndims(temp1.(f1{j}){rep1})==2 % Get the size
                [dim1(rep1), dim2(rep1)] = size(temp1.(f1{j}){rep1});
            else
                error('Combining sessions not written for 3D matrix')
            end
            dim3(rep1) = size(S.session{rep1},1);
        end
        
        % If the field size matches session size, concatenate it
        if dim1==dim3
            temp1.(f1{j})= cat(1, temp1.(f1{j}){:});
        end
        
        % If particular messages are missing, still concatenate by creating empty field
        if any(dim1==0)
            for i=1:length(dim1)
                if dim1(i)==0 && ismatrix(temp1.(f1{j}){i})
                    temp1.(f1{j}){i}=NaN(dim3(i),1);
                elseif dim1(i)==0 && iscell(temp1.(f1{j}){i})
                    temp1.(f1{j}){i}=cell(dim3(i),1);
                end
            end
            temp1.(f1{j})= cat(1, temp1.(f1{j}){:});
        end
        
        
    end
    S.eyelink_events = temp1;
end

%=============
% Concatenate user input fields (done only for user input fields)
% For example, expmatrix is concatenated to one row one trial form

if exist ('S', 'var')
    for i1=1:length(varargin)
        
        b1=strsplit(varargin{i1}, '.');
        
        if size(b1,2)==2
            
            mat1 = S.(b1{1}).(b1{2}); % Structure to be re-organized
            mat_s = S.session; % Structure with session numbers
            num_s = size(mat_s,1); % Total number of sessions tested
            
            % Extract dimensions from each field
            dim1=NaN(size(S.session,1),1);
            dim2=NaN(size(S.session,1),1);
            dim3=NaN(size(S.session,1),1);
            for rep1=1:num_s
                if ndims(mat1{rep1})==2 % Get the size
                    [dim1(rep1), dim2(rep1)] = size(mat1{rep1});
                else
                    error('Combining sessions is only possible for two dimensional matrix/structure')
                end
                dim3(rep1) = size(mat_s{rep1},1);
            end
            
            % If the field size matches session size, concatenate it
            if dim1==dim3 % Combine vertically
                if length(unique(dim2))==1
                    S.(b1{1}).(b1{2}) = cat(1, mat1{:});
                else
                    S.(b1{1}).(b1{2}) = mat1; % No combining
                end
            elseif dim2==dim3 % Combine horizontally
                if length(unique(dim1))==1
                    S.(b1{1}).(b1{2})= cat(2, mat1{:})'; % Flip to vertical, to keep up with conventions
                else
                    S.(b1{1}).(b1{2}) = mat1; % No combining
                end
                
            elseif dim1>=dim3  % Remove some rows if there are too many; Combine horizontally
                if length(unique(dim2))==1
                    for rep1=1:num_s
                        x1 = mat1{rep1};
                        x1 = x1(1:size(mat_s{rep1},1),:);
                        mat1{rep1} = x1;
                    end
                    S.(b1{1}).(b1{2})=cat(1,mat1{1:end});
                else
                    S.(b1{1}).(b1{2}) = mat1;
                end
            else
                error ('Combining sessions was not prepared for this contingency')
            end
            
        else
            error ('Combining matrices workds only for structure arrays at the moment')
        end
        
    end
    
    
    S.session = cat(1,S.session{1:end});
    S.date = cat(1,S.date{1:end});
end

%% Convert cells with repetitions into cells for each trial
% This makes later analysis easier

%===========
% Convert single-settings into matrix of setting per trial

if exist('S', 'var')
    fnames1 = fieldnames(S);
    for i_field1 = 1:length(fnames1)
        if isstruct(S.(fnames1{i_field1}))
            fnames2 = fieldnames(S.(fnames1{i_field1}));
            for i_field2 = 1:length(fnames2) % Go to second level structure
                
                var_temp = S.(fnames1{i_field1}).(fnames2{i_field2}); % Retrieve field of interest (this is data)
                
                if iscell(var_temp) % Only if field is a cell (and not a matrix, matrix means it was already combined earlier)
                    
                    if size(var_temp, 1) == max(S.session) % If a number of recorded sessions exist as a row vector
                        
                        % Determine how big cell array with inputs is
                        m=NaN(max(S.session), 1);
                        n=NaN(max(S.session), 1);
                        o=NaN(max(S.session), 1);
                        for rep1 = 1:max(S.session)
                            [m(rep1),n(rep1),o(rep1)] = size(var_temp{rep1});
                        end
                        
                        a = cell(1); % Initialize empty cell
                        
                        % For each session extract values
                        if sum(m)<=max(S.session) % Only if data is a row vector
                            
                            for rep1 = 1:max(S.session)
                                num_t = sum(S.session==rep1); % How many trials per session are expected?
                                b=cell(num_t,1);  % Initialize empty cell
                                
                                % If it's a character or a row vector, then
                                % replicate values for each trial
                                if isnumeric (var_temp{rep1}) || ...
                                        ischar (var_temp{rep1})
                                    for rep2 = 1:num_t
                                        b{rep2,1}=var_temp{rep1};
                                    end
                                    a{rep1} = b;
                                    % If cell is empty, then fill it with NAN
                                    % values
                                elseif isempty(var_temp{rep1})
                                    for rep2 = 1:num_t
                                        b{rep2,1}=NaN;
                                    end
                                    a{rep1} = b;
                                end
                            end
                        end
                        
                        a = cat(1, a{:}); % Concatenate
                        
                        % Will work only if variables exist
                        if isempty (a)
                            % Do nothing
                        else
                            S.(fnames1{i_field1}).(fnames2{i_field2}) = a;
                        end
                        
                    end
                    
                end
            end
        end
    end
end


%% Split off raw saccade data

if exist ('S', 'var')
    SR.eye_raw = S.eye_data.eye_raw; % With blinks removed
    SR.eye_preblink = S.eye_data.eye_preblink; % Without blinks removed
    SR.session = S.session;
    SR.date = S.date;
    % Remove raw saccade data
    S.eye_data = rmfield(S.eye_data, 'eye_raw');
    S.eye_data = rmfield(S.eye_data, 'eye_preblink');
end



%% Save

if exist('S', 'var')
    
    folder_name = [settings.subject_name, num2str(i_date)];
    path1=[settings.path_data_output, folder_name, '/'];
    
    if isdir(path1)
        rmdir(path1, 's')
        fprintf('\nPreprocessing folder already exists, contents cleared\n')
        mkdir(path1);
    else
        fprintf('\nCreated new preprocessing directory\n')
        mkdir(path1);
    end
    
    
    % Save main data
    path_in = [path1, folder_name, '_settings'];
    save (eval('path_in'), 'S')
    
    % Save raw eye position data separately, as usually it does not need to be
    % loaded
    path_in = [path1, folder_name, '_eye_traces'];
    save (eval('path_in'), 'SR')
    
    % Output
    fprintf('Preprocessed data successfully saved under following name:\n')
    fprintf ('%s\n', path1)
else
    fprintf('Data for given day did not exist, no files saved\n')
end


