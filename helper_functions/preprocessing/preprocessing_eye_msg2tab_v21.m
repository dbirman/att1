% Extract message time and message names
%
% path_in - folder within which .asc file resides
% f_name - name of the asc file (for example: "dj2016" without .asc notation)
% y - data structure with message times and headers
%
% Function created by Donatas Jonikaitis, inspired by Martin Szinte &
% Martin Rolfs
% First version made by DJ: 05/07/2011
%
% v1.1, July 8, 2015: Adapted to primate setup; Changed how the files are
% loaded
% v2.0 September 6, 2016: Complete over-write. Extracts any message with
% structure: MSG text time. This extracts default eyelink messages and user
% generated messages. The big change is that it does not need user input to
% specify which messages to read out.
% v2.1 February 3, 2017: Fixes the bug in which START or END messages might
% be missing (corrupt edf file)


function preprocessing_eye_msg2tab_v21(path_in, file_name)



%% Analysis

% Open .asc file
try
    path1 = sprintf('%s%s.asc', path_in, file_name);
    msgfid = fopen(path1,'r');
catch
    if msgfid==-1
        fprintf(1,'\n Could not open the file %s.asc ... \n ', file_name)
    end
end

% Initialize output structure
events.msg = {};
events.time = [];
events.saccades=[];
events.sampling_frequency = [];
events.blinks = [];

while ~feof(msgfid) % While end of the file is not reached
    
    % Get line and check if there is data in it
    line = fgetl(msgfid);
    
    % Read data
    if ~isempty(line) % Skip empty lines
        
        % One line; produces cell array; 1st element is MSG, 2nd element is time; 3rd element is message itself; 4rth element (if exists is trial number)
        mat1 = textscan(line,'%s');
        mat1 = mat1{1}; % Remove cell wrapping (comes by default from textscan command)
        
        
        if length(mat1) >= 3 % If message is potentially interesting
            
            % Also extract BlockStart times (block finish is ignored)
            switch char(mat1(1))
                case 'START'
                    events.time(end+1,1) = str2double(char(mat1(2))); % Timestamp
                    events.msg{end+1,1} = char(mat1(1)); % Message
                case 'END'
                    events.time(end+1,1) = str2double(char(mat1(2))); % Timestamp
                    events.msg{end+1,1} = char(mat1(1)); % Message
            end
            
            % Record all message timestamps. Message name is read as a
            % third collumn
            if strcmp (mat1(1), 'MSG')
                events.time(end+1,1) = str2double (char(mat1(2))); % Timestamp
                events.msg{end+1,1} = char(mat1(3)); % Message name
            end
            
            % Extract sampling frequency
            switch char(mat1(1))
                case 'EVENTS'
                    events.sampling_frequency(end+1,1)=str2double(char(mat1(5))); % And if message matches, get the timestamp
            end
            
            % Also extract all saccade onsets from eyelink
            switch char(mat1(1))
                case 'ESACC'
                    i = size(events.saccades,1);
                    events.saccades(i+1,1)=str2double(char(mat1(3))); % Saccade onset
                    events.saccades(i+1,2)=str2double(char(mat1(4))); % Saccade offset
                    events.saccades(i+1,3)=str2double(char(mat1(6))); % x start
                    events.saccades(i+1,4)=str2double(char(mat1(7))); % y start
                    events.saccades(i+1,5)=str2double(char(mat1(8))); % x end
                    events.saccades(i+1,6)=str2double(char(mat1(9))); % y end
                    events.saccades(i+1,7)=str2double(char(mat1(11))); % pupil size
            end
            
            % Also extract all blinks from the eyelink
            switch char(mat1(1))
                case 'EBLINK'
                    i = size(events.blinks,1);
                    events.blinks(i+1,1)=str2double(char(mat1(3))); % Blink onset
                    events.blinks(i+1,2)=str2double(char(mat1(4))); % Blink offset
                    events.blinks(i+1,3)=str2double(char(mat1(5))); % Blink duration
            end
            
            
        end
    end
    % Done reading the line
    
end
% Done reading the file




%% Restructure events file into trial-based file

% Setup trial start and trial end times
index1 = strcmp(events.msg, 'START');
index2 = strcmp(events.msg, 'END');
skip_file = 0;
if sum(index1)~=sum(index2)
    fprintf ('Extracting eyelink messages: START and END count mismatch. Skipping this file')
    skip_file = 1;
elseif sum(index1)==0 || sum(index2)==0
    fprintf ('Extracting eyelink messages: START and END messages dont exist. Skipping this file')
    skip_file = 1;
end

if skip_file == 0 % If all is ok with start and end messages 
    
    % Create structure which contains each message as a field
    unique_fields = unique(events.msg);
    temp1 = struct;
    for j=1:length(unique_fields)
        index1 = strcmp(events.msg, unique_fields{j});
        time1 = events.time(index1);
        try
            temp1.(unique_fields{j})=time1;
        catch % Some messages might start with illegal character, possible to save them
            % name1 = ['unknown_fieldname_', num2str(j)];
            % temp1.(name1) = time1;
        end
    end
    
    % Remove message field if they are contain no timestamps
    % This might happen if message contains other info, like calibration
    % coordinates
    unique_fields = fieldnames(temp1);
    a=cell(1);
    for j=1:length(unique_fields)
        if isempty(temp1.(unique_fields{j}))
            if isempty(a{1})
                a{1} = unique_fields{j};
            else
                a{end+1} = unique_fields{j};
            end
        end
    end
    if ~isempty(a{1})
        temp1 = rmfield(temp1, a);
    end
    
    
    % Create a structure with 1 line per 1 trial of messages
    % At the moment 1 timestamp per message only
    
    t_start = temp1.START; % trial start
    t_end = temp1.END; % trial end
    
    unique_fields = fieldnames(temp1);
    temp2 = struct;
    
    for j= 1:length(unique_fields)
        m_in = temp1.(unique_fields{j});
        m_out = NaN(length(t_start),1);
        for i=1:length(t_start)
            index = m_in>=t_start(i) & m_in<=t_end(i);
            if sum(index)>=1
                a = m_in(index);
                m_out(i)=a(1); % If multiple messages, save ONLY first one
            end
        end
        temp2.(unique_fields{j})=m_out;
    end
    
    % Remove message field if they contain no timestamps
    unique_fields = fieldnames(temp2);
    a=cell(1);
    t_start = temp2.START;
    for j=1:length(unique_fields)
        index = isnan(temp2.(unique_fields{j}));
        if sum(index)==length(t_start)
            if isempty(a{1})
                a{1} = unique_fields{j};
            else
                a{end+1} = unique_fields{j};
            end
        end
    end
    if ~isempty(a{1})
        temp2 = rmfield(temp2, a);
    end
    
    temp2.sampling_frequency = events.sampling_frequency;
    eyelink_events = temp2;
    
    
    %% Save events structure
    
    % Check whether sampling frequency is uniform and terminate file it is
    % not. If sampling is fine, then save the file
    a=unique(eyelink_events.sampling_frequency);
    if length(a)>1
        error ('Non-uniform sampling freqency. No contingency plan at the moment')
    elseif length(a)==1
        path1 = sprintf('%s%s_events.mat', path_in, file_name);
        save (eval('path1'), 'eyelink_events')
    end
    
end




