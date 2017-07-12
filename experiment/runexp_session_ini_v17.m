% Function creates a subject folder. Automatically checks which repetition
% of the experiment it is and suggests to save session number
%
% input 1: Path to main data folder (necessary)
% output 1: Directory to subject data file (expsetup.general.directory_data_subject)
% output 2: Subject initials (expsetup.general.subject_id)
% output 3: Session number (expsetup.general.subject_session)
% output 4: Initials+session number (expsetup.general.subject_filename)
%
% v1.0 DJ: February 11, 2015
% v1.5 DJ: February 14, 2015. Removes checking for long subject names (no
% limit). Adds date as part of a subject name (for neurophysiology this keeps
% track across different days).
% v1.6 DJ: January 24, 2016. Changes into regular date format, such as "2016_01_25"
% v1.7 DJ: July 31, 2016. Got rid of the code creating the folder "directory_data_main_subject" for
% better compatibility with other codes

function runexp_session_ini_v17 (expdir)


%% Settings

global expsetup;
manual_auto = expsetup.general.filename_auto;
try
    subject_name = expsetup.general.subject_id;
catch
    subject_name=[];
end
fileFormat=expsetup.general.data_structure_appendix; % Which file to check for? (example: '_data_structre.mat')

num_rep1=500; % How many session numbers to check


%% Save variable with participants name

if isempty (subject_name)
    fprintf ('\n')
    sdName=input('Provide participant id: ', 's');
    fprintf ('\n')
elseif ~isempty (subject_name)
    sdName=subject_name;
end


%% Add date as part of participants name

a=clock;
if a(2)<10 && a(3)<10
    num1 = sprintf('%i_0%i_0%i_', a(1),a(2),a(3));
elseif a(2)<10 && a(3)>=10
    num1 = sprintf('%i_0%i_%i_', a(1),a(2),a(3));
elseif a(2)>=10 && a(3)<10
    num1 = sprintf('%i_%i_0%i_', a(1),a(2),a(3));
elseif a(2)>=10 && a(3)>=10
    num1 = sprintf('%i_%i_%i_', a(1),a(2),a(3));
end

sdName=sprintf('%s_%s', sdName, num1);


%% Cycle through number of directories to check whether particular directyr and file exists

n1_empty=NaN(num_rep1,1); % How many repetitions to check
for rep1=1:length(n1_empty)
    
    % Check whether directory  & data file (of specified format) exists
    cdDir=(sprintf('%s%s%i',expdir,sdName,rep1));
    if isdir (cdDir)
        cd(cdDir); % Change to directory
        datfile = sprintf('%s%i%s',sdName,rep1,fileFormat);
        if exist(datfile,'file')
            n1_empty(rep1)=1; % Save whether directory contains data file
        elseif  ~exist(datfile,'file')
            n1_empty(rep1)=0; % Save whether directory does not contain data file
        end
    else
        n1_empty(rep1)=0;
    end
    cd (expdir) % Change back to main directory
    
end


%% Determine which recording is missing

a=find(n1_empty==0);
n1_blockNo=a(1);


%% Show instruction text onscreen

if n1_blockNo==1;
    
    fprintf ('\n\nWelcome to the first session of the experiment! \n')
    fprintf ('If the data is properly synced and there are no missing files on this computer \n')
    fprintf ('Then I know which session and condition you have to do \n')
    
    if manual_auto==1
        fprintf ('You just have to confirm the session number or enter any number you want \n')
        fprintf ('Just follow instructions onscreen \n')
    elseif manual_auto==2
        fprintf ('All directories will be created automatically \n')
    end
    
else
    
    t1_text{1}= 'My records show that you will be running session number ';
    t1_text{2}= 'Let me see... you will be running session number ';
    t1_text{3}= 'I used some computer magic to determine that it is session ';
    t1_text{4}= 'You are about to start session number ';
    t1_text{5}= 'Ta-da! It is session number ';
    t1_text{6}= 'On we go with session number ';
    t1_text{7}= 'Specified session number is ';
    
    a=Shuffle(1:length(t1_text));
    temp_text1 = sprintf('%s%i',t1_text{a(1)}, n1_blockNo);
    fprintf ('%s\n', temp_text1)
    
end


%% Wait for confirmation of the session number (if manual, else skip it)

if manual_auto==1
    InitChoicexx=0;
    while InitChoicexx ~= 1;
        n1_temp = input('\nEnter "y" to confirm or "n" to specify new session number yourself   ', 's');
        if strcmp(n1_temp,'y')
            s1 = sprintf('%s%i',sdName, n1_blockNo);
            fprintf ('\nDataset %s will be initialized \n', s1)
            InitChoicexx=1;
            % If empty folder exists, remove it
            cdDir=(sprintf('%s%s%i',expdir,sdName,n1_blockNo));
            if isdir (cdDir)
                rmdir(cdDir, 's');
            end
        elseif strcmp(n1_temp,'n');
            fprintf ('\n')
            n1_blockNo = input('Enter session number:   ');
            s1 = sprintf('%s%i',sdName, n1_blockNo);
            fprintf ('\nChecking whether dataset %s can be created \n', s1)
            InitChoicexx=1;
        end
    end
end


%% Checks one more time whether selected name exists or not

% Check whether selected directory & data file exists
cdDir=(sprintf('%s%s%i',expdir,sdName,n1_blockNo));
if isdir (cdDir)
    dir_exists=1;
else
    dir_exists=0;
end

% If name exist, prompt for choice
if dir_exists==1 && manual_auto==1
    
    string=input('Such session number already exists. You want to overwrite ("o"), specify new session number ("n") or quit ("q")?  ', 's');
        
    if strcmp (string,'o') % Overwrite
        
        rmdir(cdDir, 's');
        fprintf('\nDirectory %s%i will be over-written \n', sdName, n1_blockNo )
        
    elseif strcmp (string,'n') % Or new
        quit_loop1=0;
        n1_blockNo=input('Enter session number: ');
        while quit_loop1~=1
            cdDir=(sprintf('%s%s%i',expdir,sdName,n1_blockNo));
            if isdir (cdDir)
                quit_loop1=0;
            else
                quit_loop1=1;
            end
            if quit_loop1==0
                n1_blockNo=input('This session number already exists! Use another session number: ');
            end
        end
        
    elseif strcmp (string,'q') % Or just quit
        error ('No subject id/session number provided')
    else % Or just quit
        error ('No subject id/session number provided')
    end

elseif dir_exists==1 && manual_auto==2 % Remove existing directory (if its empty)
    rmdir(cdDir, 's');
end


%% Output

expsetup.general.subject_session = n1_blockNo;
expsetup.general.subject_filename = sprintf('%s%i',sdName,n1_blockNo);


end