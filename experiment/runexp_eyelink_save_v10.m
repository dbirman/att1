% Saves eyelink file into indicated directory
%
% input 1 - path where to save the file
% input global variable (necessary) - name of default eyelink file
% input global variable (necessary) - subject file name (for example dj1)
% output: file saved
%
% v1.0 DJ: February 10, 2015
%

function runexp_eyelink_save_v10 (directory1)


%% Initialize settings

global expsetup;
edfname1 = expsetup.general.edfname_temporary;
newname1 = sprintf('%s.edf', expsetup.general.subject_filename);


%% Check whether directory exists and change to it

if isdir (directory1)
    cd (directory1);
else
    mkdir(directory1);
    cd (directory1);
end


%% Close file

Eyelink('CloseFile');


%% Download data file

try
    fprintf('\nReceiving eyelink data file\n', edfname1 );
    status=Eyelink('ReceiveFile');
    if 2==exist(edfname1, 'file')
        fprintf('Data file ''%s'' can be found in ''%s''\n', edfname1, pwd );
    end
    Eyelink('ShutDown');
catch
    fprintf('Problem receiving data file ''%s''\n', edfname1 );
    Eyelink('ShutDown');
    return
end


%% Ouptut

movefile(edfname1,newname1);


end