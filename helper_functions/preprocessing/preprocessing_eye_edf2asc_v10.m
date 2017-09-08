% Function converts .edf file to .dat and .asc files
%
% pathin - edf file is within this folder
% pathout - asc file will be outputed to this folder
% f_name - file name within the folder (for example: aq_2000_12_31)
%
% Add edf2asc in "/bin" folder; 
% use the file from
% "/Applications/EyeLink/EDF_Access_API/Example/"
% Otherwise this analysis wont work
%
% DJ V1.0 September 6, 2016. Initial version



function preprocessing_eye_edf2asc_v10(path_in, path_out, file_name)


% Create .dat file
try
    path_edf = sprintf('%s%s.edf', path_in, file_name);
    system(['edf2asc', ' ', sprintf('%s', path_edf),' -s -miss -1.0 -y']);
catch
    error('edf2asc conversion failed. Possibly you dont have edf2asc in /bin folder setup')
end
% Convert and move .asc file into .dat file
path_1 = sprintf('%s%s.asc', path_in, file_name);
path_2 = sprintf('%s%s.dat', path_out, file_name);
movefile(sprintf('%s',path_1), sprintf('%s',path_2))

% Create .asc file
try
    system(['edf2asc', ' ', sprintf('%s',path_edf),' -e -y']);
catch
    error('edf2asc conversion failed. Possibly you dont have edf2asc in /bin folder setup')
end
% Move .asc file into it's folder
path_1 = sprintf('%s%s.asc', path_in, file_name);
path_2 = sprintf('%s%s.asc', path_out, file_name);
movefile(sprintf('%s',path_1), sprintf('%s',path_2));


