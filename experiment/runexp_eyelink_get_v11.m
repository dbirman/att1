% Gets newest sample of eyelink recording
%
% global variable (necessary) - eyelinkini structure
% global variable (necessary) - is is eyetracking experiment or not
%
% v1.0 DJ: February 11, 2015
% v1.1 DJ: August 16, 2017. Bug fix.


function [mx, my] = runexp_eyelink_get_v11

%% Settings

global expsetup;
recordeyes = expsetup.general.recordeyes;
if isfield(expsetup, 'eyelink')
    el = expsetup.eyelink;
end

%% Get sample

if recordeyes==1
    eye_used = Eyelink('EyeAvailable');
    try
        evt = Eyelink('NewestFloatSample'); % Get the sample in the form of an event structure
        if eye_used==0 || eye_used==1 % If left or right eye is recorded
            x = evt.gx(eye_used+1); % +1 as we're accessing MATLAB array
            y = evt.gy(eye_used+1);
        elseif eye_used==2
            x = evt.gx(2); % Use right eye
            y = evt.gy(2);
        end
        % Do we have valid data and is the pupil visible?
        if x~=el.MISSING_DATA && y~=el.MISSING_DATA && evt.pa(eye_used+1)>0
            mx=x;
            my=y;
        end
    catch
        mx=NaN;
        my=NaN;
    end
elseif recordeyes==0
    [mx, my]=GetMouse;
else
    fprintf ('\nSpecify whether eyetracking experiment is run or not in runexp_eyelink_get\n')
    mx=NaN; my=NaN;
    return
end

end