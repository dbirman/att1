% Initialized audio setup
%
% "expsetup.general.audio_target_device" is necessary, even if empty
%
% Multiple outputs as structure 'audio' containing all initalized audio
% properties
%
% v1.0 DJ: February 11, 2015

function runexp_audio_ini_v10


%% Settings

global expsetup;
audio_target_device = expsetup.general.audio_target_device;


%% Initialize psych sound

InitializePsychSound;
devices = PsychPortAudio('GetDevices');
nDevices = size(devices);
reqlatencyclass=2; % Class 2 empirically the best, 3 & 4 ~= 2


%% Check whether audio-target device matches one provided

if ~isempty(audio_target_device)
    
    % Check whether device exists
    for i=fliplr(1:nDevices(2)) % Find a device with a specified name
        if strcmp(devices(i).DeviceName, audio_target_device)==1
            TargetID = devices(i).DeviceIndex;
        end
    end
    
    % Try to open audio handle
    try
        audio.handle = PsychPortAudio('Open', TargetID, [], reqlatencyclass, [], []);
    catch
        error ('Could not open an audio device from PsychPortAudio')
    end
    
else % If target device was not specified use defauld device
    try
        audio.handle = PsychPortAudio('Open', [], [], reqlatencyclass, [], []);
    catch
        error ('Could not open an audio device from PsychPortAudio')
    end
end

% Give feedback initiation was successful
for rep1=1:2
    freq=300;
    mynoise=[];
    mynoise(1,:) = MakeBeep(freq, 0.2, []);
    mynoise(2,:) = mynoise(1,:);
    PsychPortAudio('FillBuffer', audio.handle, mynoise); % Fill the audio playback buffer with the audio:
    PsychPortAudio('Start', audio.handle, 1, 0, 0);
    PsychPortAudio('Stop', audio.handle, 1);
    WaitSecs(0.1);
end


%% Save the device used for the sound generation

for i=1:nDevices(2) % Find a device which is being used for the audio channels
    if devices(i).DeviceIndex==audio.handle
        audio.device=devices(i);
    end
end


%% Output

expsetup.audio = audio;

end

