% Computer settings for particular setup
% Always read through the file to check settings are correct
% output: expsetup.general variable is modified
%
% v1 DJ July 10, 2015
% v2 DJ July 5, 2016. NI daq settings naming changed to be compatible
% across multiple devices connected to it (v1 it was just reward)
% v21 DJ April 1, 2017. Added tcp ip connection settings


function att1_computer_settings_v22_dj
 
global expsetup

% Specify in which folder your experiment is contained
% for example experiment "attention1" is contained in the folder called "experiments"
% In this case set path to experiments


expsetup.general.directory_baseline_code = '~/proj/experiments/';
expsetup.general.directory_baseline_data = '~/proj/experiments_data/';


% This is viewpix display
% Laptop
expsetup.general.monwidth_cm = 28; % Horizontal dimension of viewable screen (cm)
expsetup.general.monheight_cm = 18;
% Desktop
% expsetup.general.monwidth_cm = 52; % Horizontal dimension of viewable screen (cm)
% expsetup.general.monheight_cm = 29.5;
expsetup.general.viewingdist_cm = 60; % Viewing distance (cm)
expsetup.general.fps_expected = []; % Expected refresh rate for the setup; otherwise experiment will be stopped

% expsetup.general.audio_target_device = 'ASIO4ALL v2'; % EXAMPLE: audio device used in the set
expsetup.general.audio_target_device=[];

% General national instruments card settings
expsetup.ni_daq.device_name = 'dev1';

% Reward settings
expsetup.ni_daq.reward_channel_id = 'ao2';
expsetup.ni_daq.reward_rate = 1000;
expsetup.ni_daq.reward_measurement_type = 'Voltage';
expsetup.ni_daq.reward_voltage = 10;

% Stimulator settings
expsetup.ni_daq.stimulator_input_channel_id = 'ai16';
expsetup.ni_daq.stimulator_input_rate = 250000;
expsetup.ni_daq.stimulator_input_measurement_type = 'Voltage';

% Plexon to psychtoolbox connection settings
expsetup.tcpip.plex_address = '192.168.0.2';
expsetup.tcpip.plex_port = 4013;
expsetup.tcpip.psych_address = '192.168.0.1';
expsetup.tcpip.psych_port = 4013;
expsetup.tcpip.success_ini = 0; % Starting state
expsetup.tcpip.data_file_size = 10000; % How many elements in one packet

  
% Path to edf file, PC only
if ispc
    expsetup.general.edf2asc_path = '"C:\Program Files (x86)\SR Research\EyeLink\EDF_Access_API\Example\edf2asc.exe"';
end

% How eyelink files are saved by default?
expsetup.general.edfname_temporary = 'djexp1.edf';

% How is data structure called?
expsetup.general.data_structure_appendix = '_data_structure.mat';

end