% Plot figures for visualising the data

% % Graph properties
% wlinegraph = 1; % Width of line for the graph
% fontsz = 8; fontszlabel = 10;
% 
% color1(1,:,:) = [0.2, 0.2, 0.2]; % Correct trial
% color1(2,:,:) = [0.2, 0.2, 1]; % Error 1 - fix not acquired
% color1(3,:,:) = [0.2, 0.2, 1]; % Error 2 - drift not maintained
% color1(4,:,:) = [0.9, 0.8, 0.2]; % Error 3 - fix broken
% color1(5,:,:) = [1, 0.3, 0.3]; % Error 4 - Looked at wrong target
% color1(6,:,:) = [1, 0.7, 0.7]; % Error 5 - Saccade not initiated error
% color1(7,:,:) = [1, 0.7, 0.7]; % Error 6 - left saccade target
% color1(8,:,:) = [0.5, 0.5, 0.5]; % Error 99 - Unknown error
% 
% color1(9,:,:) = [0.2, 0.2, 1]; % Texture
% color1(10,:,:) = [0.2, 0.2, 0.2]; % No texture
% 
% close all
% hfig = figure;
% 
% if expsetup.general.plexon_online_spikes == 1
%     set(hfig, 'units', 'normalized', 'position', [0.1, 0.4, 0.8, 0.5]);
%     fig_size = [0,0,8,5];
% elseif expsetup.general.plexon_online_spikes == 0
%     set(hfig, 'units', 'normalized', 'position', [0.1, 0.4, 0.8, 0.25]);
%     fig_size = [0,0,8,2.2];
% end
% 
% 
% %===========================
% 
% %% FIGURE 1
% 
% % Correct/error rates
% 
% if expsetup.general.plexon_online_spikes == 1
%     hfig = subplot(2,4,1); hold on
% elseif expsetup.general.plexon_online_spikes == 0
%     hfig = subplot(1,4,1); hold on
% end
% 
% look2_online_plot_fig1
% 
% %% FIGURE 2
% 
% % Correct/errors over time
% 
% if expsetup.general.plexon_online_spikes == 1
%     hfig = subplot(2,4,2); hold on
% elseif expsetup.general.plexon_online_spikes == 0
%     hfig = subplot(1,4,2); hold on
% end
% 
% look2_online_plot_fig2
% 
% %% FIGURE 3
% 
% % Eye position
% 
% if expsetup.general.recordeyes==1
%     
%     if expsetup.general.plexon_online_spikes == 1
%         hfig = subplot(2,4,[3,4]); hold on
%     elseif expsetup.general.plexon_online_spikes == 0
%         hfig = subplot(1,4,[3,4]); hold on
%     end
%     
%     look2_online_plot_fig3
% end
% 
% 
% %% Make sure figures are plotted
% 
% drawnow;
% 
% %% Save data for inspection
% 
% 
% if expsetup.general.plexon_online_spikes==1
%     
%     dir1 = [expsetup.general.directory_baseline_data, expsetup.general.expname, '/figures_plex_online/', expsetup.general.subject_id, '/', expsetup.general.subject_filename, '/'];
%     if ~isdir (dir1)
%         mkdir(dir1)
%     end
%     file_name = [dir1, 'trial_' num2str(tid)];
%     set(gcf, 'PaperPositionMode', 'manual');
%     set(gcf, 'PaperUnits', 'inches');
%     set(gcf, 'PaperPosition', fig_size);
%     set(gcf, 'PaperSize', [fig_size(3), fig_size(4)])
%     print(file_name, '-dpdf');
% end