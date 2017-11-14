% Plot each trial with raw data saccade data
% Latest revision: October 17, 2017
% Donatas Jonikaitis

close all;

% Show file you are running
p1 = mfilename;
fprintf('\n=========\n')
fprintf('\n Current file:  %s\n', p1)
fprintf('\n=========\n')

% Loading the files needed
if ~exist('settings', 'var')
    settings = struct;
end
settings = get_settings_ini_v10(settings);


%% Some settings

settings.figure_folder_name = 'saccades_raw_traces';
settings.fig_size_current = [0, 0, 6, 2.5]; % Unique figure size settings in this case

% How many trials per error type to plot?
trials_to_plot = 25;
% trials_to_plot = 'all';


%% Run preprocessing

for i_subj=1:length(settings.subjects)
    
    % Select curent subject
    settings.subject_current = settings.subjects{i_subj};
    
    % Get subject folder paths and dates to analyze
    settings = get_settings_path_and_dates_ini_v11(settings);
    dates_used = settings.data_sessions_to_analyze;
    
    % Analysis for each day
    for i_date = 1:numel(dates_used)
        
        % Current folder to be analysed (raw date, with session index)
        date_current = dates_used(i_date);
        ind = date_current==settings.index_dates;
        folder_name = settings.index_directory{ind};
        
        % Figure folder
        path_fig = sprintf('%s%s/%s/%s/', settings.path_figures, settings.figure_folder_name, settings.subject_current, folder_name);
        
        % Overwrite figure folders
        if ~isdir(path_fig)
            mkdir(path_fig)
        elseif isdir(path_fig)
            rmdir(path_fig, 's')
            mkdir(path_fig)
        end
        
        % Data folders
        path1 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '.mat'];
        path2 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_eye_traces.mat'];
        path3 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_saccades.mat'];
        path4 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_individual_saccades.mat'];
        
        % Load all settings
        S = get_struct_v10(path1);
        saccraw1 = get_struct_v10(path2);
        sacc1 = get_struct_v10(path3);
        sacc1_individual = get_struct_v10(path4);
        
        
        %% Select trials to plot
        
        if ischar(trials_to_plot)
            ind_trials = 1:length(S.START);
        else
            if trials_to_plot < length(S.START)
                ind_trials = [];
                a = unique(sacc1.trial_accepted);
                for i=1:numel(a)
                    b = strcmp (sacc1.trial_accepted, a(i));
                    b = find(b==1);
                    if length(b)<=trials_to_plot
                        c = b;
                    else
                        b_perm_ind = randperm(length(b));
                        b=b(b_perm_ind);
                        c = b(1:trials_to_plot);
                    end
                    if size(c,2)>1
                        c=c';
                    end
                    ind_trials=[ind_trials;c];
                end
                
            else
                ind_trials = 1:length(S.START);
            end
        end
        
        
        %% Plot
        
        for tid_0=1:length(ind_trials)
            
            tid = ind_trials(tid_0);
            
            % Data to be used in the trial
            sx1 = saccraw1.eye_processed{tid}; % Raw eye-traces
            sx2 = S.saccades_EK{tid}; % Saccades data
            ind = sacc1_individual.trial_no==tid;
            s_class = sacc1_individual.sacc_classify(ind);
            
            
            h = subplot(1,3,[1]);
            hold on;
            
            %================
            % Plot fixation
            
            % Size
            objsize = S.esetup_fixation_size(tid,3:4);
            % Position
            f_rad = S.esetup_fixation_radius(tid);
            f_arc = S.esetup_fixation_arc(tid);
            [xc,yc] = pol2cart(f_arc*pi/180, f_rad);
            pos1=[xc,yc];
            % Color
            fcolor1 = S.esetup_fixation_color(tid,:)./255;
            % Shape
            if strcmp(S.esetup_fixation_shape{tid}, 'circle') || strcmp(S.esetup_fixation_shape{tid}, 'empty_circle')
                v1 = 1;
            elseif strcmp(S.esetup_fixation_shape{tid}, 'square') || strcmp(S.esetup_fixation_shape{tid}, 'empty_square')
                v1 = 0;
            end
            
            % Plot
            h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                'EdgeColor', fcolor1, 'FaceColor', fcolor1, 'Curvature', v1, 'LineWidth', 1);
            
            %===============
            % Plot fixation tracking window
            
            % Size
            objsize = S.esetup_fixation_size_eyetrack(tid,3:4) * 2; % Eyetracking windows are double size
            
            % Plot
            h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                'EdgeColor', fcolor1, 'FaceColor', 'none', 'Curvature', 1, 'LineWidth', 1);
            
            
            %==========
            % Plot saccade target 1
            
            if ~isnan(S.esetup_st1_coord(tid,1))
                
                %==============
                % Size
                objsize = S.esetup_target_size(tid,3:4);
                % Position
                pos1 = S.esetup_st1_coord(tid,:);
                % Color
                fcolor1 = S.esetup_st1_color(tid,:)./255;
                % Shape
                if strcmp(S.esetup_target_shape{tid}, 'circle') || strcmp(S.esetup_target_shape{tid}, 'empty_circle')
                    v1 = 1;
                elseif strcmp(S.esetup_target_shape{tid}, 'square') || strcmp(S.esetup_target_shape{tid}, 'empty_square')
                    v1 = 0;
                end
                
                % Plot
                h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                    'EdgeColor', fcolor1, 'FaceColor', fcolor1, 'Curvature', v1, 'LineWidth', 1);
                
                %=============
                % Plot saccade detection window
                % Size
                objsize = S.esetup_target_size_eyetrack(tid,3:4) * 2; % Eyetracking windows are double size
                
                % Plot
                h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                    'EdgeColor', fcolor1, 'FaceColor', 'none', 'Curvature', 1, 'LineWidth', 1);
            end
            
            
            %==========
            % Plot distractor
            
            if ~isnan(S.esetup_st2_coord(tid,1))
                
                % Size
                objsize = S.esetup_target_size(tid,3:4);
                % Position
                pos1 = S.esetup_st2_coord(tid,:);
                % Color
                fcolor1 = S.esetup_st2_color(tid,:)./255;
                % Shape
                if strcmp(S.esetup_target_shape{tid}, 'circle') || strcmp(S.esetup_target_shape{tid}, 'empty_circle')
                    v1 = 1;
                elseif strcmp(S.esetup_target_shape{tid}, 'square') || strcmp(S.esetup_target_shape{tid}, 'empty_square')
                    v1 = 0;
                end
                
                % Plot
                h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                    'EdgeColor', fcolor1, 'FaceColor', fcolor1, 'Curvature', v1, 'LineWidth', 1);
                
            end
            
            
            %====================
            % Plot memory target in single probe trials (otherwise we know where memory is)
            if S.esetup_target_number(tid)==1
                
                % Size
                objsize = S.esetup_memory_size(tid,3:4);
                % Position
                pos1 = S.esetup_memory_coord(tid,:);
                % Color
                fcolor1=S.esetup_memory_color(tid,:)./255;
                if sum(fcolor1)>2.7 % For very bright colors reduce brightness
                    fcolor1 = [0.5, 0.5, 0.5];
                end
                
                if strcmp(S.esetup_memory_shape{tid}, 'empty_circle')
                    v1 = 1; c1 = [1,1,1];
                elseif strcmp(S.esetup_memory_shape{tid}, 'empty_square')
                    v1 = 0; c1 = [1,1,1];
                elseif strcmp(S.esetup_memory_shape{tid}, 'circle')
                    v1 = 1; c1 = fcolor1;
                elseif strcmp(S.esetup_memory_shape{tid}, 'square')
                    v1 = 0; c1 = fcolor1;
                end
                
                % Plot
                h=rectangle('Position', [pos1(1)-objsize(1)/2, pos1(2)-objsize(2)/2, objsize(1), objsize(2)],...
                    'EdgeColor', fcolor1, 'FaceColor', c1, 'Curvature', v1, 'LineWidth', 1);
            end
            
            
            %====================
            % Plot raw data (select only part of the time for it)
            t1 = S.first_display(tid);
            t2 = S.loop_over(tid);
            index1 = sx1(:,1)>=t1 & sx1(:,1)<=t2;
            fcolor1 = [0.5, 0.5, 0.5];
            if sum(index1)>1
                h=plot(sx1(index1,2), sx1(index1,3), 'Color', fcolor1, 'LineWidth', 1);
            end
            
            % Plot selected saccade
            index1=sx1(:,1)>=sacc1.saccade_matrix(tid,1) & sx1(:,1)<=sacc1.saccade_matrix(tid,2);
            if length(index1)>1
                x1=sx1(index1,2); y1=sx1(index1,3); % Convert to eye position in space
                h=plot(x1, y1, 'Color', [0.2, 0.8, 0.2], 'LineWidth', 1);
            end
            
            axis equal
            
            % Radius
            xc = S.esetup_st1_coord(tid, 1);
            yc = S.esetup_st1_coord(tid, 2);
            amp1 = sqrt(xc^2 + yc^2);
            
            if strncmp(sacc1.trial_accepted{tid}, 'aborted', 7)
                set(gca,'YTick',[-25, 0, 25]);
                set(gca,'XTick',[-25, 0, 25]);
                set(gca,'XLim',[-40,40]);
                set(gca,'YLim',[-40,40]);
            else
                set(gca,'YLim',[-12,12]);
                set(gca,'XLim',[-12,12]);
                set(gca,'YTick',[-round(amp1), 0, round(amp1)]);
                set(gca,'XTick',[-round(amp1), 0, round(amp1)]);
            end
            
            a = sprintf('%s, %s target(s)', var1.esetup_block_cond{tid}, num2str(var1.esetup_target_number(tid)));
            title (a, 'FontSize', settings.fontszlabel)
            
            %% Position in time
            
            %===========
            % Start figure
            hfig=subplot(1,3,[2:3]);
            hold on;
            
            %=====================
            % Plot fixation before eye movement acquired it
            
            if ~isnan(S.fixation_on(tid))
                yc=S.esetup_fixation_radius(tid);
                if S.esetup_fixation_drift_correction_on(tid)==1
                    s_window = S.esetup_fixation_size_drift(tid,4) * 2;
                else
                    s_window = S.esetup_fixation_size_eyetrack(tid,4) * 2;
                end
                t_start = S.fixation_on(tid) - S.first_display(tid);
                if ~isnan(S.fixation_acquired(tid))
                    t_dur = (S.fixation_acquired(tid) - S.fixation_on(tid));
                else
                    t_dur = (S.fixation_off(tid) - S.fixation_on(tid));
                end
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [0.9, 0.9, 0.9],  'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
            end
            
            
            %=====================
            % Plot fixation after eye movement acquired it (drift maintenance period)
            
            if ~isnan(S.fixation_acquired(tid))
                yc=S.esetup_fixation_radius(tid);
                if S.esetup_fixation_drift_correction_on(tid)==1
                    s_window = S.esetup_fixation_size_drift(tid,4) * 2;
                else
                    s_window = S.esetup_fixation_size_eyetrack(tid,4) * 2;
                end
                t_start = S.fixation_acquired(tid) - S.first_display(tid);
                if ~isnan(S.fixation_drift_maintained(tid))
                    t_dur = S.fixation_drift_maintained(tid) - S.fixation_acquired(tid);
                elseif isnan(S.fixation_drift_maintained(tid))
                    t_dur = S.fixation_off(tid) - S.fixation_acquired(tid);
                end
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [0.9, 0.9, 0.9],  'FaceColor', [0.9, 0.9, 0.9], 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -7, 'Fix', 'Color', [0.8, 0.8, 0.8],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            
            %===================
            % Plot fixation where eye is allowed before saccade target onset
            
            if ~isnan(S.fixation_drift_maintained(tid))
                yc=S.esetup_fixation_radius(tid);
                s_window = S.esetup_fixation_size_eyetrack(tid,4) * 2;
                t_start = S.fixation_drift_maintained(tid) - S.first_display(tid);
                if ~isnan(S.fixation_maintained(tid))
                    t_dur = S.target_on(tid) - S.fixation_drift_maintained(tid);
                elseif isnan(S.fixation_maintained(tid))
                    t_dur = S.fixation_off(tid) - S.fixation_drift_maintained(tid);
                end
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [0.7, 0.7, 0.7],  'FaceColor', [0.7, 0.7, 0.7], 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -7, 'Drif done', 'Color', [0.6, 0.6, 0.6],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            %==================
            % Message about drift correction size
            
            c1 = [0.6, 0.6, 0.6];
            if isfield(S, 'drift_output')
                a = S.predrift_xy_average(tid,:);
                dist = sqrt(a(1).^2 + a(2).^2);
                text(t_start, -9, sprintf ('%s: %s deg', S.drift_output{tid}, num2str(round(dist, 2))), 'Color', c1,  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left');
            else
                text(t_start, -9, sprintf ('Exp has no drift detection'), 'Color', c1,  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left');
            end
            %============
            % Plot saccade target 1 time window
            if ~isnan(S.target_on(tid)) && ~isnan(S.target_off(tid))
                
                xc = S.esetup_st1_coord(tid,1);
                yc = S.esetup_st1_coord(tid,2);
                yc = sqrt(xc.^2 + yc.^2); % Transform yc to distance
                
                s_window = S.esetup_target_size_eyetrack(tid,4) * 2;
                t_start = S.target_on(tid) - S.first_display(tid);
                t_dur = S.target_off(tid) - S.target_on(tid) ;
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [1, 0.5, 0.5], 'FaceColor', [1, 0.5, 0.5], 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -4, 'ST1', 'Color', [1, 0.5, 0.5],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            %============
            % Plot saccade target 2 time window
            if ~isnan(S.st2_on(tid))
                
                xc = S.esetup_st2_coord(tid,1);
                yc = S.esetup_st2_coord(tid,2);
                yc = sqrt(xc.^2 + yc.^2); % Transform yc to distance
                
                s_window = S.esetup_target_size_eyetrack(tid,4) * 2;
                t_start = S.st2_on(tid) - S.first_display(tid);
                if ~isnan(S.st2_off(tid))
                    t_dur = S.st2_off(tid) - S.st2_on(tid) ;
                else
                    t_dur = S.loop_over(tid) - S.st2_on(tid) ;
                end
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [0.7, 0.1, 0.1], 'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -7, 'ST2', 'Color', [0.7, 0.1, 0.1],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            %===================
            % Plot memory onset
            if ~isnan(S.memory_on(tid))
                xc=S.esetup_memory_coord(tid,1);
                yc=S.esetup_memory_coord(tid,2);
                yc = sqrt(xc.^2 + yc.^2); % Transform yc to distance
                s_window = S.esetup_target_size_eyetrack(tid,4) * 2;
                t_start=S.memory_on(tid)-S.first_display(tid);
                if ~isnan(S.memory_off(tid))
                    t_dur = S.memory_off(tid) - S.memory_on(tid);
                else
                    t_dur = S.esetup_memory_duration(tid)*1000;
                end
                h=rectangle('Position', [t_start, yc-s_window/2, t_dur, s_window],...
                    'EdgeColor', [0.4, 0.4, 1],  'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -2, 'M', 'Color', [0.4, 0.4, 1],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            %=================
            % Plot reward timing
            if ~isnan(S.reward_on(tid))
                yc = 0;
                s_window = 1;
                t_start = S.reward_on(tid)-S.first_display(tid);
                t_dur = S.edata_reward_size_ms(tid);
                h=rectangle('Position', [t_start, yc-s_window, t_dur, s_window],...
                    'EdgeColor', [0.7, 0.7, 0.1], 'FaceColor', 'none', 'Curvature', 0, 'LineWidth', 1);
                text(t_start, -3, 'Reward', 'Color', [0.7, 0.7, 0.1],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            end
            
            %==============
            % Plot raw eye position data
            t1=S.first_display(tid); % Relative to first display
            if length(sx1)>1
                index1=sx1(:,1)>=t1; % Select time
                sx1(:,1)=sx1(:,1)-t1; % Reset eye position timing relative to t1
                x1=sx1(index1,2); y1=sx1(index1,3); % Select position
                eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
                h=plot(sx1(index1,1), eyecoord1, 'Color', [0.1, 0.1, 0.1], 'LineWidth', 1); % Plot eye position in space and time
            end
            
            %=============
            % Add data of the blinks
            index1=sx1(:,4)==0; % Select blinks
            x1=sx1(index1,2); y1=sx1(index1,3); % Select positiion
            eyecoord1 = sqrt(x1.^2 + y1.^2); % Calculate amplitude of the eye position
            h=plot(sx1(index1,1), eyecoord1, '.y', 'LineWidth', 1);
            
            
            %============
            % Plot every single saccade detected
            t1 = S.first_display(tid); % Relative to first display
            if size(sx2,2)>1
                sx2(:,1)=sx2(:,1)-t1; % Reset eye position timing relative to t1
                sx2(:,2)=sx2(:,2)-t1; % Reset eye position timing relative to t1
                for i=1:size(sx2,1)
                    index1=sx1(:,1)>=sx2(i,1) & sx1(:,1)<=sx2(i,2);
                    x1=sx1(index1,2); y1=sx1(index1,3); % Convert to eye position in space
                    eyecoord1 = sqrt(x1.^2 + y1.^2); % Convert to position in space
                    
                    % Determine color of each saccade
                    if strcmp(s_class(i), 'correct')
                        color1 = [0.5, 0.5, 0.8];
                    elseif strcmp(s_class(i), 'correct target') || strcmp(s_class(i), 'wrong target')
                        color1 = [0.2, 0.8, 0.2];
                    elseif strcmp(s_class(i), 'no sorting started')
                        color1 = [1, 1, 0];
                    else % Every other saccade
                        color1 = [0.5, 0.5, 0.8];
                    end
                    
                    h=plot(sx1(index1,1), eyecoord1, 'Color', color1,  'LineWidth', 1);
                end
            end
            
            % X Lim
            set (gca,'FontSize', settings.fontsz);
            if S.reward_on(tid)>0
                t_start = ((S.first_display(tid) - S.first_display(tid))) - 100;
                t_end = ((S.reward_on(tid) - S.first_display(tid))) + 500;
                set(gca,'XLim',[t_start,t_end]);
            else
                t_start = ((S.first_display(tid) - S.first_display(tid))) - 100;
                t_end = ((S.fixation_off(tid) - S.first_display(tid))) + 500;
                set(gca,'XLim',[t_start,t_end]);
            end
            
            % Y tick label
            xc = S.esetup_st1_coord(tid,1);
            yc = S.esetup_st1_coord(tid,2);
            yc = sqrt(xc.^2 + yc.^2); % Transform yc to distance
            if round(yc,1)>0
                set(gca,'YTick',[0,round(yc,1)]);
            else
                set(gca,'YTick',[0, 1]);
            end
            if strncmp(sacc1.trial_accepted(tid), 'aborted', 7)
                set(gca,'YLim',[-12,40]);
            else
                set(gca,'YLim',[-12,15]);
            end
            
            % X tick labels
            if ~isnan(S.memory_on(tid)) && ~isnan(S.target_on(tid)) && ~isnan(S.target_off(tid))
                if S.target_on(tid)-S.memory_on(tid)>500
                    t1 = S.memory_on(tid) - S.first_display(tid);
                    t2 = S.target_on(tid) - S.first_display(tid);
                    set(gca,'XTick', [0, t1, t2])
                else
                    t1 = S.target_on(tid) - S.first_display(tid);
                    set(gca,'XTick', [0, t1])
                end
            elseif ~isnan(S.memory_on(tid)) && isnan(S.target_on(tid))
                t1 = S.memory_on(tid) - S.first_display(tid);
                t2 = S.fixation_off(tid) - S.first_display(tid);
                set(gca,'XTick', [0, t1, t2]);
            elseif ~isnan(S.fixation_off(tid)) && isnan(S.memory_on(tid))
                t1 = S.fixation_off(tid) - S.first_display(tid);
                set(gca,'XTick', [0, t1])
            end
            
            xlabel ('Time', 'FontSize', settings.fontszlabel)
            ylabel ('Gaze position', 'FontSize', settings.fontszlabel)
            
            
            % Add error description
            t_start = S.fixation_on(tid)-S.first_display(tid);
            text(t_start, 14, sacc1.trial_accepted(tid), 'Color', [0.1, 0.1, 0.1],  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
            
            
            %% Save the output
            
            
            %============
            % Export the figure & save it
            p_temp=sprintf('%s%s/', path_fig, sacc1.trial_accepted{tid});
            if ~isdir(p_temp)
                mkdir(p_temp)
            end
            
            % Save figure
            f_name = sprintf('%strial %s', p_temp, num2str(tid));
            set(gcf, 'PaperPositionMode', 'manual');
            set(gcf, 'PaperUnits', 'inches');
            set(gcf, 'PaperPosition', settings.fig_size_current)
            set(gcf, 'PaperSize', [settings.fig_size_current(3),settings.fig_size_current(4)]);
            print (f_name, '-dpdf')
            %===============
            
            close all;
            
            
            
        end
        % End of analysis for each trial
        
    end
    % End of analysis for each day
    
end
% End of analysis for each subject

