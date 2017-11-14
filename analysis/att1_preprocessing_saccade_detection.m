% Detection of correct saccades
% Latest version: October 23, 2017
% Donatas Jonikaitis


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


%% Extra settings

settings.figure_folder_name = 'saccade_detection';
settings.figure_size_temp = [0, 0, 4.5, 2];
figure_on_temp = 0;

%% Analysis


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
        if ~isdir(path_fig) || settings.overwrite==1
            if ~isdir(path_fig)
                mkdir(path_fig)
            elseif isdir(path_fig)
                rmdir(path_fig, 's')
                mkdir(path_fig)
            end
        end
        
        % Data folders
        path1 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '.mat'];
        path2 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_eye_traces.mat'];
        path3 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_saccades.mat'];
        path4 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_individual_saccades.mat'];
        
        % Run analysis
        if exist(path1, 'file') && (~exist(path3, 'file') || settings.overwrite==1)
            
            
            %% Reshape saccades matrix
            
            % Load all settings
            var1 = get_struct_v10(path1);
            sacc1_raw = get_struct_v10(path2);
            
            % Initialize few variables
            sacc1 = var1.saccades_EK;
            saccade_matrix = NaN(size(sacc1,1), 7); % Only one saccade is taken
            trial_accepted = cell(size(sacc1,1), 1); % Initialize matrix which will track rejected saccades
            
            %============
            %============
            % Restructure saccades matrix into one saccade per row
            % This is for fast plotting of all saccades observed in the
            % experiments
            
            ST = struct; % Initalize output matrix
            temp_data = var1.saccades_EK; % Saccades data
            
            % Create a matrix with trial numbers (so that data could be re-accessed)
            mat1 = []; % Output matrix
            i = 1; % Row counts
            n = 1; % As many columns as there is data
            
            for tid = 1:numel(temp_data)
                
                [m,~] = size(temp_data{tid}); % One row - one saccade
                if m>0
                    mat1(i:i+m-1,1:n) = tid; % Save one row for one saccade
                elseif m==0
                    mat1(i,1:n) = tid; % Save one row for one saccade
                end
                i = size(mat1,1)+1; % Update number of rows for the next trial
                
            end
            
            ST.trial_no = mat1;
            
            %=================
            % Create matrix with all settings
            
            f1 = fieldnames(var1);
            
            for fid = 1:numel(f1) % For each data field
                if ~iscell(var1.(f1{fid}))
                    
                    mat1 = []; % Output matrix
                    i = 1; % Row counts
                    [~,n] = size(var1.(f1{fid})); % As manu columns as there is data
                    for tid = 1:numel(temp_data) % For each trial
                        [m,~] = size(temp_data{tid}); % One row - one saccade
                        if m>0
                            mat1(i:i+m-1,1:n) = repmat(var1.(f1{fid})(tid,:),m,1); % Save one row for one saccade
                        elseif m==0
                            mat1(i,1:n) = var1.(f1{fid})(tid,:); % Save one row for one saccade
                        end
                        i = size(mat1,1)+1; % Update number of rows for the next trial
                    end
                    
                    % Save output
                    ST.(f1{fid}) = mat1;
                    
                end
            end
            
            %=============
            % Restructure saccades matrix
            f1 = cell(1);
            f1{1} = 'sacc1';
            
            for fid = 1:numel(f1)
                
                %=========
                % Determine how many columns each trial data contains, as its
                % necessary for data concatenation
                temp1 = [];
                for tid=1:length(temp_data)
                    temp1(tid) = size(temp_data{tid},2);
                end
                n = max(temp1);
                
                %========
                mat1 = []; % Empty output matrix
                
                % For each trial extract data and concatenate with previous
                % trial
                for tid=1:numel(temp_data)
                    % If there is data
                    if ~isempty(temp_data{tid})
                        % Save each row as an event
                        for j=1:size(temp_data{tid},1) % One row - one event
                            m1=temp_data{tid}(j,:);
                            m=size(mat1,1);
                            mat1(m+1,1:n)=m1;
                        end
                        % Else save as NaN
                    elseif isempty(temp_data{tid})
                        % Add an extra line with empty matrix
                        m=size(mat1,1);
                        mat1(m+1,1:n)=NaN;
                    end
                end
                % Save output
                ST.(f1{fid})=mat1;
            end
            
            clear temp_data;
            
            %============
            %  Create matrix to track why saccades were accepted/rejected
            
            ST.sacc_classify = cell(numel(ST.START), 1);
            ST.sacc_classify (1:end) = {'no sorting started'};
            
            
            %%  Calculate some extra variables for data evaluation
            
            
            %=============
            % Saccade onset distance from the fixation
            
            sx1 = ST.sacc1;
            
            x1=sx1(:,3);
            y1=sx1(:,4);
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_start_fix_dist = l1;
            
            %=============
            % Saccade endpoint distance from the fixation
            
            sx1 = ST.sacc1;
            
            x1=sx1(:,5);
            y1=sx1(:,6);
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_end_fix_dist = l1;
            
            %=============
            % Calculate saccade amplitude
            
            sx1 = ST.sacc1;
            
            x1=sx1(:,5) - sx1(:,3);
            y1=sx1(:,6) - sx1(:,4);
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_amp = l1;
            
            
            %================
            % Calculate saccade start distance from st1
            
            sx1 = ST.sacc1;
            
            xc = ST.esetup_st1_coord(:,1);
            yc = ST.esetup_st1_coord(:,2);
            x1=sx1(:,3)-xc;
            y1=sx1(:,4)-yc;
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_start_st1_dist = l1;
            
            %================
            % Calculate saccade end distance from st1
            
            sx1 = ST.sacc1;
            
            xc = ST.esetup_st1_coord(:,1);
            yc = ST.esetup_st1_coord(:,2);
            x1=sx1(:,5)-xc;
            y1=sx1(:,6)-yc;
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_end_st1_dist = l1;
            
            %================
            % Calculate saccade end distance from st2
            
            sx1 = ST.sacc1;
            
            xc = ST.esetup_st2_coord(:,1);
            yc = ST.esetup_st2_coord(:,2);
            x1=sx1(:,5)-xc;
            y1=sx1(:,6)-yc;
            l1=sqrt((x1.^2)+(y1.^2));
            ST.sacc_end_st2_dist = l1;
            
            %===============
            % Calculate saccade angle from st1
            
            sx1 = ST.sacc1;
            temp1 = [];
            
            % Calculate st1 angle
            a=atan2(ST.esetup_st1_coord(:,2), ST.esetup_st1_coord(:,1)); % Angle of saccade endpoint
            index = a>=0; % Convert PI angle into degrees
            a(index)=180*(a(index)/pi);
            index = a<0; % Convert PI angle into degrees
            a(index)=360+(180*a(index)/pi);
            ST.st1_angle = a;
            
            % Rotate st1 angle?
            var11=90; % Goal of rotation
            displace1 = var11 - ST.st1_angle;
            
            % Calculate resetting angles
            displace1(displace1>=360)=displace1(displace1>=360)-360;
            displace1(displace1<0)=displace1(displace1<0)+360;
            
            % Start position rotation
            angle = displace1; x=sx1(:,3); y=sx1(:,4);
            xn = cosd(angle).*x - sind(angle).*y;
            yn = sind(angle).*x + cosd(angle).*y;
            temp1(:,3)=xn; temp1(:,4)=yn;
            
            % End position rotation
            angle=displace1; x=sx1(:,5); y=sx1(:,6);
            xn = cosd(angle).*x - sind(angle).*y;
            yn = sind(angle).*x + cosd(angle).*y;
            temp1(:,5)=xn; temp1(:,6)=yn;
            
            % Calculate the angle of the rotated data endpoints
            a=atan2(temp1(:,6), temp1(:,5)); % Angle of saccade endpoint
            index = a>=0; % Convert PI angle into degrees
            a(index)=180*(a(index)/pi);
            index = a<0; % Convert PI angle into degrees
            a(index)=360+(180*a(index)/pi);
            temp1(:,7) = a; % Save as column number 7
            
            % Rotated output is saved into ST.rotated_t1
            ST.sacc1_rotated_to_st1 = temp1;
            
            
            %% remove outlier saccades
            
            index = isnan(ST.sacc1(:,1));
            ST.sacc_classify(index) =  {'reject - no data'};
            
            % Saccades before/after trial
            temp1 = ST.first_display;
            temp2 = ST.loop_over;
            index = (ST.sacc1(:,1) < temp1) | (ST.sacc1(:,1) > temp2) ;
            ST.sacc_classify(index) =  {'reject - saccade timing'};
            
            % Distance outliers
            ind = strcmp(ST.sacc_classify, 'no sorting started');
            v1 = ST.sacc_start_fix_dist(ind);
            a_m = nanmean(v1); a_s = nanstd(v1);
            a_sdev_th = 3;
            
            index = ST.sacc_end_fix_dist > a_m + a_s*a_sdev_th; % 3 standard deviations
            ST.sacc_classify(index) =  {'reject - outlier amplitudes'};
            
            
            %% Figure: outlier saccades
            
            if figure_on_temp==1
                % Plot all data
                h_fig = subplot(1,2,1); hold on
                
                % Plot
                ind = strcmp(ST.sacc_classify, 'no sorting started');
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1, [0.2, 0.2, 0.2]);
                ind = strcmp(ST.sacc_classify, 'reject - outlier amplitudes');
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1, [1, 0.5, 0.5]);
                
                % Labels for plotting
                h_fig = gca;
                h_fig.XTick = [-25, 0, 25];
                h_fig.YTick = [-25, 0, 25];
                h_fig.XLim = [-50, 50];
                h_fig.YLim = [-50, 50];
                title ('All data', 'FontSize', settings.fontszlabel)
                xlabel ('Horizontal', 'FontSize', settings.fontszlabel);
                ylabel ('Vertical', 'FontSize', settings.fontszlabel);
                
                %=================
                % Plot zoomed in data
                h_fig = subplot(1,2,2); hold on
                
                % Plot
                ind = strcmp(ST.sacc_classify, 'no sorting started');
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1, [0.2, 0.2, 0.2]);
                ind = strcmp(ST.sacc_classify, 'reject - outlier amplitudes');
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1, [1, 0.5, 0.5]);
                
                % Labels for plotting
                h_fig = gca;
                h_fig.XTick = [-10, 0, 10];
                h_fig.YTick = [-10, 0, 10];
                h_fig.XLim = [-15, 15];
                h_fig.YLim = [-15, 15];
                title ('Zoom in', 'FontSize', settings.fontszlabel)
                xlabel ('Horizontal', 'FontSize', settings.fontszlabel);
                ylabel ('Vertical', 'FontSize', settings.fontszlabel);
                
                % Save data
                plot_set.figure_size = settings.figure_size_temp;
                plot_set.figure_save_name = 'sacc';
                plot_set.path_figure = path_fig;
                plot_helper_save_figure;
                
            end
            %% aborted - fixation not acquired
            
            % sacc endpoint threshold
            th1 = NaN(numel(ST.START), 1);
            ind = ST.esetup_fixation_drift_correction_on == 1;
            th1(ind) = ST.esetup_fixation_size_drift(ind,4);
            ind = ST.esetup_fixation_drift_correction_on == 0;
            th1(ind) = ST.esetup_fixation_size_eyetrack(ind,4);
            
            %=============
            % Saccade does not end in fixation area
            ind = isnan(ST.fixation_acquired) & ~isnan(ST.fixation_on) & ~isnan(ST.fixation_off) & ...
                ST.sacc1(:,1) >= ST.fixation_on & ST.sacc1(:,1) <= ST.fixation_off & ST.sacc_end_fix_dist>th1;
            
            trial_select_code = 'aborted - fixation not acquired';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                if isnan(var1.fixation_acquired(tid))
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %=============
            % OK saccades before fixation acquired
            ind = isnan(ST.fixation_acquired) & ~isnan(ST.fixation_on) & ~isnan(ST.fixation_off) & ...
                ST.sacc1(:,1) >= ST.fixation_on & ST.sacc1(:,1) <= ST.fixation_off & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            %% aborted before drift maintained
            
            % sacc endpoint threshold
            th1 = NaN(numel(ST.START), 1);
            ind = ST.esetup_fixation_drift_correction_on == 1;
            th1(ind) = ST.esetup_fixation_size_drift(ind,4);
            ind = ST.esetup_fixation_drift_correction_on == 0;
            th1(ind) = ST.esetup_fixation_size_eyetrack(ind,4);
            
            %=============
            % Too large saccades before drift maintained
            ind = ~isnan(ST.fixation_acquired) & isnan(ST.fixation_drift_maintained) & ...
                ~isnan(ST.fixation_on) & ~isnan(ST.fixation_off) & ...
                ST.sacc1(:,1) >= ST.fixation_acquired & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist>=th1;
            
            trial_select_code = 'aborted - broke fixation before drift';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %=============
            % OK saccades before drift maintained
            ind = ~isnan(ST.fixation_acquired) & isnan(ST.fixation_drift_maintained) & ...
                ~isnan(ST.fixation_on) & ~isnan(ST.fixation_off) & ...
                ST.sacc1(:,1) >= ST.fixation_acquired & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            
            %% aborted: broke fixation before memory
            
            % sacc endpoint threshold
            th1 = NaN(numel(ST.START), 1);
            th1 = ST.esetup_fixation_size_eyetrack(:,4);
            
            %=============
            % Broke fixation before memory (memory didnt appear)
            ind = ~isnan(ST.fixation_drift_maintained) & isnan(ST.memory_on) & ...
                ST.sacc1(:,1) >= ST.fixation_drift_maintained & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist>=th1;
            
            trial_select_code = 'aborted - broke fixation before memory';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %===============
            % Correct saccades before memory target (memory didnt appear)
            ind = ~isnan(ST.fixation_drift_maintained) & isnan(ST.memory_on) & ...
                ST.sacc1(:,1) >= ST.fixation_drift_maintained & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            %=============
            % Broke fixation before memory (memory appeared)
            ind = ~isnan(ST.fixation_drift_maintained) & ~isnan(ST.memory_on) & ...
                ST.sacc1(:,1) >= ST.fixation_drift_maintained & ST.sacc1(:,1) <= ST.memory_on &...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist>=th1;
            
            trial_select_code = 'aborted - broke fixation before memory';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %===============
            % Correct saccades before memory target (memory appeared)
            ind = ~isnan(ST.fixation_drift_maintained) & ~isnan(ST.memory_on) & ...
                ST.sacc1(:,1) >= ST.fixation_drift_maintained & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            %% aborted: broke fixation during memory delay
            
            % sacc endpoint threshold
            th1 = NaN(numel(ST.START), 1);
            th1 = ST.esetup_fixation_size_eyetrack(:,4);
            
            %=============
            % Broke fixation before st1 (st1 didnt appear)
            ind = ~isnan(ST.memory_on) & isnan(ST.target_on) & ...
                ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.fixation_off & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist>=th1;
            
            trial_select_code = 'broke fixation during memory';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %===============
            % Correct saccades before st1 (st1 didnt appear)
            ind = ~isnan(ST.memory_on) & isnan(ST.target_on) & ...
                ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.target_on & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            %=============
            % Broke fixation before st1 (st appeared)
            ind = ~isnan(ST.memory_on) & ~isnan(ST.target_on) & ...
                ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.target_on & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist>=th1;
            
            trial_select_code = 'broke fixation during memory';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            %==============
            % Correct saccades before st1 (st1 appeared)
            ind = ~isnan(ST.memory_on) & isnan(ST.target_on) & ...
                ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.target_on & ...
                ST.sacc_start_fix_dist<=th1 & ST.sacc_end_fix_dist<=th1;
            
            trial_select_code = 'correct';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            %%  Plot memory delay saccades
            
            if figure_on_temp == 1
                %=================
                % Plot correct/error saccades during memory delay
                close all
                
                plot_set.var1_group_name{1} = 'Correct';
                plot_set.var1_group_name{2} = 'Error';
                plot_set.color1(1,:) = [0.2, 0.2, 0.2];
                plot_set.color1(2,:) = [1, 0.5, 0.5];
                
                h_fig = subplot(1,2,1); hold on
                
                % Plot correct trials
                ind = ~isnan(ST.memory_on) & ~isnan(ST.target_on) & ...
                    ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.target_on;
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1,  plot_set.color1(1,:));
                
                % Plot error trials
                ind = ~isnan(ST.memory_on) & isnan(ST.target_on) & ...
                    ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.fixation_off &...
                    ~strcmp (ST.sacc_classify, 'reject - outlier amplitudes');
                hfig = scatter(ST.sacc1(ind,5), ST.sacc1(ind,6), 1,  plot_set.color1(2,:));
                
                
                % Labels for plotting
                h_fig = gca;
                h_fig.XTick = [-10, 0, 10];
                h_fig.YTick = [-10, 0, 10];
                h_fig.XLim = [-15, 15];
                h_fig.YLim = [-15, 15];
                title ('Memory delay saccades', 'FontSize', settings.fontszlabel)
                xlabel ('Horizontal', 'FontSize', settings.fontszlabel);
                ylabel ('Vertical', 'FontSize', settings.fontszlabel);
                
                %==============
                h_fig = subplot(1,2,2); hold on
                
                
                mat1=[]; plot_bins=[];
                
                % Plot correct trials
                ind = ~isnan(ST.memory_on) & ~isnan(ST.target_on) & ...
                    ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.target_on;
                [mat1(:,:,1), plot_bins(:,:,1)] = hist(ST.sacc_amp(ind), 50);
                
                % Plot rejected trials
                ind = ~isnan(ST.memory_on) & isnan(ST.target_on) & ...
                    ST.sacc1(:,1) >= ST.memory_on & ST.sacc1(:,1) <= ST.fixation_off &...
                    ~strcmp (ST.sacc_classify, 'reject - outlier amplitudes');
                [mat1(:,:,2), plot_bins(:,:,2)] = hist(ST.sacc_amp(ind), 50);
                
                
                % Normalize data
                for i=1:size(mat1,3)
                    mat1(:,:,i)=mat1(:,:,i)/max(mat1(:,:,i));
                end
                
                % Plot
                for i=1:size(mat1,3)
                    h=plot(plot_bins(:,:,i), mat1(:,:,i));
                    graphcond=i;
                    set (h(end), 'LineWidth', settings.wlinegraph, 'Color', plot_set.color1(graphcond,:))
                end
                
                
                %==============
                % Legend
                
                % Determine y position for the legend
                for k=1:size(mat1,3)
                    if size(mat1,1)>1
                        t_max(k)=max(nanmean(plot_bins(:,:,k)));
                    else
                        t_max(k)=max(plot_bins(:,:,k));
                    end
                end
                ax_max = max(t_max);
                
                % Set coordinates for legend
                legend1 = plot_set.var1_group_name;
                legend_pos_x = repmat(ax_max, numel(legend1), 1);
                legend_pos_y = [1, 0.85];
                
                % Plot legend text
                for i=1:length(legend1)
                    graphcond = i;
                    text(legend_pos_x(i), legend_pos_y(i), legend1{i}, 'Color', plot_set.color1(graphcond,:),  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'right')
                end
                
                
                % Labels for plotting
                h_fig = gca;
                h_fig.XTick = [0: 10 : ax_max];
                h_fig.YTick = [0, 0.5, 1];
                h_fig.XLim = [-0.5, ax_max+ax_max*0.05];
                h_fig.YLim = [-0.01, 1.1];
                title ('Saccade amp', 'FontSize', settings.fontszlabel)
                xlabel ('Saccade amplitude', 'FontSize', settings.fontszlabel);
                ylabel ('Saccade frequency', 'FontSize', settings.fontszlabel);
                
                
                % Save data
                plot_set.figure_size = settings.figure_size_temp;
                plot_set.figure_save_name = 'memory_delay';
                plot_set.path_figure = path_fig;
                plot_helper_save_figure;
                
            end
            %% detect target saccades
            
            % sacc endpoint threshold
            th0 = NaN(numel(ST.START), 1);
            th1 = NaN(numel(ST.START), 1);
            th0 = ST.esetup_fixation_size_eyetrack(:,4);
            th1 = ST.esetup_target_size_eyetrack(:,4);
            
            %=================
            %=================
            % Correct saccades, dual target trials
            ind = strcmp(ST.sacc_classify, 'no sorting started') & ...
                ~isnan(ST.target_on) & ~isnan(ST.st2_on) & ...
                ST.sacc1(:,1) >= ST.target_on & ST.sacc1(:,1) <= ST.target_off & ...
                ST.sacc_start_fix_dist <= th0 & ST.sacc_end_st1_dist <= th1 & ...
                ST.sacc_end_st1_dist<=ST.sacc_end_st2_dist;
            
            trial_select_code = 'correct target';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                    temp_data = ST.sacc1(ind,:);
                    temp_ind = find (strcmp (ST.sacc_classify(ind), trial_select_code)==1);
                    saccade_matrix(tid,:) = temp_data(temp_ind(1),:);
                end
            end
            
            %=================
            %=================
            % Correct saccades, probe trials
            ind = strcmp(ST.sacc_classify, 'no sorting started') & ...
                ~isnan(ST.target_on) & isnan(ST.st2_on) & ...
                ST.sacc1(:,1) >= ST.target_on & ST.sacc1(:,1) <= ST.target_off & ...
                ST.sacc_start_fix_dist <= th0 & ST.sacc_end_st1_dist <= th1;
            
            trial_select_code = 'temp1';
            ST.sacc_classify(ind) = {trial_select_code};
            
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                    temp_data = ST.sacc1(ind,:);
                    temp_ind = find (strcmp (ST.sacc_classify(ind), trial_select_code)==1);
                    saccade_matrix(tid,:) = temp_data(temp_ind(1),:);
                end
            end
            
            % Over-write selection iwth older message
            trial_select_code = 'temp1';
            trial_replace_code = 'correct target';
            ind = strcmp(ST.sacc_classify, trial_select_code);
            ST.sacc_classify(ind) = {trial_replace_code};
            ind = strcmp(trial_accepted, trial_select_code);
            trial_accepted(ind) = {trial_replace_code};
            
            %=================
            %==================
            % Saccades to wrong target
            ind = strcmp(ST.sacc_classify, 'no sorting started') & ...
                ~isnan(ST.target_on) & ~isnan(ST.st2_on) &...
                ST.sacc1(:,1) >= ST.target_on & ST.sacc1(:,1) <= ST.target_off & ...
                ST.sacc_start_fix_dist <= th0 & ST.sacc_end_st2_dist <= th1 & ...
                ST.sacc_end_st2_dist<=ST.sacc_end_st1_dist;
            
            trial_select_code = 'wrong target';
            ST.sacc_classify(ind) = {trial_select_code};
            
            % Save trial accepted
            for tid = 1:numel(var1.START)
                ind = ST.trial_no==tid;
                if sum(strcmp (ST.sacc_classify(ind), trial_select_code))>0
                    trial_accepted{tid} = trial_select_code;
                    temp_data = ST.sacc1(ind,:);
                    temp_ind = find (strcmp (ST.sacc_classify(ind), trial_select_code)==1);
                    saccade_matrix(tid,:) = temp_data(temp_ind(1),:);
                end
            end
            
            
            %% No response saccade detected
            
            trial_select_code = 'no response saccade detected';
            
            for tid = 1:numel(var1.START)
                sx1 = var1.saccades_EK{tid};
                if ~isempty(sx1)
                    a1 = ST.sacc_amp(ST.trial_no==tid);
                    ind = sx1(:,1)>var1.target_on(tid) & sx1(:,1)<var1.loop_over(tid) & a1>1;
                    if sum(ind)==0 && ~isnan(var1.target_on(tid))
                        trial_accepted{tid} = trial_select_code;
                    end
                else
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            
            %% Blink during the trial
            
            trial_select_code = 'blink during the trial';
            
            for tid = 1:numel(var1.START)
                sx1 = sacc1_raw.eye_processed{tid};
                if ~isempty(sx1)
                    ind = sx1(:,1)>var1.fixation_drift_maintained(tid) & sx1(:,1)<var1.target_off(tid) & sx1(:,4)==0;
                    if sum(ind)>0
                        trial_accepted{tid} = trial_select_code;
                    end
                end
            end
            
            
            %% Experimenter terminated the trial
            
            trial_select_code = 'experimenter terminated the trial';
            
            for tid=1:size(var1.START,1)
                if strcmp(var1.edata_error_code{tid}, trial_select_code)
                    trial_accepted{tid} = trial_select_code;
                end
            end
            
            
            %% Unknown errors
            
            
            for tid=1:size(var1.START,1)
                if isempty(trial_accepted{tid})
                    trial_accepted{tid} = 'unknown error';
                end
            end
            
            
            %% Save errors into text file
            
            % Initialize empty file
            f_name = sprintf('%s%s.txt', path_fig, folder_name);
            fclose('all');
            fout = fopen(f_name,'w');
            
            % Print out errors
            targettext='\nTrials accepted and removed: \n';
            fprintf(targettext);
            fprintf(fout, targettext);
            
            a=numel(trial_accepted); % a-total number of trials
            targettext='Total trials tested: %d; \n\n';
            fprintf(targettext, a);
            fprintf(fout, targettext, a);
            
            e1 = unique(trial_accepted);
            for i_data=1:numel(e1)
                c = sum(strcmp(trial_accepted, e1{i_data}));
                targettext='%s: %d trials (%d percent) \n';
                fprintf(targettext, e1{i_data}, c, round((c/a)*100));
                fprintf(fout, targettext, e1{i_data}, c, round((c/a)*100));
            end
            
            
            %% Save sacc1
            
            sacc1 = struct;
            sacc1.saccade_matrix = saccade_matrix;
            sacc1.trial_accepted = trial_accepted;
            save (eval('path3'), 'sacc1')
            targettext='Saved saccades data: %s; \n\n';
            fprintf(targettext, path3);
            
            
            %% Save ST structure
            
            sacc2 = struct;
            sacc2.trial_no = ST.trial_no;
            sacc2.sacc_classify = ST.sacc_classify;
            sacc2.sacc1 = ST.sacc1;
            save (eval('path4'), 'sacc2')
            
        elseif ~exist(path3, 'file')
            fprintf('\nInput folder %s does not exist, skipping saccade detection\n', folder_name)
        elseif exist(path1, 'file')
            fprintf('\nFolder name %s already exists, skipping saccade detection\n', folder_name)
        end
        % End of analysis
        
    end
    % End of analysis for each day
    
end
% End of analysis for each subject

