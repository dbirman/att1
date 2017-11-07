% Psychophysics performance

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

settings.figure_folder_name = 'training performance';
settings.figure_size_temp = settings.figsize_2col;
settings.stats_file_name = sprintf('statistics_%s_', settings.figure_folder_name);

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
        ind0 = date_current==settings.index_dates;
        folder_name = settings.index_directory{ind0};
        
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
        
        % Initialize text file for statistics
        nameOut = sprintf('%s%s.txt', path_fig, settings.stats_file_name); % File to be outputed
        fclose('all');
        fout = fopen(nameOut,'w');
        
        % Data folders
        path1 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '.mat'];
        path2 = [settings.path_data_combined_subject, folder_name, '/', folder_name, '_saccades.mat'];
        
        % Load all files
        S = get_struct_v10(path1);
        sacc1 = get_struct_v10(path2);
        
        %===============
        % Data analysis
        %===============
                
        % Determine exp versions used
        a=cell(1); a{1} = 'placeholder text'; % Dummy text, removed after
        for i=1:numel(S.training_stage_matrix)
            a = cat(2, a, S.training_stage_matrix{i});
        end
        a = unique(a);
        a (strcmp(a, 'placeholder text')) = [];
        conds1 = a;
        
        % Initialize matrices
        if i_date==1
            int_bins = 1:50:numel(S.session);
            test1 = NaN(numel(dates_used), length(int_bins)-1, numel(conds1), 3);
        end
        
        % Sliding window analysis
        for i = 1:size(test1,1)
            for j = 1:length(int_bins)-1
                for k=1:numel(conds1)
                    
                    ind = int_bins(j):1:int_bins(j+1);
                    
                    % Correct trials
                    index1 = strcmp(S.edata_error_code(ind), 'correct') & ...
                        strcmp(S.esetup_exp_version(ind), conds1{k});
                    test1(1,j,k,1)= sum(index1);
                    
                    % Wrong target trials
                    index1 = strcmp(S.edata_error_code(ind), 'looked at st2') & ...
                        strcmp(S.esetup_exp_version(ind), conds1{k});
                    test1(1,j,k,2)= sum(index1);
                    
                    % Aborted trials
                    index1 = strcmp(S.edata_error_code(ind), 'broke fixation') & ...
                        strcmp(S.esetup_exp_version(ind), conds1{k});
                    test1(1,j,k,3)= sum(index1);
                    
                end
            end
        end
        
    end
    % End of analysis for each day
end
% End of analysis for each subject

pbins = [];
for i = 1:numel(int_bins)-1
    pbins(i) = (int_bins(i)+int_bins(i+1))/2;
end


%%  Plot

num_figs = 2;

for fig1=1

    % Initialize data
    %=================
    figcolor1=[]; legend1={}; total1=[];
    mat1=[];
    save_name='Trial counts';

    % General motivation
    if fig1==1
        
        total1 = nansum(test1(:,:,:,1:2),4); % Check all conditions combined
        [~,~,~,p]=size(test1);
        mat1 = test1(:,:,:,1)./total1*100; 
        mat1 = 100 - (100 - mat1)*2; % Convert into target selection
        
        % Initialize structure
        plot_set = struct;
        plot_set.mat1 = mat1;
        plot_set.pbins = pbins;
        
        plot_set.data_color_min = [0.5,0.5,0.5];
        plot_set.data_color_max = settings.color1(42,:);
        
        for i=1:size(mat1,3)
           plot_set.legend{i} = conds1{i};
           plot_set.legend_y_coord(i) = i*-10;
           plot_set.legend_x_coord(i) = [pbins(1)];
        end

        % Labels for plotting
        plot_set.XTick = [];
        plot_set.x_plot_bins = pbins;
        plot_set.XLim = [pbins(1)-50, pbins(end)+50];
        plot_set.YTick = [0:25:100];
        plot_set.YLim = [min(plot_set.legend_y_coord)-10, 110];
        plot_set.figure_title = 'Performance';
        plot_set.xlabel = 'Trial number';
        plot_set.ylabel = 'Correct, % of trials';
        
        % Save data
        plot_set.figure_size = settings.figure_size_temp;
        plot_set.figure_save_name = 'training effects';
        plot_set.path_figure = path_fig;

    end

    %==================
    % Plot 
    
    hfig = subplot(1, num_figs, fig1);
    hold on;
    plot_helper_basic_line_figure;

end

%% Plot

for fig1=2

    % Initialize data
    %=================
    figcolor1=[]; legend1={}; total1=[];
    mat1=[];
    save_name='Trial counts';
    
    % General motivation
    if fig1==2
        
        total1 = nansum(test1(:,:,:,1:3),4); % Check all conditions combined
        [~,~,~,p]=size(test1);
        mat1 = test1(:,:,:,3)./total1*100; 
        
        % Initialize structure
        plot_set = struct;
        plot_set.mat1 = mat1;
        plot_set.pbins = pbins;
        
        plot_set.data_color_min = [0.5,0.5,0.5];
        plot_set.data_color_max = settings.color1(42,:);
        
        for i=1:size(mat1,3)
           plot_set.legend{i} = conds1{i};
           plot_set.legend_y_coord(i) = i*-10;
           plot_set.legend_x_coord(i) = [pbins(1)];
        end

        % Labels for plotting
        plot_set.XTick = [];
        plot_set.x_plot_bins = pbins;
        plot_set.XLim = [pbins(1)-50, pbins(end)+50];
        plot_set.YTick = [0:25:100];
        plot_set.YLim = [min(plot_set.legend_y_coord)-10, 110];
        plot_set.figure_title = 'Aborted trials';
        plot_set.xlabel = 'Trial number';
        plot_set.ylabel = '% of trials';
        
        % Save data
        plot_set.figure_size = settings.figure_size_temp;
        plot_set.figure_save_name = 'training effects';
        plot_set.path_figure = path_fig;

    end

    %==================
    % Plot 
    % Initialize figure
    hfig = subplot(1, num_figs, fig1);
    hold on;
    plot_helper_basic_line_figure;

end


%============
% Export the figure & save it

% Save data
plot_set.figure_size = settings.figure_size_temp;
plot_set.figure_save_name = 'training effects';
plot_set.path_figure = path_fig;

plot_helper_save_figure;
close all;
%===============
