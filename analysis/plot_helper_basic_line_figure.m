% Plots lines and sets up figure settings 

% Calculate colors if it is a range
if isfield (plot_set, 'data_color_min')
    
    % Settings
    col_min = plot_set.data_color_min;
    col_max = plot_set.data_color_max;
    n = size(mat1,3);
    
    % Other location colors are calculated as a range
    if n>1
        d1 = col_max-col_min;
        stepsz = 1/(n-1);
        for i=1:n
            color1_range(i,:)=col_min + (d1*stepsz)*(i-1);
        end
    else
        color1_range(1,:)=col_min;
    end
    
end

 
% Plot lines
mat1 = plot_set.mat1;
pbins = plot_set.pbins;

for k=1:size(mat1,3)
    
    % Draw line
    if size(mat1,1)>1
        h=plot(pbins, nanmean(mat1(1,:,k),1));
    elseif size(mat1,1)==1
        h=plot(pbins, mat1(1,:,k));
    end
    
    % Select color
    if isfield(plot_set, 'data_color_min')
        c1 = color1_range(k,:);
    elseif isfield(plot_set, 'data_color') && ~isempty(plot_set.data_color)
        graphcond = plot_set.data_color(k);
        c1 = settings.color1(graphcond,:);
    else
        error ('Figure colors not specified')
    end
    
    set (h(end), 'LineWidth', settings.wlinegraph, 'Color', c1)
end

%% Legend


% Plot legend text
if isfield (plot_set, 'legend')
    for k=1:numel(plot_set.legend)
        
        legend1 = plot_set.legend{k};
        y1 = plot_set.legend_y_coord(k);
        x1 = plot_set.legend_x_coord(k);
        
        % Select color
        if isfield(plot_set, 'data_color_min')
            c1 = color1_range(k,:);
        elseif isfield(plot_set, 'data_color') && ~isempty(plot_set.data_color)
            graphcond = plot_set.data_color(k);
            c1 = settings.color1(graphcond,:);
        else
            error ('Figure colors not specified')
        end

        text(x1, y1, legend1, 'Color', c1,  'FontSize', settings.fontszlabel, 'HorizontalAlignment', 'left')
    end
end

hfig = gca;
set (hfig, 'FontSize', settings.fontsz);

% Y Tick
if isfield (plot_set, 'YTick')
    hfig.YTick = plot_set.YTick;
end

% X Tick
if isfield (plot_set, 'XTick') && ~isempty(plot_set.XTick)
    hfig.XTick = plot_set.XTick;
elseif isfield (plot_set, 'x_plot_bins')
    pbins = plot_set.x_plot_bins;
    if pbins(end)<=10
        hfig.XTick = [1,5,10];
    elseif pbins(end)<=20
        hfig.XTick = [1,5:5:pbins(end)];
    elseif pbins(end)<=50
        hfig.XTick = [1,10:10:pbins(end)];
    elseif pbins(end)<=100
        hfig.XTick = [1,20:20:pbins(end)];
    elseif pbins(end)<=250
        hfig.XTick = [1,50:50:pbins(end)];
    elseif pbins(end)<=500
        hfig.XTick = [1,100:100:pbins(end)];
    elseif pbins(end)<=1000
        hfig.XTick = [1,250:250:pbins(end)];
    elseif pbins(end)<=2500
        hfig.XTick = [1,500:500:pbins(end)];
    elseif pbins(end)<=5000
        hfig.XTick = [1,1000:1000:pbins(end)];
    end
    
end

% Y Lim
if isfield (plot_set, 'YLim')
    hfig.YLim = plot_set.YLim;
end

% X Lim
if isfield (plot_set, 'XLim')
    hfig.XLim = plot_set.XLim;
end

% X label
if isfield (plot_set, 'xlabel')
    xlabel (plot_set.xlabel, 'FontSize', settings.fontszlabel);
end

% Y label
if isfield (plot_set, 'ylabel')
    ylabel (plot_set.ylabel, 'FontSize', settings.fontszlabel);
end

% Figure title
if isfield (plot_set, 'figure_title')
    title (plot_set.figure_title, 'FontSize', settings.fontszlabel)
end
