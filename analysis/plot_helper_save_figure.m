% Save figure

if isfield (plot_set, 'path_figure') && ...
        isfield (plot_set, 'figure_save_name') && ...
        isfield (plot_set, 'path_figure')
    if isdir(plot_set.path_figure)
        f_name = sprintf('%s%s', plot_set.path_figure , plot_set.figure_save_name);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperPosition', plot_set.figure_size)
        set(gcf, 'PaperSize', [plot_set.figure_size(3),plot_set.figure_size(4)]);
        print (f_name, '-dpdf')
    else
        error ('Figure path not defined, can not plot data')
    end
else
    error ('Not all fields defined for saving data')
end