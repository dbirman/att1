% ONLINE data figure

trials_to_plot = 25; % Window size of the moving average

if tid<trials_to_plot
    
    set(gca,'FontSize', fontsz);
    text(0, 0, ['Min. trials needed - ', num2str(trials_to_plot)],...
        'Color', [0.1,0.1,0.1], 'FontSize', fontsz, 'HorizontalAlignment', 'center')
    set(gca,'YLim',[-5,5]);
    set(gca,'XLim', [-19, 19]);
    set(gca,'YTick', [-10,10]);
    set(gca,'XTick', [-20,20]);
    
elseif tid>=trials_to_plot % Plot data
    
    intervalbins = [trials_to_plot:1:tid];
    temp1=NaN(4,length(intervalbins));
    
    % Calculate time course of errors
    for j=1:length(intervalbins)
        
        ind = intervalbins(j)-trials_to_plot+1:intervalbins(j);
        
        % Correct trials
        index1 = expsetup.stim.expmatrix(ind,em_data_reject)==1;
        temp1(1,j)= sum(index1);
        % Aborted trials
        index1 = expsetup.stim.expmatrix(ind,em_data_reject)==2 | ...
            expsetup.stim.expmatrix(ind,em_data_reject)==3;
        temp1(2,j)= sum(index1);
        % Fixation break (likely looked at memory)
        index1 = expsetup.stim.expmatrix(ind,em_data_reject)==4;
        temp1(3,j)= sum(index1);
        % Saccade response errors
        index1 = expsetup.stim.expmatrix(ind,em_data_reject)==5 | ...
            expsetup.stim.expmatrix(ind,em_data_reject)==6 | ...
            expsetup.stim.expmatrix(ind,em_data_reject)==7;
        temp1(4,j)= sum(index1);
        
    end
    
    temp1=temp1./trials_to_plot*100;
    
    % Calculate plotbins
    plot_bins = (intervalbins-intervalbins(end)-1).*-1;
    plot_bins = log((plot_bins)); % Calculate to log
    plot_bins = -1*plot_bins; % Revert the plotting direction (most recent trials to the right)
    
    % Plot each error as a line
    for i=1:4
        % Colors and legend
        if i==1
            graphcond=1;
            text(-log(1), 138, 'Correct', 'Color', color1(graphcond,:), 'FontSize', fontsz, 'HorizontalAlignment', 'right')
        elseif i==2
            graphcond=2;
            text(-log(1), 128, 'Aborted', 'Color', color1(graphcond,:), 'FontSize', fontsz, 'HorizontalAlignment', 'right')
        elseif i==3
            graphcond=4;
            text(-log(1), 118, 'Fix break', 'Color', color1(graphcond,:), 'FontSize', fontsz, 'HorizontalAlignment', 'right')
        elseif i==4
            graphcond=5;
            text(-log(1), 108, 'Saccade errors', 'Color', color1(graphcond,:), 'FontSize', fontsz, 'HorizontalAlignment', 'right')
        end
        % Lines
        h = plot(plot_bins(:), temp1(i,:));
        set (h(end), 'LineWidth', wlinegraph, 'Color', color1(graphcond,:))
    end
        
    % Figure settings
    set(gca,'FontSize', fontsz);
    set(gca,'YLim',[0,155])
    ylabel ('Trials (%)', 'FontSize', fontszlabel);
    set(gca,'YTick',25:25:100);
    
    xlabel ('Trials', 'FontSize', fontszlabel);
    if tid<100
        a=[50,25,10,1];
    elseif tid<500
        a=[250,100,50,10,1];
    elseif tid<1000
        a=[500,100,10,1];
    else
        a=[1000,500,100,10,1];
    end
    set(gca,'XTick', -log(a));
    set(gca,'XTickLabel', -1*a, 'FontSize', fontszlabel)
    set(gca,'XLim', [-log(max((intervalbins-intervalbins(end)-2).*-2)), -log(0.5)])
    
end
