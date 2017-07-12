% ONLINE data figure

min_trials_to_plot = 50; % How many trials to use for evaluation of errors/corrects

if tid<=min_trials_to_plot
    var1 = expsetup.stim.expmatrix(1:tid, em_data_reject);
elseif tid>min_trials_to_plot
    var1 = expsetup.stim.expmatrix(1:tid, em_data_reject);
end

% Bar positions
bnum = 8; % How many bars
barwdh = 0.05; spacewidth=0.3;
rngbar = [barwdh*bnum+barwdh*(bnum-1)*spacewidth]; % Bars plus spaces between them take that much space in total
rngbar = rngbar/2; % Position to both sides of the unit
plotbins = [1-rngbar:barwdh+barwdh*spacewidth:1+rngbar];

% Plot each bar
for i = 1:bnum
    
    % Plot bars
    if tid<=min_trials_to_plot
        h=bar(plotbins(i), sum(var1==i), barwdh); % Plot all trials
    else
        h=bar(plotbins(i), sum(var1==i)/tid*100, barwdh); % Plot proportion
    end
    graphcond=i;
    set (h(end), 'LineWidth', wlinegraph, 'EdgeColor', color1(graphcond,:), 'FaceColor', color1(graphcond,:));
    
    % Legend, plotted on each bar
    if i==1
        text(plotbins(i), 3, 'Correct', 'Color', [1,0.6,0.6], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==2
        text(plotbins(i), 3, 'No fix', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==3
        text(plotbins(i), 3, 'Drift min missing', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==4
        text(plotbins(i), 3, 'Fix broken', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==5
        text(plotbins(i), 3, 'Wrong T', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==6
        text(plotbins(i), 3, 'No sacc', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==7
        text(plotbins(i), 3, 'Left T', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    elseif i==8
        text(plotbins(i), 3, 'Unknown', 'Color', [0.2,0.2,0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'left', 'Rotation', 90)
    end
end

% Figure settings
set(gca,'FontSize', fontsz);
if tid<=min_trials_to_plot/2
    set(gca,'YLim',[0,min_trials_to_plot/2+min_trials_to_plot*0.1]);
    ylabel ('Trials', 'FontSize', fontszlabel);
    set(gca,'YTick',0:5:min_trials_to_plot);
elseif tid > min_trials_to_plot/2 && tid<=min_trials_to_plot
    set(gca,'YLim',[0,tid+min_trials_to_plot*0.1]);
    ylabel ('Trials', 'FontSize', fontszlabel);
    set(gca,'YTick',0:10:min_trials_to_plot);
elseif tid>min_trials_to_plot
    set(gca,'YLim',[0,120]) % Max percent
    ylabel ('Trials (%)', 'FontSize', fontszlabel);
    set(gca,'YTick',25:25:100);
end
set(gca,'XLim',[plotbins(1)-0.1 plotbins(end)+0.1]);
xlabel ('Trial type', 'FontSize', fontszlabel);
set(gca,'XTickLabel', ' ', 'FontSize', fontszlabel)

% Add total amount of reward and trials correct
if tid>min_trials_to_plot
    if expsetup.general.reward_on>0
        reward_given = nansum(expsetup.stim.expmatrix(:, em_data_reward_size_ml));
        a =  sprintf ('\nTotal reward: %i ml\n', round(reward_given));
        text(plotbins(end),110, a, 'Color', [0.2, 0.2, 0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'right');
    end
    index1 = expsetup.stim.expmatrix(:, em_data_reject) == 1;
    a = sprintf ('\nCorrect trials: %i (%i percent) \n', sum(index1), round((sum(index1)/tid)*100));
    text(plotbins(end), 100, a, 'Color', [0.2, 0.2, 0.2], 'FontSize', fontsz, 'HorizontalAlignment', 'right');
end



