% Detecting and removing blinks. Blink data is replaced with average pre-blink and post-blink data
% DJ V1.0 : September 15, 2016. Use absolute median deviation to detect
% blink duration. Data is smoothed using wind_size variable.
% DJ V1.1 : February 23, 2017. Corrected 2 bugs:
% 1: b_diff now is calculated before b_threshold
% 2: previously if blink ended at trial end, data was not extrapolated


function [t_pupil_new, t_pos_new] = blink_detect_v11 (t_pupil, t_pos, wind_size)


blinkplotting=0; % Plot blink detection for debugging purposes


%===========
% Find pupil size change over time
%===========

% Determine absolute difference between two consecutive timeseries points
% (Raw data is not used in order to compensate slow pupil size drift)
b = abs(diff(t_pupil));

% Add a dummy in the beggining to make matrices of equal length 
b_diff = [b(1); b]; clear b;

% Calculate moving window average
if wind_size>0
    b_diff = conv(b_diff, ones(wind_size,1)/wind_size, 'same');
end

% Find median deviation, to be used as threshold
b_threshold = median(abs(b_diff-median(b_diff)));
% b_threshold = median(b_diff);



%===========
% Find how many blinks are recorded
%===========

idx = find(t_pupil==0); % 0 codes for blinks detected by the eyelink

% Blink start
ind1 = diff([-1; idx]); % Add dummy in the beggining (to extract first blink start)
i1 = find(ind1~=1);
% Blink end
ind1 = diff([idx; -1]); % Add dummy in the end (to extract last blink end)
i2 = find(ind1~=1);
% Blink index
blink_data = [idx(i1) idx(i2)];


%=============
% Use sliding window to determine whether blink is over
%=============

% New blink start and end matrix;
blink_data_f = NaN(size(blink_data)); 


% Sliding window for each blink onset
for i=1:size(blink_data,1)
    a = blink_data(i,1); % Start of blink
    n1 = 0;
    ind = a-1;
    while n1 == 0
        if n1==0 & ind>1
            if b_diff(ind)>=b_threshold
                ind=ind-1;
            elseif b_diff(ind)<b_threshold
                blink_data_f(i,1)=ind;
                n1=1;
            end
        elseif n1==0 & ind<=1
            blink_data_f(i,1)=1;
            n1=1;
        end
    end
end


% Sliding window for each blink offset
for i=1:size(blink_data,1)
    a = blink_data(i,2); % End of blink
    n1 = 0;
    ind = a+1; % First sample after a blink has ended
    while n1 == 0
        if n1==0 & ind<length(b_diff)
            if b_diff(ind)>=b_threshold
                ind=ind+1;
            elseif b_diff(ind)<b_threshold
                blink_data_f(i,2)=ind;
                n1=1;
            end
        elseif n1==0 & ind>=length(b_diff);
            blink_data_f(i,2)=length(b_diff);
            n1=1;
        end
    end
end


%=============
% Fill in the data
%=============

% Copy the data instead of over-writing it
mat1 = t_pos; % Eye position
mat2 = t_pupil; % Pupil size

for i=1:size(blink_data_f,1) % For each blink
    a = blink_data_f(i,:); % Start of blink
    dur1 = a(2)-a(1)+1;
    if a(1)>1 && a(2)<size(mat1,1) % Regular blinks
        mat1(a(1):a(2), 1) = linspace(mat1(a(1),1), mat1(a(2),1),  dur1);
        mat1(a(1):a(2), 2) = linspace(mat1(a(1),2), mat1(a(2),2), dur1);
        mat2(a(1):a(2), 1) = linspace(mat2(a(1),1), mat2(a(2),1), dur1);
    elseif a(1)==1 && a(2)<size(mat1,1) % Blinks from recording start
        mat1(a(1):a(2), 1) = mat1(a(2),1);
        mat1(a(1):a(2), 2) = mat1(a(2),2);
        mat2(a(1):a(2), 1) = mat2(a(2),1);
    elseif a(1)>=1 && a(2)==size(mat1,1) % Blinks until recording end
        mat1(a(1):a(2), 1) = mat1(a(1),1);
        mat1(a(1):a(2), 2) = mat1(a(1),2);
        mat2(a(1):a(2), 1) = mat2(a(1),1);
    end
end

   

%=============
% Plot
%=============

% Checking blink detection algorithm accuracy
if blinkplotting==1
    
    close all
    hfig = figure;
    set(hfig, 'units', 'normalized', 'position', [0.2, 0.4, 0.6, 0.3]);
    
    %=============
    % PANEL 1
    
    h = subplot(1,3,[1]);
    hold on;
    
    % Plot raw pupil position
    color1 = [0.2, 0.2, 0.2];
    mat1_plot = t_pupil;
    h=plot(mat1_plot, 'Color', color1, 'LineWidth', 1); % Plot eye position in space and time
    % Legend
    y1 = max(mat1_plot) + max(mat1_plot)*0.1;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'Raw pupil', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')
    % Fig settings
    set(gca,'YLim',[min(mat1_plot) - max(mat1_plot)*0.1, max(mat1_plot) + max(mat1_plot)*0.2]);
    set(gca,'XLim',[0 - length(mat1_plot)*0.1, length(mat1_plot) + length(mat1_plot)*0.2]);

    % Plot corrected pupil position
    color1 = [0.2, 0.2, 1];
    mat1_plot = mat2;
    mat1_plot = mat1_plot + mat1_plot*0.01;
    h=plot(mat1_plot, 'Color', color1, 'LineWidth', 1); % Plot eye position in space and time
    % Legend
    y1 = max(mat1_plot) + max(mat1_plot)*0.15;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'Processed pupil', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')
    
    % Plot blink data
    color1 = [1, 0.2, 0.2];
    
    for i = 1:size(blink_data_f,1)
        mat1_plot = mat2(blink_data_f(i,1):blink_data_f(i,2));
        x1 = mat1_plot + mat1_plot*0.01;
        t1 = blink_data_f(i,1):blink_data_f(i,2);
        h=plot(t1, x1, '.', 'Color', color1, 'LineWidth', 1); % Plot eye position in space and time
    end


        
    %============
    % Panel 2
    
    h = subplot(1,3,[2]);
    hold on;
    
    % Plot pupil position difference
    color1 = [0.8, 0.2, 0.2];
    mat1_plot = b_diff;
    h=plot(mat1_plot, 'Color', color1, 'LineWidth', 1); % Plot eye position in space and time
    y1 = -1;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'Pupil abs. diff.', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')
    % Fig settings
    set(gca,'YLim',[-3, 10]);
    set(gca,'XLim',[0 - length(mat1_plot)*0.1, length(mat1_plot) + length(mat1_plot)*0.1]);

    % Plot threshold used for the task
    color1 = [0.2, 0.2, 0.9];
    mat1_plot = b_diff;
    b_threshold;
    plot ([0, length(mat1_plot)], [b_threshold, b_threshold], 'Color', color1, 'LineWidth', 2);
    y1 = -2;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'median abs. diff. threshold', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')

    %==========
    % Panel 3
    
    h = subplot(1,3,[3]);
    hold on;
    
    % Plot eye position
    color1 = [0.1, 1, 0.1];
    mat1_plot = t_pos;
    mat1_plot = sqrt(mat1_plot(:,1).^2 + mat1_plot(:,2).^2); % Calculate amplitude of the eye position
    h=plot(mat1_plot, 'Color', color1 , 'LineWidth', 1); % Plot eye position in space and time
    % Legend
    y1 = max(mat1_plot) + max(mat1_plot)*0.1;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'Raw eye position', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')
    % Fig settings
    set(gca,'YLim',[min(mat1_plot) - max(mat1_plot)*0.1, max(mat1_plot) + max(mat1_plot)*0.2]);
    set(gca,'XLim',[0 - length(mat1_plot)*0.1, length(mat1_plot) + length(mat1_plot)*0.2]);

    % Plot corrected eye position
    color1 = [0.2, 0.2, 1];
    mat1_plot = mat1;
    mat1_plot = sqrt(mat1_plot(:,1).^2 + mat1_plot(:,2).^2); % Calculate amplitude of the eye position
    mat1_plot = mat1_plot + mat1_plot*0.01;
    h=plot(mat1_plot, 'Color', color1, 'LineWidth', 1); % Plot eye position in space and time
    % Legend
    y1 = max(mat1_plot) + max(mat1_plot)*0.15;
    x1 = length(mat1_plot)*0.1;
    text(x1, y1, 'Processed eye position', 'Color', color1,  'FontSize', 10, 'HorizontalAlignment', 'left')

    
    %================
    % Collect response to the trial
    disp (' ')
    aaa=['Press "Enter" to accept, "x" to quit '];
    an_trial=input(aaa, 's');
    disp(' ')
    if strcmp (an_trial,'x')
        return
        % Code will crash if you stop plotting
    end
    close all;
        
end

t_pos_new = mat1;
t_pupil_new = mat2;
