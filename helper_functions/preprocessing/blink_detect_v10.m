% Detecting and removing blinks. Blink data is replaced with average pre-blink and post-blink data
% DJ V1.0 : September 15, 2016. Use absolute median deviation to detect
% blink duration. Data is smoothed using wind_size variable.


function [t_pupil_new, t_pos_new] = blink_detect_v10 (t_pupil, t_pos, wind_size)


blinkplotting=0; % Plot blink detection for debugging purposes


%===========
% Find pupil size change over time
%===========

% Determine absolute difference between two consecutive timeseries points
% (Raw data is not used in order to compensate slow pupil size drift)
b = abs(diff(t_pupil));


% Find median deviation, to be used as threshold
b_threshold = median(abs(b-median(b)));

% Add a dummy in the beggining to make matrices of equal length
b_diff = [b(1); b];

% Calculate moving window average
if wind_size>0
    b_diff = conv(b_diff, ones(wind_size,1)/wind_size, 'same');
end



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
    elseif a(1)==1 && a(2)==size(mat1,1) % Blinks until recording end
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
        
        h=figure;
        hold on;
        
        % Plot eye position
        mat1_plot = mat1;
        mat1_plot = sqrt(mat1_plot(:,1).^2 + mat1_plot(:,2).^2); % Calculate amplitude of the eye position
        h=plot(mat1_plot, 'Color', [0.1, 1, 0.1], 'LineWidth', 1); % Plot eye position in space and time
        
        % Plot raw pupil position
        mat1_plot = t_pupil;
        h=plot(mat1_plot, 'Color', [0.2, 0.2, 0.2], 'LineWidth', 1); % Plot eye position in space and time

        % Plot pupil position
        mat1_plot = mat2;
        h=plot(mat1_plot, 'Color', [0.2, 0.2, 1], 'LineWidth', 1); % Plot eye position in space and time

        % Plot blink position
        mat1_plot = NaN(size(mat2));
        for i=1:size(blink_data_f, 1)
            a = blink_data_f(i,:);
            mat1_plot(a(1):a(2))=mat2(a(1):a(2));
            h=plot(mat1_plot, 'Color', [0.2, 0.2, 1], 'LineWidth', 2); % Plot eye position in space and time
        end
        
        % Plot change in pupil size
        mat1_plot=b_diff;
        h=plot(mat1_plot, 'Color', [0.8, 0.2, 0.2], 'LineWidth', 1);
        
        % Plot detected blink start and end
        mat1_plot = NaN(size(b_diff));
        for i=1:size(blink_data_f, 1)
            a = blink_data_f(i,:);
            mat1_plot([a(1),a(2)])=b_diff([a(1),a(2)]);
            h=plot(mat1_plot, 'Marker', 'o', 'Color', [0.2, 1, 0.2], 'LineWidth', 2, 'MarkerSize', 5); % Plot eye position in space and time
        end
        
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
