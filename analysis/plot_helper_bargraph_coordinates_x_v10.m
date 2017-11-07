% Calculate the number of bars to be plotted
% Provide matrix, bar_width and space_width (between experiments)

function pbins = plot_helper_bargraph_coordinates_x_v10(plot_set)

%===========
% Initialize variables
if isfield(plot_set, 'mat1')
    mat1 = plot_set.mat1;
else
    error ('plot_set.mat1 does not exist')
end

if isfield (plot_set, 'bar_width')
    bar_width = plot_set.bar_width;
else
    bar_width = 0.05;
end

if isfield (plot_set, 'space_width')
    space_width = plot_set.space_width;
else
    space_width = bar_width*0.3;
end

%==============
% Calculate number of bars
% Add one bar between experiments
b=[];
for i = 1:size(mat1,3)
    t1 = size(mat1,2);
    if i==1
        b(1:t1) = 1;
    else
        m = numel(b);
        b(m+1) = 0; % Space between experiments
        m = numel(b);
        b(m+1:m+t1) = 1;
    end
end

%=============
% Bar positions
m = numel(b);
range_bar = [bar_width*m + (m-1)*space_width]; % Bars plus spaces between them take that much space in total

range_bar = range_bar/2; % Position to both sides of the unit
xcoord = [1-range_bar:bar_width+space_width:1+range_bar];
xcoord(b==0)=[];

pbins = xcoord;
