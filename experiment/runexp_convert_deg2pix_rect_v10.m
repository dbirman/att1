%
% output = runexp_convert_deg2pix_rect(var1, var2)
%
% var1: coordinates in degrees 
% var2: rectangle size in degrees
% output: rectangle in pixels relative to display center 
%
% var1 is 2 row matrix/vector: [x coordinate; y coordinate]; More than
% one column - more than one coordinate set
% One column - one set of coordinates.
% var2 is 4 row matrix/vector: [0; 0; size x; size y]; More than one
% column - more than one coordinate set
%
% For function to work global variable expsetup.stim is necessary with
% fields: deg2pix and dispcenter
% Necessary psychtoolbox function CenterRectOnPoint
%
% v1.0 DJ: April 27, 2017


function d1_rect = runexp_convert_deg2pix_rect_v10(s1, r1)

%% Settings

global expsetup;

if ~isfield (expsetup.screen, 'deg2pix') && ~isfield (expsetup.screen, 'dispcenter')
    error ('Global variable expsetup.stim with fields deg2pix and dispcenter does not exist')
else
    deg1 = expsetup.screen.deg2pix;
    cen1 = expsetup.screen.dispcenter;
end

%% Do conversion of coordinates

% Rotate coordinate matrix if needed
% One column - one set of coordinates (psychtoolbox requirement)
[m,n] = size(s1);
if (m<2 && n==2) || (m>2 && n==2)
    s1 = s1';
end

% Convert matrix to degrees and center on display center
s2 = NaN(size(s1));
s2(1,:) = cen1(1) + (s1(1,:) .* deg1);
s2(2,:) = cen1(2) - (s1(2,:) .* deg1);

% Rotate size matrix if needed;
% One row - one set of coordinates (psychtoolbox requirement)
[m,n] = size(r1);
if (m==4 && n<4) || (m==4 && n>4)
    r1 = r1';
end
% Convert matrix to degrees
r1 = r1.*deg1;

% Prepare rectangle of centered coordinates
d1_rect = NaN(4,size(s2,2));
for i=1:size(d1_rect,2)
    if numel(r1)==4
        d1_rect(:,i) = CenterRectOnPoint(r1, s2(1,i), s2(2,i));
    else
        d1_rect(:,i) = CenterRectOnPoint(r1(i,:), s2(1,i), s2(2,i));
    end
end
    

end