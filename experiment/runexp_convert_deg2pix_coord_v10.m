%
% output = runexp_convert_deg2pix_coord(var1)
%
% var1: coordinates in degrees 
% output: coordinates in pixels relative to display center 
%
% var1 is 2 row matrix/vector: [x coordinate; y coordinate]; More than
% one column - more than one coordinate set
% One column - one set of coordinates.
%
% For function to work global variable expsetup.stim is necessary with
% fields: deg2pix and dispcenter
%
% v1.0 DJ: April 27, 2017



function s2 = runexp_convert_deg2pix_coord_v10(s1)

%% Settings

global expsetup;

if ~isfield (expsetup.screen, 'deg2pix') && ~isfield (expsetup.screen, 'dispcenter')
    error ('Global variable expsetup.stim with fields deg2pix and dispcenter does not exist')
else
    deg1 = expsetup.screen.deg2pix;
    cen1 = expsetup.screen.dispcenter;
end

%% Do conversion of coordinates

% Rotate matrix if needed
[m,n]=size(s1);
if (m<2 && n==2) || (m>2 && n==2)
    s1 = s1';
end

% Convert matrix to degrees
s2 = NaN(size(s1));
s2(1,:) = cen1(1) + (s1(1,:) .* deg1);
s2(2,:) = cen1(2) - (s1(2,:) .* deg1);

end