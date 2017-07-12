function grating = GenerateGrating(m,n,angle,lambda,ph,con)

% grating = GenerateGrating(256, 256, pi/2, 16, pi/2, 1);
% m/n = x/y patch size in pixels; angle = orientation of grating in degrees; 
% lambda = spatial period in pixels; p=phase (0 to 1); con=contrast;
%
% J Greenwood 2009 & D Jonikaitis 2011

[X,Y] = meshgrid(-m/2:m/2-1,-n/2:n/2-1); % Grid size
angle=-angle+90; % Make grating vertical (instead of horizontal)
theta=angle*pi/180; % Orientation in radians
phase=ph*2*pi; % Phase converted to radians

% rotate co-ordinates
Xt2 = X.*(cos(pi/2-theta)) + Y.*(sin(pi/2-theta));
grating = (0.5*con)*cos(Xt2.*((2.0*pi)/lambda)+phase); % use 0.5*contrast to set max and min values around zero
