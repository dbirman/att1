% Function checks whehter eye is on specified target
%
% input x1 measured x pos
% input y1 measured y pos
% input x1_target vector with x coordinates of potential targets
% input y1_target vector with y coordinates of potential targets
% input error1 max distance from the target allowed
%
% outpu1 target number where the eye fell (for example, 2 - 2nd target on x1_target vector)
%
% v1.0 DJ: February 17, 2015
% v1.1 DJ: July 10, 2015  Selects the target with minimum distance. This
% was a bug in v1.0


function tsel1 = runexp_check_eye_target_v11 (x1, y1, x1_target, y1_target, error1)

% Find difference between desired and observed target
coord_x = x1_target - x1;
coord_y = y1_target - y1;

% Distance from eye to target
eye_dist1 = sqrt((coord_x.^2)+(coord_y.^2));
eye_error1 = eye_dist1 <= error1;

% Record which target was selected
tsel1=0;
if sum(eye_error1) > 0
    a = find (eye_dist1 == min(eye_dist1));
    tsel1 = a;
end

% Check whether no target was selected
if sum(eye_error1) == 0
    tsel1 = 0;
end

end