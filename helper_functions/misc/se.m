function [y1] = se (x1);
% Standard error

% Remove NaN values
x1(isnan(x1))=[];

% Find se
[m,n]=size(x1);
if m>n
    x1;
else
    x1=x1';
end
a1=sqrt(m);
a2=std(x1);

y1=a2/a1;