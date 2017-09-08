function bsData = bootstrap(data,ni)
%
% function bsData = bootstrap(data,ni)
% 
% bootstrap data by sampling with replacement
% from the rows in data. ni specifies the 
% number of bootstrap samples. 
%
% 2006 by Martin Rolfs


if nargin<2
    ni = 100;
end

nc = size(data,1);    % number of cases

for i = 1:ni
    bsSample = randsample(1:nc,nc,1);    
    bsData(i,:) = mean(data(bsSample,:));
end
