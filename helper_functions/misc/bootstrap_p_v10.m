 % V 1.0 APril 18, 2016 - simple comparison of two conditions; Provide bootstrapped data for analysis 
 % V 1.0 update June 14, 2016 - Can use only one input and compare it against
 % zero


function [y1,y2,y3,y4,y5] = bootstrap_p_v10 (a,b)

if size(a,1)>1 && nargin==2 && size(a,2)==1 && size(a,3)==1
    diff1=a-b;
    tboot1 = size(a,1);
elseif size(a,1)>1 && nargin==1 && size(a,2)==1 && size(a,3)==1
    diff1=a;
    tboot1 = size(a,1);
else
    error ('Only 2d matrix accepted for bootstrap statistics')
end


% Find p-values
pval=NaN(1,size(diff1,2));

for i=1:length(pval)
    if mean (diff1(:,i))>0 && sum(diff1(:,i)>0)>0
        pval(i) = (sum(diff1(:,i)<0)/tboot1);
    elseif mean (diff1(:,i))>0 && sum(diff1(:,i)>0)==0
        pval(i) = (1/tboot1);
    elseif mean (diff1(:,i))<0 && sum(diff1(:,i)>0)>0
        pval(i) = (sum(diff1(:,i)>0)/tboot1);
    elseif mean (diff1(:,i))<0 && sum(diff1(:,i)>0)==0
        pval(i) = (1/tboot1);
    end
end

% Make a two sided t-test
pval=pval*2;

% Roun to 4th smallest decimal
pval(pval<0.0001)=0.0001;


% Bootstrapped mean 
y1 = nanmean(diff1);
y2 = std(diff1);
y3 = pval;
y4 = prctile(diff1,2.5);
y5 = prctile(diff1,97.5);

