 % V 1.0 APril 18, 2016 - simple Cohen d calculation; uses average of 2 std    

function [y1] = effect_size_v10 (a,b)

if size(a,1)>1 && size(a,3)==1 && size(a,2)==1
    y1 = (mean(a)-mean(b))/((std(a)+std(b))/2);
    y1 = abs(y1);
else
    error ('Only 1 column vectors are accepted for effect size calculation')
end