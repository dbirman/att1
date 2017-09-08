function y = removeNaN (x1)
% This function removes NaN values (all of them) from the matrix. If
% columns contain different number of NaN values, matrix will be reshaped
% into one row.
% MUST: only one sheet matrix is accepted (2 DIMENSIONS).

[m,n,o]=size(x1);
if o>1
    disp ('EMPTY MATRIX PRODUCED!');
    y=[];
    error ('Function does not work on multiple dimensions');
else
    IsnanFromMatrix = isnan(x1);
    NaNIndexes = find(IsnanFromMatrix == 1);
    x1(NaNIndexes) = [];
    y=x1;
end