% Get particular fields a "struct.field" or "struct.struct.field"
% V1.0 August 30, 2016
%
% var1 is structure from which to get the fields
% var_names is a cell array with fields, such as "field" or "structure.field"
% output is "struct.field"

function y = get_fields_v10 (var1, var_names)


% Get requested fieldnames and save them into structe var_out
var_out = struct;
for i=1:length(var_names)
    a = var_names{i};
    b = strsplit(a, '.');  % check whether it is a structure
    if length(b)==1
        var_out.(b{1}) = var1.(b{1});
    end
    if length(b)==2
        var_out.(b{2}) = var1.(b{1}).(b{2});
    end
end


y = var_out;