% Loads a structure and renames it

function y = get_struct_v11(path1)


if exist (path1, 'file')
    var1 = struct;
    varx = load(path1);
    f1 = fieldnames(varx);
    if length(f1)==1
        var1 = varx.(f1{1});
    end
else
    var1 = struct;
end

y = var1;