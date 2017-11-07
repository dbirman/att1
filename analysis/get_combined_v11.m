% Load data from 
% path1/folder_name/file_name
% And combine multiple files into one big file
% V1.0 September 20, 2016
% V1.1 October 30, 2016. Corrects error where some fields were not combined
%
% Inputs
% path1 - path to the subject folder, with all dates inside it
% folder_index - list with every folder name inside path1
% file_index - list of files to be combined (number of names is same as folder_index)
% comp_name - string used as reference for trial numbers, for example 'session' or 'trial_accepted' 
%


function y = get_combined_v11 (path1, folder_index, file_index, comp_name)

S_comb = struct; % Initialize empty output structure


for i_date = 1:length(folder_index) % (Can not deal a column, has to be a row vector)
    
    % Which folder to load?
    folder_name = folder_index{i_date};
    file_name = file_index{i_date};
    path_in = sprintf('%s%s/', path1, folder_name);
    
    % Load daily data
    var1 = load([path_in, file_name]);
    f1 = fieldnames(var1);
    if length(f1)==1
        var1 = var1.(f1{1});
    end
    S_new = var1; % Rename
    
    % Add day count
    fn1 = fieldnames(S_comb);
    if isempty(fn1)
        S_new.day = ones(size(S_new.(comp_name)));
    else
        S_new.day = ones(size(S_new.(comp_name))) + max(S_comb.day);
    end
    
    % Concatenate data. Uses S.session for reference of missing fields
    f_n_old = fieldnames(S_comb);
    f_n_new = fieldnames(S_new);
    if isempty(fn1) % On first repetition
        S_comb = S_new;
    else % On other repetitions
        
        %============
        % Read out field names of the old structure, and add them to new
        % struture if missing
        fn1 = f_n_old;
        for i=1:length(fn1)
            if isfield(S_new, fn1{i})
                S_comb.(fn1{i})=cat(1, S_comb.(fn1{i}), S_new.(fn1{i}));
            elseif ~isfield(S_new, fn1{i})
                if ismatrix(S_comb.(fn1{i})) % If it is a matrix
                    if iscell (S_comb.(fn1{i}))
                        S_new.(fn1{i})=cell(size(S_new.(comp_name),1), size(S_comb.(fn1{i}),2));
                        S_comb.(fn1{i})=cat(1, S_comb.(fn1{i}), S_new.(fn1{i}));
                    else
                        S_new.(fn1{i})=NaN(size(S_new.(comp_name),1), size(S_comb.(fn1{i}),2));
                        S_comb.(fn1{i})=cat(1, S_comb.(fn1{i}), S_new.(fn1{i}));
                    end
                else
                    error ('DJ deal with this at section 1')
                end
            end
        end
        
        %============
        % Read out field names of the new structure, and add them to old
        % struture if missing
        fn1 = f_n_new;
        for i=1:length(fn1)
            if isfield(S_comb, fn1{i})
                % Do nothing, as combining is already done
            elseif ~isfield(S_comb, fn1{i})
                if ismatrix(S_new.(fn1{i})) % If it is a matrix
                    if iscell (S_new.(fn1{i}))
                        S_comb.(fn1{i})=cell(size(S_comb.(comp_name),1), size(S_new.(fn1{i}),2));
                        i1 = size(S_comb.(fn1{i}), 1) - size(S_new.(fn1{i}), 1) + 1;
                        i2 = size(S_new.(fn1{i}), 2);
                        S_comb.(fn1{i})(i1:end, 1:i2)=S_new.(fn1{i});
                    else
                        S_comb.(fn1{i})=NaN(size(S_comb.(comp_name),1), size(S_new.(fn1{i}),2));
                        i1 = size(S_comb.(fn1{i}), 1) - size(S_new.(fn1{i}), 1) + 1;
                        i2 = size(S_new.(fn1{i}), 2);
                        S_comb.(fn1{i})(i1:end, 1:i2)=S_new.(fn1{i});
                    end
                else
                    error ('DJ deal with this at section 2')
                end
            end
        end
        
        
    end
    
end

y = S_comb;