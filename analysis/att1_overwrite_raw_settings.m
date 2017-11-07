%% Specify conditions to be modified 

% Example code
overwrite_temp_index{1} = 20170101:20170102;


%% Example code
if settings.overwrite_temp_switch == 1 && date_current <= overwrite_temp_index{1}(1) && date_current <= overwrite_temp_index{1}(end)
    
%     v1 = 'esetup_block_cond';
%     
%     clear temp1_old; clear temp1_new
%     temp1_old = var1.stim.(v1);
%     
%     if ~iscell(temp1_old) % Check whether to do analysis
%         
%         fprintf('Correcting field esetup_block_cond\n')
%         
%         temp_new = cell(numel(temp1_old),1);
%         
%         index = temp1_old==1;
%         temp_new(index) = {'look'};
%         index = temp1_old==2;
%         temp_new(index)= {'avoid'};
%         index = temp1_old==3;
%         temp_new(index)= {'control'};
%         
%         % Save corrected data
%         var1.stim.(v1) = temp_new;
%     else
%         fprintf('Structure esetup_block_cond already exists, no changes written\n')
%     end
    
end

