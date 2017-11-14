
% Example code
overwrite_temp_index{1} = 20170101:20170102;

%% Example code

if settings.overwrite_temp_switch == 1 && date_current <= overwrite_temp_index{1}(1) && date_current <= overwrite_temp_index{1}(end)
    
%     v1 = 'fixation_off';
%     v2 = 'target_on';
%         
%     if isfield(var1.eyelink_events, v1) % check whether to do analysis
%         
%         fprintf('Correcting messages fixation_off\n')
%         
%         clear temp1_old; clear temp1_new;
%         temp1_old = var1.eyelink_events.(v1);
%         temp1_new = var1.eyelink_events.(v2);
%         
%         % If ST1 appeared, that means fixation disappeared simultaneously
%         % with it. If ST1 didnt appear, then fixation_off time is end of
%         % the trial loop.
%         ind = ~isnan(temp1_new);
%         temp1_old(ind) = temp1_new(ind);
%         
%         % Save corrected data
%         var1.eyelink_events.(v1) = temp1_old;
%     else
%         fprintf('Field fixation_off already corrected, no changes written\n')
%     end
    
end