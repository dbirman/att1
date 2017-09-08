% Match plexon messages with psychtoolbox timing
% V1.0 September 29, 2016. Very basic version. Barely works
% V1.1 Novemver 1, 2016. Now checks for event timing for each session
% separately. 


function y = preprocessing_match_plexon_events_v11(psy1, sp1, ses1)

% Some default settings
no_t = 10; % How many trials to use for finding first session; This is minimum recorded trial number
ms_dur = 20; % Allowed error in plexon event timing in mili-seconds
evt1 = NaN(length(ses1), 1); % Initialize empty timing matrix


% Loop for each session separatelly
for s1 = 1:max(ses1)
    
    
    evt0 = NaN(sum(ses1==s1), 1); % Initialize empty matrix for saving plexon events in current session
    diff_sp1 = diff(sp1); % Take all plexon events
    diff_psy1 = diff(psy1(ses1==s1)); % Only subset of psychtoolbox events (current session)
    ind_psych = 1; % Start with first psychtoolbox trial and update this index within loop
    
    
    %% Find current session onset timing
    
    % Cycle through the loop until match is found or none of the trials was
    % found
    loop_over = 0; % Cycle untill you find psych recordings matching plexon recordings in a given session
    while loop_over==0 && (ind_psych+no_t-1) <= length(diff_psy1)
        
        temp_psy = diff_psy1(ind_psych:ind_psych+no_t-1); % Select subset of psych trials
        c2 = NaN(size(diff_sp1)); % Clear out the variable
        
        % Compare selected temp_psy trials to every trial in plexon events
        for tid = 1:length(diff_sp1)
            if tid+no_t-1 <= length(diff_sp1)
                % Select plexon events
                temp_sp = diff_sp1(tid:tid+no_t-1);
                % Find mean difference between event repeats
                c2(tid) = mean(abs(temp_sp-temp_psy));
            else
                c2(tid) = NaN;
            end
        end
        
        % Check whether threshold of accuracy is achieved
        ind_plex = find(c2<ms_dur);
        
        if length(ind_plex)==0 % If failed to find match
            ind_psych=ind_psych+1; % Update index to check further set of trials
        elseif length(ind_plex)>1 % If found too many matching trials
            error ('DJ, unanticipated multiple trials satisfy threshold criterion found')
        elseif length(ind_plex)==1
            loop_over = 1;
        end
    end
    % End of trying to find psych trials to match plex events
    
    
    
    %% Check for match between psych and plex for each trial within session
    
    if ~isempty(ind_plex) % If match with plexon was found
        
        for tid = ind_psych:length(diff_psy1) % For each recorded trial in current psych session
            
            % If current psych trial does not match spikes recording, there are
            % two potential reasons:
            % 1 - plexon failed to report event
            % 2 - plexon reported event which did not happen
                
            %==============
            % If match was found
            if abs(diff_psy1(tid)-diff_sp1(ind_plex)) <= ms_dur % If inter-trial difference is withing error limits, save both trials constituting that difference
                evt0(tid:tid+1) = sp1(ind_plex:ind_plex+1); % Save data from plexon events
                ind_plex = ind_plex+1; % Remove first plex evt timing
                                
                
                %===============
                % Deal with option 1 by advancing to next psychtoolbox
                % trial, without updating ind_plex
            elseif diff_psy1(tid)-diff_sp1(ind_plex) < -ms_dur
                % It will upate psych trial index automatically
                
                
                %================
                % Deal with option 2 - skip current plexon event until
                % difference is lower than threshold
            elseif diff_psy1(tid)-diff_sp1(ind_plex) > ms_dur
                
                while (diff_psy1(tid)-diff_sp1(ind_plex) > ms_dur) && (ind_plex <= length(diff_sp1))
                    ind_plex = ind_plex+1;
                end
                if abs(diff_psy1(tid)-diff_sp1(ind_plex)) <= ms_dur % If inter-trial difference is withing error limits, save both trials constituting that difference
                    evt0(tid:tid+1) = sp1(ind_plex:ind_plex+1); % Save data from plexon events
                    ind_plex = ind_plex+1; % Remove first plex evt timing
                end
            end
        end
        % End of analysis for each recorded trial
        
        % Save evt0 matrix into evt1
        evt1(ses1==s1)=evt0;
    end
    
    
end

y = evt1;
