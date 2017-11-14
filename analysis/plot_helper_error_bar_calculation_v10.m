% Calculate error bars

function e_bars = plot_helper_error_bar_calculation_v10(mat1, settings)


a1=[]; b1=[]; pct1=[]; lw1=[]; up1=[];
% Bootstrap the sample

if size(mat1,1)>1
    for k=1:size(mat1,3)
        if k==1
            a1 = [mat1(:,:,k)];
        else
            a1 = [a1, mat1(:,:,k)];
        end
    end
    b1 = bootstrapnan(a1,settings.tboot1);
    pct1 = prctile(b1,[2.5,97.5]);
    for k=1:size(mat1,3)
        i1=1+(size(mat1,2)*k)-size(mat1,2);
        i2=(size(mat1,2)*k);
        lw1(:,:,k) = pct1(1,i1:i2,:); % Lower bound (2.5 percentile)
        up1(:,:,k) = pct1(2,i1:i2,:); % Upper bound (97.5 percentile)
        bt1(:,:,k) = b1(:,i1:i2,:); % Restructure original matrix for bootstrap stats
    end
    bt1_bootstrap = bt1;
    lw1_bootstrap = lw1;
    up1_bootstrap = up1;
end

% SEM
lw1=[]; up1=[];
if size(mat1,1)>1
    for k=1:size(mat1,3)
        for i=1:size(mat1,2)
            lw1(:,i,k) = nanmean(mat1(:,i,k))-se(mat1(:,i,k)); % Standard error, lower bound (identical to upper one)
            up1(:,i,k) = nanmean(mat1(:,i,k))+se(mat1(:,i,k)); % Standard error, upper bound (identical to lower one)
        end
    end
end

if size(mat1,1)>1
    e_bars.bootstrap_upper = up1_bootstrap;
    e_bars.bootstrap_lower = lw1_bootstrap;
    e_bars.bootstrap_matrix = bt1_bootstrap;
    e_bars.se_upper = up1;
    e_bars.se_lower = lw1;
else 
    e_bars = struct;
end
