%% Calculates Average Sensitivity and Specificity for multiclass segmentation
% function rearranges estimated segmentation and choses the best score
% it provides a way to properly evaluate unsupervised segmentation
%---input---------------------------------------------------------
% gt: Ground Truth segmentation
% est: Estimated segmentation
% k: Number of segmentation classes
%---output--------------------------------------------------------
% tpr: Sensitivity (Recall)
% tnr: Specificity
% map: Estimated segmentation
function [tpr, tnr, map] = TruePositiveNegativeRates(gt, est, k)

tprs = zeros(1, k);
tnrs = zeros(1, k);
inv_est = -est;
t_est = zeros(size(est));
combinations = perms(1:k);
combinations_len = size(combinations, 1);

maps = zeros([combinations_len, numel(gt)]);
for i=1:combinations_len
    % get next combination
    for j=1:k
        t_est(inv_est==-j) = combinations(i, j);
    end
    % Sensitivity and Specifity averaged by class number
    sum_tpr = 0;
    sum_tnr = 0;
    for j=1:k
        sum_tpr = sum_tpr + sum(t_est(:) == j & gt(:) == j) / (sum(t_est(:) == j & gt(:) == j) + sum(t_est(:) ~= j & gt(:) == j));
        sum_tnr = sum_tnr + sum(t_est(:) ~= j & gt(:) ~= j) / (sum(t_est(:) ~= j & gt(:) ~= j) + sum(t_est(:) == j & gt(:) ~= j));
    end
    tprs(i) = sum_tpr / k;
    tnrs(i) = sum_tnr / k;
    % Save current label map
    maps(i, :) = t_est(:);
end

[~, ind] = max(tprs+tnrs);
map = maps(ind, :);
tpr = tprs(ind);
tnr = tnrs(ind);