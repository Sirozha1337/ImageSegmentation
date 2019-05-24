%% Calculates Jaccard Similarity score for multiclass segmentation
% function rearranges estimated segmentation and choses the best score
% it provides a way to properly evaluate unsupervised segmentation
%---input---------------------------------------------------------
% gt: Ground Truth segmentation
% est: Estimated segmentation
% k: Number of segmentation classes
%---output--------------------------------------------------------
% dsc: Jaccard Similarity score
% map: Estimated segmentation
function [dsc, map] = SimilarityScore(gt, est, k)

dscs = zeros(1, k);
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
    % Jaccard multiclass score
    numer = 0;
    denom = 0;
    for j=1:k
        numer = numer + sum(t_est(:) == j & gt(:) == j);
        denom = denom + sum(t_est(:) == j & gt(:) == j) + sum(t_est(:) == j & gt(:) ~= j) + sum(t_est(:) ~= j & gt(:) == j);
    end
    dscs(i) = numer / denom;
    % Save current label map
    maps(i, :) = t_est(:);
end

[dsc, ind] = max(dscs);
map = maps(ind, :);