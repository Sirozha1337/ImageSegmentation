%% Calculates Simple Similarity Score for multiclass segmentation
% function rearranges estimated segmentation and choses the best score
% it provides a way to properly evaluate unsupervised segmentation
%---input---------------------------------------------------------
% gt: Ground Truth segmentation
% est: Estimated segmentation
% k: Number of segmentation classes
%---output--------------------------------------------------------
% dsc: Simple Similarity Score
% map: Estimated segmentation
function [dsc, map] = SimpleSimilarityScore(gt, est, k)

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
    % Simple Similarity Score
    dscs(i) = sum(t_est(:)==gt(:))/numel(gt);
    % Save current label map
    maps(i, :) = t_est(:);
end

[dsc, ind] = max(dscs);
map = maps(ind, :);