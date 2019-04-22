function [dsc] = SimilarityScore(gt, est, k)

dscs = zeros(1, k);
inv_est = -est;
t_est = zeros(size(est));
combinations = perms(1:3);
combinations_len = size(combinations, 1);

for i=1:combinations_len
    for j=1:k
        t_est(inv_est==-j) = combinations(i, j);
    end
    dscs(i) = 2 * sum(gt(:)==t_est(:)) / (size(t_est(:), 1)+size(gt(:),1));
end

dsc = max(dscs);