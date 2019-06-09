%% Calculates Area Under the Curve score
% function rearranges estimated segmentation and choses the best score
% it provides a way to properly evaluate unsupervised segmentation
%---input---------------------------------------------------------
% gt: Ground Truth segmentation
% posterior: Probability that pixel i belongs to class j, matrix KxN
% k: Number of segmentation classes
%---output--------------------------------------------------------
% auc: Area Under the Curve score
function [auc] = AUC(gt, posterior, k, iter)

combinations = perms(1:k);
combinations_len = size(combinations, 1);
inv_est = -gt;
t_est = zeros(size(gt));
t_posterior = zeros(size(posterior));

aucs = zeros(1, k);
for c=1:combinations_len
    for j=1:k
        t_est(inv_est==-j) = combinations(c, j);
        t_posterior(j, :) = posterior(combinations(c, j), :);
    end
    
    summ = 0;
    for i=1:k
        for j=(i+1):k
            Aij = 0.0;
            Aji = 0.0;
            is = randsample(find(t_est==i), iter, true);
            js = randsample(find(t_est==j), iter, true);
            for ind=1:iter
                Aij = Aij + double(posterior(i, is(ind)) > posterior(i, js(ind)));
                Aji = Aji + double(posterior(j, js(ind)) > posterior(j, is(ind)));
            end
            Aij = Aij / iter;
            Aji = Aji / iter;
            summ = summ + (Aij + Aji) / 2;
        end
    end
    aucs(c) = 2/(k*(k-1))*summ;
end

auc = max(aucs);