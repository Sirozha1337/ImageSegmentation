%% Generate samples from Gibbs Distribution
% Yinit - initial state
% B - burn in samples
% M - number of samples to produce
% k - number of possible values (they start from 1)
% beta - Potts model parameter
% logprobs - negative log likelihood probabilities, KxN matrix
% neighbours_count - number of neighbours
%% Returns
%% Latest produced sample, same dimensions as Yinit
%% All produced samples: MxN
function [Y, sample] = GibbsSamplerLabelCost(Yinit, B, M, k, beta, logprobs, neighbours_count, labelCosts)

sz = size(Yinit);
Yflat = Yinit(:);
flatsz = size(Yflat,1);
all_neighbours_ind = GetNeighbours(sz, neighbours_count);
Y = zeros(M, flatsz);

for j=1:(B+M)
    permutations = randperm(flatsz);
    non_zero_perms = permutations(Yinit(permutations)~=0);
    for i=non_zero_perms
       neighbours = repmat(all_neighbours_ind(all_neighbours_ind(:, i)~=i, i), 1, k);
       neib = -beta * sum(Yflat(neighbours)~=1:k, 1);
       
       if nargin < 8
           pex = min(exp(neib - logprobs(:, i)'), 10^100);
       else
           n_counts = zeros(1, k);
           for l=1:k
                n_counts(l) = sum(labelCosts(:)' .* min(sum([Yflat(neighbours); repmat(l, 1, k)] == 1:k, 1), 1));
           end
           pex = min(exp(neib - logprobs(:, i)' - n_counts), 10^100);
       end
       norm_const = sum(pex);
       P = pex/norm_const;
       
       if sum(P.^2) > 0
            ind = randsample(1:k, 1, true, P);
            Yflat(i) = ind;
       end
    end
    if j > B
        Y(j-B, :) = Yflat;
    end
end
sample = reshape(Y(end, :), sz);