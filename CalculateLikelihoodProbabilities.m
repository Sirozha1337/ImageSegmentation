%% Calculates likelihood probabilities and log-likelihood probabilites for von Mises-Fisher distribution
%---input---------------------------------------------------------
% data: for which to calculate probabilities, NxP
% k: number of classes
% kappas: vMF distribution parameters, Kx1
% mus: vMF ditribution parameters, KxP
% mask: 0-1 matrix size of data
%---output--------------------------------------------------------
% probs: likelihood probabilities, LxN
% logprobs: loglikelihood probabilities, LxN
function [probs,logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, mask)

if nargin < 5
  mask = ones(size(data, 1), 1);
end

probs = zeros([k, size(data, 1)]);
logprobs = zeros([k, size(data, 1)]);

for i=1:k
    logprobs(i, mask~=0) = -kappas(i) * mus(i, :) * data(mask~=0, :)' - log(C(size(data,2), kappas(i)));
    probs(i, mask~=0) = C(size(data,2), kappas(i)) * min(exp(kappas(i) * mus(i, :) * data(mask~=0, :)'), 10^100);
end
