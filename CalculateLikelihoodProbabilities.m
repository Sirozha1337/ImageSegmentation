
function [probs,logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus)

probs = zeros([k, size(data, 1)]);
logprobs = zeros([k, size(data, 1)]);

for i=1:k
    logprobs(i, :) = -kappas(i) * mus(i, :) * data' - log(C(size(data,2), kappas(i)));
    probs(i, :) = C(size(data,2), kappas(i)) * min(exp(kappas(i) * mus(i, :) * data'), 10^9);
end
