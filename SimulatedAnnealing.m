function [map] = SimulatedAnnealing(segment, logprobs, k, beta, t0, eta, max_iter, neighbours_count)

neighbours = GetNeighbours(size(segment), neighbours_count);
for i=1:max_iter
    pointlist = randperm(numel(segment));
    t = (eta ^ (i - 1)) * t0;
    fprintf('\tSimulated Annealing Iteration: %d of %d\n', i, max_iter);
    for j=pointlist
        lik = zeros([1, k]);
        for l=1:k
            lik(l) = exp((-logprobs(l, j) + sum(beta * (neighbours(:, j) ~= l)))/t);
        end
        segment(j) = randsample(1:l, 1, true, lik);
    end
end

map = segment;