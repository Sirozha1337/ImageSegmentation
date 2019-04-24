function [map] = SimulatedAnnealing(segment, logprobs, k, beta, t0, eta, max_iter, neighbours_count)

neighbours = GetNeighbours(size(segment), neighbours_count);
for i=1:max_iter
    pointlist = randperm(numel(segment));
    t = (eta ^ (i - 1)) * t0;
    fprintf('\tSimulated Annealing Iteration: %d of %d\n', i, max_iter);
    for j=pointlist
        new_label = randi([1,k], 1, 1);
        phi1 = (logprobs(segment(j), j) + sum(beta * (neighbours(:, j) ~= segment(j))));
        phi2 = (logprobs(new_label, j) + sum(beta * (neighbours(:, j) ~= new_label)));
        if phi1 > phi2
            segment(j) = new_label;
        else
            u = rand();
            p = exp((phi1-phi2)/t);
            if u < p
                segment(j) = new_label;
            end
        end
    end
end

map = segment;