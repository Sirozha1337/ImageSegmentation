function [map] = ICM(segment, logprobs, k, beta, max_iter, neighbours_count)

neighbours = GetNeighbours(size(segment), neighbours_count);
for i=1:max_iter
    pointlist = randperm(numel(segment));
    fprintf('\tICM Iteration: %d of %d\n', i, max_iter);
    for j=pointlist
        max_lik = -Inf;
        new_state = segment(j);
        for l=1:k
            lik = exp(-logprobs(l, j) + sum(beta * (neighbours(:, j) ~= l)));
            if lik > max_lik
                max_lik = lik;
                new_state = l;
            end
        end
        segment(j) = new_state;
    end
end

map = segment;