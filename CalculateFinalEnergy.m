%%  Calculate Energy of the segmentation
%---input---------------------------------------------------------
%   segment: segmented image, matrix of N values ranging from 1 to L
%   logprobs: negative log likelihood probability, matrix L by N
%   b: potts model parameter, scalar
%   neighbours_count: neighbours count, depends on dimensionality of segment matrix
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   energy: energy of the segmenation
function [energy]=CalculateFinalEnergy(segment, logprobs, b, neighbours_count)

all_neighbours_ind = GetNeighbours(size(segment), neighbours_count);
segment = segment(:);

energy = 0;
for i=1:numel(segment)
    energy = energy + logprobs(segment(i), i) + b * sum(segment(i) ~= all_neighbours_ind(:, i));
end
