%%  �������� ���������������� ������ ���������� ����������� ��������� ������� �� ����� ������
%---input---------------------------------------------------------
%   data: �������� ������, ������� NxP, N - ���������� ��������
%   dim: ������ ������� ��������� �����������
%   k: ���������� �����
%   lambda: �������� �����������
%   sigma: �������� �����������
%   mus: ��������� �������� ��������� vMF, ������� LxP
%   kappas: ��������� �������� ��������� vMF, ������ Lx1
%   burn_in: ���������� ������������� ������� �� ����� ������
%   sample_num: ���������� ������� ������������ ��� ������ ����������
%   max_iter: ������������ ���������� ��������
%   neighbours_count: ���������� �������, ��������� ��������
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   sample: ��������� �����������
%   beta: ��������� �������� beta
%   mus: ��������� �������� mus
%   kappas: ��������� �������� kappas
function [sample, mus, kappas, all_mus, all_kappas] = Grab_Cut(data, segment, k, lambda, sigma, mus, kappas, map_iter, max_iter, neighbours_count)

dim = size(segment);
p = size(data, 2);
all_mus = zeros([max_iter, size(mus)]);
all_kappas = zeros([max_iter, size(kappas)]);

for i=1:max_iter
    fprintf('\t GrabCut Iteration: %d of %d\n',i,max_iter);
    % ������������ ���������
    [mus, kappas] = EstimateParametersGrabCut(data, segment, k, p, mus, kappas);
    all_mus(i, :, :) = mus;
    all_kappas(i, :) = kappas;
    % ������� �����������
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % ������� Bpq
    %Neighbours = GetNeighbours(dim, neighbours_count);
    %Bpq = lambda * exp(sum((repmat(data, neighbours_count, 1)-data(Neighbours, :)).^2, 2)./(2*sigma^2));
    %Bpq = reshape(Bpq, [ prod(dim), neighbours_count ]);
    % ������� MAP
    [segment, ~] = MRF_MAP_GraphCutABSwap(segment, logprobs, 2, k, map_iter, neighbours_count);
end

sample = reshape(segment, dim);