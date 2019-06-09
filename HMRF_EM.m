%%  �������� ���������������� ������ ���������� ����������� ���������� MAP-������
%---input---------------------------------------------------------
%   data: �������� ������, ������� NxP, N - ���������� ��������
%   dim: ������ ������� ��������� �����������
%   k: ���������� �����
%   beta: ��������� �������� ��������� ������ ������, ������
%   mus: ��������� �������� ��������� vMF, ������� LxP
%   kappas: ��������� �������� ��������� vMF, ������ Lx1
%   map_iter: ������������ ���������� �������� ��� ���������� MAP-������
%   max_iter: ������������ ���������� ��������
%   neighbours_count: ���������� �������, ��������� ��������
%   2-D: 4, 8, 16
%   3-D: 6, 26
%   method: ����� ���������� MAP-������, ��������� ��������
%   expansion
%   swap
%---output--------------------------------------------------------
%   sample: ��������� �����������
%   beta: ��������� �������� beta
%   mus: ��������� �������� mus
%   kappas: ��������� �������� kappas
function [sample, beta, mus, kappas] = HMRF_EM(data, dim, k, beta, mus, kappas, map_iter, max_iter, neighbours_count, method)

p = size(data, 2);
for i=1:max_iter
    % ������� �����������
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % ���������� ��������� ������������
    segment_init = randi(k, dim);
    % ���� MAP-������
    switch method
        case "expansion"
            [sample, energy] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, beta, k, map_iter, neighbours_count);
        case "swap"
            [sample, energy] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, beta, k, map_iter, neighbours_count);
        otherwise
            error('Unknown method')
    end
    % ����������� ���������
    [beta, mus, kappas] = EstimateParametersHMRFEM(data, sample, energy, k, p, beta, mus, kappas);
end