%%  �������� ���������������� ������ ���������� ����������� ��������� ������� �� ����� ������
%---input---------------------------------------------------------
%   data: �������� ������, ������� NxP, N - ���������� ��������
%   dim: ������ ������� ��������� �����������
%   k: ���������� �����
%   beta: ��������� �������� ��������� ������ ������, ������
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
function [sample, beta, mus, kappas] = Grab_Cut(data, segment, k, beta, mus, kappas, map_iter, max_iter, neighbours_count)

dim = size(segment);
p = size(data, 2);

for i=1:max_iter
    fprintf('\t GrabCut Iteration: %d of %d\n',i,max_iter);
    % ������������ ���������
    [beta, mus, kappas] = EstimateParametersGrabCut(data, segment, k, p, beta, mus, kappas);
    % ������� �����������
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % ������� MAP
    [segment, ~] = MRF_MAP_GraphCutAExpansion(segment, logprobs, beta, k, map_iter, neighbours_count);
end

sample = reshape(segment, dim);