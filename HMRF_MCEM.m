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
%   mask: �����
%---output--------------------------------------------------------
%   sample: ��������� ����������� �� ������ ��������� ����������
%   sample2: ��������� ����������� ����� ���������� � ��� GraphCut
%   beta: ��������� �������� beta
%   mus: ��������� �������� mus
%   kappas: ��������� �������� kappas
function [sample, beta, mus, kappas, sample2] = HMRF_MCEM(data, segment_init, k, beta, mus, kappas, burn_in, sample_num, max_iter, neighbours_count, mask)

p = size(data, 2);

if(nargin < 11)
    mask = ones(size(segment_init));
end
segment_init(mask==0) = 0;
    
for i=1:max_iter
    fprintf('\tHRMF MCEM Iteration: %d of %d\n',i,max_iter);
    % ���������� ������� �� ����� ������ 
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, mask);
    [samples] = GibbsSamplerLabelCost(segment_init, burn_in, sample_num, k, beta, logprobs, neighbours_count);
    % ������������ ���������
    [beta, mus, kappas] = EstimateParametersHMRFMCEM(data, samples, k, p, beta, mus, kappas);
    % ���������� ��������� ������������
    segment_init = reshape(samples(end, :), size(segment_init));
end

% Find MAP
sample = reshape(samples(end, :), size(segment_init));
[~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, sample);
sample2 = MRF_MAP_GraphCutAExpansion(sample, logprobs, beta, k, max_iter, neighbours_count);
