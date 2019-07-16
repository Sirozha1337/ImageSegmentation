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
function [sample, beta, mus, kappas, all_mus, all_kappas] = HMRF_MCEM(data, dim, k, beta, mus, kappas, burn_in, sample_num, max_iter, neighbours_count, segment_init)

p = size(data, 2);
all_mus = zeros([max_iter, size(mus)]);
all_kappas = zeros([max_iter, size(kappas)]);

if(nargin < 11)
    segment_init = randi(k, dim);
end

for i=1:max_iter
    fprintf('\tHRMF MCEM Iteration: %d of %d\n',i,max_iter);
    % ���������� ������� �� ����� ������ 
    [samples] = GibbsSamplerVMF(data, segment_init, burn_in, sample_num, k, p, beta, mus, kappas, neighbours_count);
    % ������������ ���������
    [beta, mus, kappas] = EstimateParametersHMRFMCEM(data, samples, k, p, beta, mus, kappas);
    all_mus(i, :, :) = mus;
    all_kappas(i, :) = kappas;
    % ���������� ��������� ������������
    segment_init = reshape(samples(end, :), dim);
end

sample = reshape(samples(end, :), dim);