%%  �������� ��������� ������������� ������ �� ������������� von Mises-Fisher
%---input---------------------------------------------------------
%   init: ��������� ������������
%   p: ���������� ���������
%   k: ���������� �����
%   beta: �������� ������ ������
%   mus: �������� ������������� vMF
%   kappas: �������� ������������� vMF
%   burn_in: ���������� ������������� �������� ����� ������
%   neighbours_count: ���������� �������, ��������� ��������
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   data: ������ �� ������������� vMF
%   gt: ground truth ��� ��������������� ������
function [data, gt]=GenerateSynteticData(init, p, k, beta, mus, kappas, burn_in, neighbours_count)

% ����� ������ 
gt = GibbsSamplerPotts(init, burn_in, 1, k, beta, neighbours_count);
% ����� ������ �������� �� ��������������� �������
gt = gt(1, :);

% ��������� ��� vMF
theta = cell([k, 1]);
for i=1:k
    theta{i}.kappa = kappas(i);
    theta{i}.mu = mus(i, :)';
end

D = vmffactory(p);
data = zeros([size(init(:), 1), p]);
for i=1:k
    % ���������� ������� �� VMF � ��������� �����������
    clust = D.sample(theta{i}, sum(gt==i));
    % �������� ��������������� �������� � ��������������� ��� ��� �����
    data(gt==i, :) = clust';
end