%% ��������� ��������� beta, mu, kappa �� ������� �� ����� ������
%---input---------------------------------------------------------
% X: 4-� ������ ������� �������� ������
% Y: ������� ����������� MxN
% k: ���������� ������� �����������
% p: ����������� ������
% beta: ������� �������� ��������� ������ ������
% mu: ������� ������� ���������� ��� vMF
% kappa: ������� ������ ���������� ��� vMF
%---output--------------------------------------------------------
% beta: ����� �������� ���������, ������
% mu: ����� �������� ����������, ������� LxP
% kappa: ����� �������� ����������, ������ 1xL
function [beta, mu, kappa] = EstimateParametersHMRFMCEM(X, Y, k, p, beta, mu, kappa)

M = size(Y, 1);

for l=1:k
    R = zeros(1, p);
    N = 0;
    for i=1:M
        R = R + sum(X(Y(i, :)==l, :));
        N = N + sum(Y(i, :)==l);
    end
    if R ~= 0
        mu(l, :) = R / sqrt(sum(R .^ 2));
    end
    
    eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * N / M - sqrt(sum(R .^ 2)) / M;
    initval = kappa(l);
    if isnan(eqn(initval))
        fprintf('WARNING: function is NaN in initial point\n')
        initval = 10;
    end
    kappa(l) = fzero(eqn, initval);
    if kappa(l) < 0 || isnan(kappa(l))
        kappa(l) = abs(kappa(l));
        fprintf('WARNING: Lower than zero solution was found\n')
    end
end
