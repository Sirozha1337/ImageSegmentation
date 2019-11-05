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
    R_mod = sqrt(sum(R .^ 2));
    if R_mod ~= 0
        mu(l, :) = R / R_mod;
    end
    eqn = @(x) min(besseli(p/2, x), 10^100)/min(besseli(p/2-1,x), 10^100) * N / M - R_mod / M;
    initval = kappa(l);
    if isnan(eqn(initval))
        fprintf('WARNING: function is NaN in initial point\n')
        initval = 10;
    end
    opts1 =  optimset('display','off');
    kappa(l) =  lsqnonlin(eqn, initval, 0, Inf, opts1);
    if kappa(l) < 0
        kappa(l) = abs(kappa(l));
        fprintf('WARNING: Lower than zero solution was found\n')
    end
    
    if isnan(kappa(l))
        kappa(l) = initval;
        fprintf('WARNING: NaN solution was found\n')
    end
end
