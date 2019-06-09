%% ��������� ��������� beta, mu, kappa �� ������ � ������� ������������� ������������
%---input---------------------------------------------------------
% X: 4-� ������ ������� �������� ������
% posterior: ������������� �����������, LxN
% k: ���������� ������� �����������
% p: ����������� ������
% beta: ������� �������� ��������� ������ ������
% mu: ������� ������� ���������� ��� vMF
% kappa: ������� ������ ���������� ��� vMF
%---output--------------------------------------------------------
% beta: ����� �������� ���������, ������
% mu: ����� �������� ����������, ������� LxP
% kappa: ����� �������� ����������, ������ 1xL
function [beta, mu, kappa] = EstimateParametersHMRFEM(X, Y, posterior, k, p, beta, mu, kappa)

for l=1:k
    R = posterior(l,Y==l) * X(Y==l,:);
    Rlen = sqrt(sum(R .^ 2));
    if Rlen ~= 0
        mu(l, :) = R / Rlen;
    end
    
    eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * sum(posterior(l,Y==l)) - Rlen;
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
