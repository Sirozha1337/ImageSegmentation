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
function [beta, mu, kappa] = EstimateParametersHMRFEM(X, posterior, k, p, beta, mu, kappa)

for l=1:k
    R = posterior(l,:) * X;
    Rlen = sqrt(sum(R .^ 2));
    if Rlen ~= 0
        mu(l, :) = R / Rlen;
    end
    
    eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * sum(posterior(l,:)) - Rlen;
    initval = kappa(l);
    kappa(l) = fzero(eqn, initval);
    if kappa(l) < 0
        kappa(l) = abs(kappa(l));
    end
end
