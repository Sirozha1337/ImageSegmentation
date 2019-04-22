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
function [beta, mu, kappa] = EstimateParametersGrabCut(X, Y, k, p, beta, mu, kappa)

for l=1:k
    R = sum(X(Y(:)==l, :));
    N = sum(Y(:)==l);
    Rlen = sqrt(sum(R .^ 2));
    if R ~= 0
        mu(l, :) = R / Rlen;
    end
    
    eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) - Rlen / N;
    initval = kappa(l);
    if isnan(eqn(initval))
        initval = 10;
    end
    kappa(l) = fzero(eqn, initval);
end
