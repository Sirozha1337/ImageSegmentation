%% Оценивает параметры beta, mu, kappa по выборке из схемы Гиббса
%---input---------------------------------------------------------
% X: 4-х мерная матрица исходных данных
% Y: матрица размерности MxN
% k: количество классов сегментации
% p: размерность данных
% beta: текущее значение параметра модели Поттса
% mu: текущая матрица параметров для vMF
% kappa: текущий вектор параметров для vMF
%---output--------------------------------------------------------
% beta: новое значение параметра, скаляр
% mu: новые значения параметров, матрица LxP
% kappa: новые значения параметров, вектор 1xL
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
