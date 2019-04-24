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
