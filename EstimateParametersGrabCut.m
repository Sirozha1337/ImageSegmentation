%% Оценивает параметры beta, mu, kappa по выборке из схемы Гиббса
%---input---------------------------------------------------------
% X: 4-х мерная матрица исходных данных
% Y: матрица размерности MxN
% k: количество классов сегментации
% p: размерность данных
% mu: текущая матрица параметров для vMF
% kappa: текущий вектор параметров для vMF
%---output--------------------------------------------------------
% mu: новые значения параметров, матрица LxP
% kappa: новые значения параметров, вектор 1xL
function [mu, kappa] = EstimateParametersGrabCut(X, Y, k, p, mu, kappa)

for l=1:k
    R = sum(X(Y(:)==l, :));
    N = sum(Y(:)==l);
    Rlen = sqrt(sum(R .^ 2));
    if R ~= 0
        mu(l, :) = R / Rlen;
    end
    
    eqn = @(x) min(besseli(p/2, x), 10^100) / min(besseli(p/2-1,x), 10^100) - Rlen / N;
    initval = kappa(l);
    if isnan(eqn(initval))
        fprintf('WARNING: function is NaN in initial point\n')
        initval = 10;
    end
    
    opts1 =  optimset('display','off');
    kappa(l) =  lsqnonlin(eqn, initval, 0, Inf, opts1);
    if kappa(l) < 0 || isnan(kappa(l))
        kappa(l) = abs(kappa(l));
        fprintf('WARNING: Lower than zero solution was found\n')
    end
end
