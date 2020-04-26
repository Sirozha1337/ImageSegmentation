%% Оценивает параметры beta, mu, kappa по данным и матрице апостериорных вероятностей
%---input---------------------------------------------------------
% X: 4-х мерная матрица исходных данных
% posterior: апостериорные вероятности, LxN
% k: количество классов сегментации
% p: размерность данных
% beta: текущее значение параметра модели Поттса
% mu: текущая матрица параметров для vMF
% kappa: текущий вектор параметров для vMF
%---output--------------------------------------------------------
% beta: новое значение параметра, скаляр
% mu: новые значения параметров, матрица LxP
% kappa: новые значения параметров, вектор 1xL
function [beta, mu, kappa] = EstimateParametersHMRFEM(X, Y, posterior, k, p, beta, mu, kappa)

for l=1:k
    R = posterior(l,Y==l) * X(Y==l,:);
    Rlen = sqrt(sum(R .^ 2));
    if Rlen ~= 0
        mu(l, :) = R / Rlen;
    end
    
    eqn = @(x) besseli(p/2, x, 1)/besseli(p/2-1,x, 1) * sum(posterior(l,Y==l)) - Rlen;
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
