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
