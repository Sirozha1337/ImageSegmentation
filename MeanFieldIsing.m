%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%   Y: исходные данные, матрица XxYxZxP
%   p: последняя размерность (цвета или time series)
%   kappa: параметры фон Мизеса-Фишера
%   mu: параметры фон Мизеса-Фишера
%   lambda: параметр Mean Field
%   b: параметр модели Поттса
%   L: количество классов
%   MAP_iter: максимальное количество итераций
%   neighbours_count: количество соседей, доступные значения
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   Q: матрица вероятности принадлежности к классу
%   X: матрица финальной сегментации
function [Q, X]=MeanFieldIsing(Y,p,kappa,mu,lambda,b,L,MAP_iter,neighbours_count)

sz = size(Y);
flatsize = prod(sz(1:end-1));
flat = reshape(Y, [flatsize, p]);

% neighbours indexes
all_neighbours_ind = GetNeighbours(sz(1:end-1), neighbours_count);

Q = zeros(flatsize, L);
Q(:, 1) = ones(flatsize, 1);
for i=1:MAP_iter
    [~, logprobs] = CalculateLikelihoodProbabilities(flat, L, kappa, mu);
    for index=1:flatsize
        Qtilde = zeros(L, 1);
        for l=1:L
            Qtilde(l) = exp(b*sum(Q(all_neighbours_ind(:,index), l))-logprobs(l, index));
        end
        sumQ = sum(Qtilde);
        for l=1:L
            Q(index, l) = (1-lambda)*Q(index, l)+lambda*Qtilde(l)/sumQ;
        end
    end
    for l=1:L
        R = Q(:, l)'*flat;
        Rlen = sqrt(sum(R .^ 2));
        if Rlen ~= 0
            mu(l, :) = R / Rlen;
        end
        
        eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * R - Rlen;
        initval = kappa(l);
        
        opts1 =  optimset('display','off');
        kappa(l) = lsqnonlin(eqn, initval, 0, Inf, opts1);
    end
end

[~, X] = max(Q, [], 2);
X = reshape(X, sz(1:end-1));
