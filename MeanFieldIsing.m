%%  јлгоритм нахождени€ MAP-оценки
%---input---------------------------------------------------------
%   Y: исходные данные, матрица XxYxZxP
%   p: последн€€ размерность (цвета или time series)
%   kappa: параметры фон ћизеса-‘ишера
%   mu: параметры фон ћизеса-‘ишера
%   lambda: параметр Mean Field
%   b: параметр модели ѕоттса
%   L: количество классов
%   MAP_iter: максимальное количество итераций
%   INNER_iter: максимальное количество внутренних итераций
%   neighbours_count: количество соседей, доступные значени€
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   Q: матрица веро€тности принадлежности к классу
%   X: матрица финальной сегментации
function [Q, X]=MeanFieldIsing(Y,p,kappa,mu,lambda,b,L,MAP_iter,INNER_iter,neighbours_count)

sz = size(Y);
flatsize = prod(sz(1:end-1));
flat = reshape(Y, [flatsize, p]);

% neighbours indexes
all_neighbours_ind = GetNeighbours(sz(1:end-1), neighbours_count);

for i=1:MAP_iter
    [~, logprobs] = CalculateLikelihoodProbabilities(flat, L, kappa, mu);
    Q = zeros(flatsize, L);
    Q(:, 1) = ones(flatsize, 1);
    for u=1:INNER_iter
        Qtilde = exp(b*squeeze(sum(reshape(Q(all_neighbours_ind, :), [neighbours_count, flatsize, L]), 1))-logprobs');
        sumQ = sum(Qtilde, 2);
        Q = (1-lambda) * Q + lambda * Qtilde ./ sumQ;
    end
    for l=1:L
        R = Q(:, l)'*flat;
        Rlen = sqrt(sum(R .^ 2));
        if Rlen ~= 0
            mu(l, :) = R / Rlen;
        end
        
        eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * sum(Q(:, l)) - Rlen;
        initval = kappa(l);
        opts1 =  optimset('display','off');
        kappa(l) = lsqnonlin(eqn, initval, 0, Inf, opts1);
    end
end

[~, X] = max(Q, [], 2);
X = reshape(X, sz(1:end-1));
