%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%   data: исходные данные, матрица XxYxZxP
%   p: последняя размерность (цвета или time series)
%   k: количество классов
%   beta: параметр модели Поттса
%   mus: параметры фон Мизеса-Фишера
%   kappas: параметры фон Мизеса-Фишера
%   lambda: параметр Mean Field
%   MAP_iter: максимальное количество итераций
%   INNER_iter: максимальное количество внутренних итераций
%   neighbours_count: количество соседей, доступные значения
%   2-D: 4, 8, 16
%   3-D: 6, 26
%   mask: маска для изображения, матрица XxYxZ
%   1 для точки производятся вычисления
%   0 точка игнорируется
%---output--------------------------------------------------------
%   sample: матрица финальной сегментации
%   beta: оцененное значение beta
%   kappas: параметры фон Мизеса-Фишера
%   mus: параметры фон Мизеса-Фишера
function [sample, beta, mus, kappas, sample2]=HMRF_VEM(data, p, k, beta, mus, kappas, ...
                                        lambda, MAP_iter, INNER_iter, ...
                                        neighbours_count, mask, labelCosts)

sz = size(data);
flatsize = prod(sz(1:end-1));
flat = reshape(data, [flatsize, p]);

if(nargin < 11)
   mask = ones(sz(1:end-1));
end

% neighbours indexes
all_neighbours_ind = GetNeighbours(sz(1:end-1), neighbours_count);

for i=1:MAP_iter
    fprintf('\tIteration: %d out of %d\n',i,MAP_iter);
    [~, logprobs] = CalculateLikelihoodProbabilities(flat, k, kappas, mus, mask);
    Q = zeros(flatsize, k);
    Q(:, 1) = ones(flatsize, 1);
    for u=1:INNER_iter
        if(nargin < 12)
            Qtilde = exp(beta ...
                * squeeze(sum(reshape(Q(all_neighbours_ind, :), [neighbours_count, flatsize, k]), 1)) ...
                - logprobs');
        else
            invQ = 1 - Q;
            H = double(zeros(size(invQ)));
            A = double(zeros(size(invQ)));
            for idy=1:k
                A(1, idy) = labelCosts(idy) * prod(invQ(2:end, idy), 1);
                for idx=2:flatsize
                    A(idx, idy) = A(idx-1, idy) * invQ(idx-1, idy) / invQ(idx, idy);
                end
            end
            for idx=1:flatsize
                H(idx, 1) = sum(A(idx, 2:end), 2);
                for idy=2:k
                    H(idx, idy) = H(idx, idy-1) + A(idx, idy-1) - A(idx, idy);
                end
            end
            Qtilde = exp(beta ...
                * squeeze(sum(reshape(Q(all_neighbours_ind, :), [neighbours_count, flatsize, k]), 1)) ...
                - logprobs' ...
                + H);
        end
        Qtilde = min(Qtilde, 10^100);
        sumQ = sum(Qtilde, 2);
        sumQ(sumQ==0) = 1;
        Q = (1-lambda) * Q + lambda * Qtilde ./ sumQ;
        Q(mask==0, :) = 0;
    end
    for l=1:k
        R = Q(mask~=0, l)' * flat(mask~=0,:);
        Rlen = sqrt(sum(R .^ 2));
        if Rlen ~= 0
            mus(l, :) = R / Rlen;
        else
            fprintf('WARNING: Rlen is zero\n');
        end
        
        eqn = @(x) besseli(p/2, x, 1)/besseli(p/2-1,x, 1) * sum(Q(mask~=0, l)) - Rlen;
        initval = kappas(l);
        if isnan(eqn(initval))
            fprintf('WARNING: function is NaN at initial value\n');
            initval = 10;
        end
        opts1 =  optimset('display','off');
        kappas(l) = lsqnonlin(eqn, initval, 0, Inf, opts1);
    end
end

% Find MAP
[~, sample] = max(Q, [], 2);
sample = reshape(sample, sz(1:end-1));
sample(mask==0) = 0;
[~, logprobs] = CalculateLikelihoodProbabilities(flat, k, kappas, mus, sample);
sample2 = MRF_MAP_GraphCutAExpansion(sample, logprobs, beta, k, MAP_iter, neighbours_count);
