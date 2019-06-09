%%  јлгоритм последовательной оценки параметров посредством нахождени€ MAP-оценки
%---input---------------------------------------------------------
%   data: исходные данные, матрица NxP, N - количество пикселей
%   dim: размер матрицы финальной сегментации
%   k: количество меток
%   beta: начальное значение параметра модели ѕоттса, скал€р
%   mus: начальное значение параметра vMF, матрица LxP
%   kappas: начальное значение параметра vMF, вектор Lx1
%   map_iter: максимальное количество итераций дл€ нахождени€ MAP-оценки
%   max_iter: максимальное количество итераций
%   neighbours_count: количество соседей, доступные значени€
%   2-D: 4, 8, 16
%   3-D: 6, 26
%   method: метод нахождени€ MAP-оценки, доступные значени€
%   expansion
%   swap
%---output--------------------------------------------------------
%   sample: финальна€ сегментаци€
%   beta: оцененное значение beta
%   mus: оцененное значение mus
%   kappas: оцененное значение kappas
function [sample, beta, mus, kappas] = HMRF_EM(data, dim, k, beta, mus, kappas, map_iter, max_iter, neighbours_count, method)

p = size(data, 2);
for i=1:max_iter
    % считаем веро€тности
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % генерируем начальную конфигурацию
    segment_init = randi(k, dim);
    % ищем MAP-оценку
    switch method
        case "expansion"
            [sample, energy] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, beta, k, map_iter, neighbours_count);
        case "swap"
            [sample, energy] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, beta, k, map_iter, neighbours_count);
        otherwise
            error('Unknown method')
    end
    % настраиваем параметры
    [beta, mus, kappas] = EstimateParametersHMRFEM(data, sample, energy, k, p, beta, mus, kappas);
end