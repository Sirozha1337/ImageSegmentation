%%  Алгоритм последовательной оценки параметров посредством генерации выборки из схемы Гиббса
%---input---------------------------------------------------------
%   data: исходные данные, матрица NxP, N - количество пикселей
%   dim: размер матрицы финальной сегментации
%   k: количество меток
%   beta: начальное значение параметра модели Поттса, скаляр
%   mus: начальное значение параметра vMF, матрица LxP
%   kappas: начальное значение параметра vMF, вектор Lx1
%   burn_in: количество отбрасываемых звеньев из схемы Гиббса
%   sample_num: количество звеньев используемых для оценки параметров
%   max_iter: максимальное количество итераций
%   neighbours_count: количество соседей, доступные значения
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   sample: финальная сегментация
%   beta: оцененное значение beta
%   mus: оцененное значение mus
%   kappas: оцененное значение kappas
function [sample, beta, mus, kappas] = Grab_Cut(data, segment, k, beta, mus, kappas, map_iter, max_iter, neighbours_count)

dim = size(segment);
p = size(data, 2);

for i=1:max_iter
    fprintf('\t GrabCut Iteration: %d of %d\n',i,max_iter);
    % подстраиваем параметры
    [beta, mus, kappas] = EstimateParametersGrabCut(data, segment, k, p, beta, mus, kappas);
    % считаем вероятности
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % находим MAP
    [segment, ~] = MRF_MAP_GraphCutAExpansion(segment, logprobs, beta, k, map_iter, neighbours_count);
end

sample = reshape(segment, dim);