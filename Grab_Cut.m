%%  Алгоритм последовательной оценки параметров посредством генерации выборки из схемы Гиббса
%---input---------------------------------------------------------
%   data: исходные данные, матрица NxP, N - количество пикселей
%   dim: размер матрицы финальной сегментации
%   k: количество меток
%   lambda: параметр сглаживания
%   sigma: параметр сглаживания
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
function [sample, mus, kappas, all_mus, all_kappas] = Grab_Cut(data, segment, k, lambda, sigma, mus, kappas, map_iter, max_iter, neighbours_count)

dim = size(segment);
p = size(data, 2);
all_mus = zeros([max_iter, size(mus)]);
all_kappas = zeros([max_iter, size(kappas)]);

for i=1:max_iter
    fprintf('\t GrabCut Iteration: %d of %d\n',i,max_iter);
    % подстраиваем параметры
    [mus, kappas] = EstimateParametersGrabCut(data, segment, k, p, mus, kappas);
    all_mus(i, :, :) = mus;
    all_kappas(i, :) = kappas;
    % считаем вероятности
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % считаем Bpq
    %Neighbours = GetNeighbours(dim, neighbours_count);
    %Bpq = lambda * exp(sum((repmat(data, neighbours_count, 1)-data(Neighbours, :)).^2, 2)./(2*sigma^2));
    %Bpq = reshape(Bpq, [ prod(dim), neighbours_count ]);
    % находим MAP
    [segment, ~] = MRF_MAP_GraphCutABSwap(segment, logprobs, 2, k, map_iter, neighbours_count);
end

sample = reshape(segment, dim);