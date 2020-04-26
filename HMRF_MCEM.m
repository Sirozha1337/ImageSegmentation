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
%   mask: маска
%---output--------------------------------------------------------
%   sample: финальная сегментация на основе оцененных параметров
%   sample2: финальная сегментация после применения к ней GraphCut
%   beta: оцененное значение beta
%   mus: оцененное значение mus
%   kappas: оцененное значение kappas
function [sample, beta, mus, kappas, sample2] = HMRF_MCEM(data, segment_init, k, beta, mus, kappas, burn_in, sample_num, max_iter, neighbours_count, mask)

p = size(data, 2);

if(nargin < 11)
    mask = ones(size(segment_init));
end
segment_init(mask==0) = 0;
    
for i=1:max_iter
    fprintf('\tHRMF MCEM Iteration: %d of %d\n',i,max_iter);
    % генерируем выборку из схемы Гиббса 
    [~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, mask);
    [samples] = GibbsSamplerLabelCost(segment_init, burn_in, sample_num, k, beta, logprobs, neighbours_count);
    % подстраиваем параметры
    [beta, mus, kappas] = EstimateParametersHMRFMCEM(data, samples, k, p, beta, mus, kappas);
    % генерируем начальную конфигурацию
    segment_init = reshape(samples(end, :), size(segment_init));
end

% Find MAP
sample = reshape(samples(end, :), size(segment_init));
[~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, sample);
sample2 = MRF_MAP_GraphCutAExpansion(sample, logprobs, beta, k, max_iter, neighbours_count);
