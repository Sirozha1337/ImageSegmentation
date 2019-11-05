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
%   sample: финальная сегментация
%   beta: оцененное значение beta
%   mus: оцененное значение mus
%   kappas: оцененное значение kappas
function [sample, beta, mus, kappas, all_mus, all_kappas] = HMRF_MCEM(data, segment_init, k, beta, mus, kappas, burn_in, sample_num, max_iter, neighbours_count, mask)

p = size(data, 2);
all_mus = zeros([max_iter, size(mus)]);
all_kappas = zeros([max_iter, size(kappas)]);

if(nargin < 11)
    mask = ones(size(segment_init));
end
segment_init(mask==0) = 0;
    
for i=1:max_iter
    fprintf('\tHRMF MCEM Iteration: %d of %d\n',i,max_iter);
    % генерируем выборку из схемы Гиббса 
    [samples] = GibbsSamplerVMF(data, segment_init, burn_in, sample_num, k, p, beta, mus, kappas, neighbours_count);
    % подстраиваем параметры
    [beta, mus, kappas] = EstimateParametersHMRFMCEM(data, samples, k, p, beta, mus, kappas);
    all_mus(i, :, :) = mus;
    all_kappas(i, :) = kappas;
    % генерируем начальную конфигурацию
    segment_init = reshape(samples(end, :), size(segment_init));
end

% Find MAP
sample = reshape(samples(end, :), size(segment_init));
[~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus, sample);
sample = MRF_MAP_GraphCutAExpansion(sample, logprobs, beta, k, max_iter, neighbours_count);
