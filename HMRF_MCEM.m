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
function [sample, beta, mus, kappas] = HMRF_MCEM(data, dim, k, beta, mus, kappas, burn_in, sample_num, max_iter, neighbours_count, segment_init)

p = size(data, 2);

if(nargin < 11)
    segment_init = randi(k, dim);
end

for i=1:max_iter
    fprintf('\HRMF MCEM Iteration: %d of %d\n',i,max_iter);
    % генерируем выборку из схемы Гиббса 
    [samples] = GibbsSamplerVMF(data, segment_init, burn_in, sample_num, k, p, beta, mus, kappas, neighbours_count);
    % подстраиваем параметры
    [beta, mus, kappas] = EstimateParametersHMRFMCEM(data, samples, k, p, beta, mus, kappas);
    % генерируем начальную конфигурацию
    segment_init = reshape(samples(end, :), dim);
end

sample = reshape(samples(end, :), dim);