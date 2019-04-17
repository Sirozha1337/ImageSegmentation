% читаем данные
data = niftiread('NYU_TRT_session1a/sub05676/func/lfo.nii');

% вызываем сегментацию
% segmented = PerformSegmentation(data, 7, 3, 1, 2);

% все что идет дальше это скопированный код из PerformSegmentation
% который нужен был для тестов и визуализации

% раскладываем данные по рядам и нормализуем
k = 3;
timeseries_by_rows = reshape(data, [64*64*39, 197]);
timeseries_by_rows = double(timeseries_by_rows); 
lenghts = sqrt(sum(timeseries_by_rows .^ 2, 2));
non_empty_rows = lenghts > 1000;
non_empty_rows_indexes = find(lenghts > 1000);
timeseries_by_rows = timeseries_by_rows - mean(timeseries_by_rows(:));
timeseries_by_rows = NormalizeToUnitLength(timeseries_by_rows);

% получаем начальную сегментацию
segment_init_kmeans = kmeans(timeseries_by_rows(non_empty_rows), k, 'MaxIter',500);
segment_init = zeros([size(timeseries_by_rows, 1), 1]);
segment_init(non_empty_rows_indexes) = segment_init_kmeans;
% инициализируем структуры для параметров распределения
vmf = vmffactory(197);
mus = zeros(k, 197);
kappas = zeros(1, k);
for i=1:k
    % достаем данные соответствующие размеченной области
    roi = timeseries_by_rows(non_empty_rows & segment_init==i, :);
    [theta] = vmf.estimatedefault(roi');
    mus(i, :) = squeeze(theta.mu);
    kappas(i) = theta.kappa;
end
segment_init(~non_empty_rows) = 0;
% преобразуем обратно в 4х-мерную матрицу
normalized_data = reshape(timeseries_by_rows, [64,64,39,197]);
% преобразуем сегментированный массив в 3-х мерную матрицу
segment_init = reshape(segment_init, [64,64,39]);
g = GibbsSamplerVMF(normalized_data, segment_init, 1, 1, 7, 197, 1, mus, kappas, 26);
imagesc(segment_init(:,:,15));
% считаем вероятности P и -logP
[probs, logprobs] = CalculateLikelihoodProbabilities(timeseries_by_rows, k, kappas, mus);
% находим оценку максимального правдоподобия
segment_init = MLE(timeseries_by_rows, probs);
%figure();
%%segment_init(~non_empty_rows) = 0;
%segment_init = reshape(segment_init, [64,64,39]);
%imagesc(segment_init(:,:,15));

%g = MRF_MAP_GraphCutABSwap(segment_init, probs, 0.5, k, 1);
%imagesc(g(:,:,15));
% вызываем схему Гиббса
g = GibbsSamplerVMF(normalized_data, segment_init, 1, 1, 7, 197, 1, mus, kappas);

% визуализация
% начальная разметка
imagesc(segment_init(:,:,6));
figure();
% разметка сгенерированная схемой Гиббса
g = reshape(g, [64,64,39]);
imagesc(g(:, :, 6));
