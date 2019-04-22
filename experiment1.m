%ћодель наблюдени€ vMF, параметры заданы, данные синтетические. ƒл€ заданных значений параметров один раз считаетс€ MAP оценка.

% «адаем параметры
Size = 300;
p = 3;
k = 3;
kappas = repmat(10, 1, k);
mus = [[-0.57, 0.57, 0.57];[0, -1, 0]; [1, 0, 0];];
[data, gt]=GenerateSynteticData(randi(k, [Size, Size]), p, k, 2, mus, kappas, 20, 4);

% считаем веро€тности
[~, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
% генерируем начальную конфигурацию
segment_init = randi(k, [Size, Size]);
% ищем MAP-оценку
[est, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, k, 5, 4);

dsc = SimilarityScore(gt, est, k);
