%Модель наблюдения GMM, данные синтетические. Для заданных значений параметров один раз считается MAP оценка.

% Задаем параметры
Size = 300;
p = 3;
k = 3;
kappas = repmat(10, 1, k);
mus = [[-0.57, 0.57, 0.57];[0, -1, 0]; [1, 0, 0];];
[data, gt] = GenerateSynteticData(randi(k, [Size, Size]), p, k, 2, mus, kappas, 20, 4);

gmm1 = fitgmdist(data(gt==1, :), 1);
gmm2 = fitgmdist(data(gt==2, :), 1);
gmm3 = fitgmdist(data(gt==3, :), 1);

prob1 = gmm1.pdf(data);
prob2 = gmm2.pdf(data);
prob3 = gmm3.pdf(data);
probs = [prob1(:)'; prob2(:)'; prob3(:)'];
logprobs = -log(probs);

% генерируем начальную конфигурацию
segment_init = randi(k, [Size, Size]);
% ищем MAP-оценку
[est, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, k, 5, 4);

dsc = SimilarityScore(gt, est, k);
