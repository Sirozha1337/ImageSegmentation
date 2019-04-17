% Задаем параметры
Size = 300;
p = 3;
k = 3;

% параметры для vMF
kappas = repmat(10, 1, k);
kappas_restore = kappas;
mus = [[-0.57, 0.57, 0.57];[0, -1, 0];[1, 0, 0];];
mus_restore = mus;
theta1.kappa = 10;
theta1.mu = mus(1, :)';

theta2.kappa = 10;
theta2.mu = mus(2, :)';

theta3.kappa = 10;
theta3.mu = mus(3, :)';

% генерируем случайную матрицу с числами от 1 до k
InitY = randi(k, [Size, Size]);
% Схема Гиббса 
InitY = GibbsSamplerPotts(InitY, 20, 1, k, 2, 4);
% Берем первую картинку из сгенерированной выборки
Sample = InitY(1, :);

% Генерируем выборку из VMF с заданными параметрами
D = vmffactory(p);
clust1 = D.sample(theta1, sum(Sample==1));
clust2 = D.sample(theta2, sum(Sample==2));
clust3 = D.sample(theta3, sum(Sample==3));

% Собираем сгенерированные значения в предназначенные для них места
data = zeros([Size * Size, p]);
data(Sample==1, :) = clust1';
data(Sample==2, :) = clust2';
data(Sample==3, :) = clust3';

% Усредняем данные по третьему измерению для визуализации
% mdata = mean(data, 2);
% imagesc(reshape(Sample, [Size, Size])); figure(); imagesc(reshape(mdata, [Size, Size]));

% задаем начальные mu отличные от тех, с которыми генерировалась выборка
mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];
for i=1:5
    % считаем вероятности
    [probs, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % получаем MLE
    segment_init = MLE(data, probs);
    % трансформируем сегментацию в двумерную матрицу
    segment_init = reshape(segment_init, [Size, Size]);
    % ищем MAP-оценку
    %[map, energy] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 2, k, 10, 4);
    [map, energy] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, k, 10, 4);
    % настраиваем параметры
    [~, mus, kappas] = EstimateParametersOneSample(data, energy, k, p, 2, mus, kappas);
end
% сохраняем посчитанные mu, kappa и финальную сегментацию
mus_em = mus;
kappas_em = kappas;
final_segm_hmrf_em = map;

% возвращаем оригинальные kappa и устанавливаем mu
kappas = kappas_restore;
mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];
for i=1:5
    % считаем вероятности
    [probs, ~] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % получаем начальное приближение
    segment_init = MLE(data, probs);
    % трансформируем сегментацию в двумерную матрицу
    segment_init = reshape(segment_init, [Size, Size]);
    % ищем MAP-оценку 
    [samples] = GibbsSamplerVMF(data, segment_init, 10, 10, k, p, 2, mus, kappas, 4);
    % подстраиваем параметры
    [~, mus, kappas] = EstimateParameters(data, samples, k, p, 2, mus, kappas);
end
% сохраняем посчитанные mu, kappa и финальную сегментацию
mus_mcem = mus;
kappas_mcem = kappas;
final_segm_hmrf_mcem = reshape(samples(end, :), [Size, Size]);

% Отображение GroundTruth и результатов
imagesc(reshape(Sample, [Size, Size]));
figure();
imagesc(final_segm_hmrf_em);
figure();
imagesc(final_segm_hmrf_mcem);

