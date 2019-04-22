% Задаем параметры
Size = 300;
p = 3;
k = 3;

% параметры для vMF
kappas = repmat(10, 1, k);
kappas_restore = kappas;
mus = [[-0.57, 0.57, 0.57];[0, -1, 0]; [1, 0, 0];];
mus_restore = mus;

% генерируем случайную матрицу с числами от 1 до k
InitY = randi(k, [Size, Size]);
[data, Sample]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);

% Усредняем данные по третьему измерению для визуализации
%mdata = mean(data, 2);
%imagesc(reshape(Sample, [Size, Size])); 
%figure(); 
%imagesc(reshape(mdata, [Size, Size]));

% задаем начальные mu отличные от тех, с которыми генерировалась выборка
%mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];

[final_segm_hmrf_em, beta_em, mus_em, kappas_em] = HMRF_EM(data, [Size, Size], k, 2, mus, kappas, 10, 5, 4, 'expansion');

% возвращаем оригинальные kappa и устанавливаем mu
kappas = kappas_restore;
mus = mus_restore;
% сохраняем посчитанные mu, kappa и финальную сегментацию
[final_segm_hmrf_mcem, beta_mcem, mus_mcem, kappas_mcem] = HMRF_MCEM(data, [Size, Size], k, 2, mus, kappas, 10, 10, 5, 4);

% Отображение GroundTruth и результатов
imagesc(reshape(Sample, [Size, Size]));
figure();
imagesc(final_segm_hmrf_em);
figure();
imagesc(final_segm_hmrf_mcem);

dsc_em = SimilarityScore(Sample, final_segm_hmrf_em, k);
dsc_mcem = SimilarityScore(Sample, final_segm_hmrf_mcem, k);

