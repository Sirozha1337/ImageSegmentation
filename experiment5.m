
% Задаем параметры
Size = 300;
p = 3;
k = 2;
kappas = repmat(10, 1, k);
mus = [[-0.57, 0.57, 0.57];[0, -1, 0];];
close_mus = [[-0.21, 0.64, 0.13];[0.5, -1, 0];];
close_kappas = [5, 8];

% генерируем случайную матрицу с числами от 1 до k
InitY = randi(k, [Size, Size]);
[data, gt]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);

% старт со случайной конфигурации
[final_segm_rand, mus_rand, kappas_rand] = Grab_Cut(data, InitY, k, 10, 40, close_mus, close_kappas, 5, 5, 4);

% старт с MLE
[probs, ~] = CalculateLikelihoodProbabilities(data, k, close_kappas, close_mus);
mle = MLE(zeros([Size * Size, 1]), probs);
mle = reshape(mle, [Size, Size]);
[final_segm_mle, mus_mle, kappas_mle] = Grab_Cut(data, mle, k, 10, 40, close_mus, close_kappas, 5, 5, 4);

% старт с kmeans
segment_init_kmeans = kmeans(data, k, 'MaxIter',500);
vmf = vmffactory(p);
mus_kmeans = zeros(k, p);
kappas_kmeans = zeros(1, k);
for i=1:k
    % достаем данные соответствующие размеченной области
    roi = data(segment_init_kmeans==i, :);
    [theta] = vmf.estimatedefault(roi');
    mus_kmeans(i, :) = squeeze(theta.mu);
    kappas_kmeans(i) = theta.kappa;
end
[final_segm_kmeans, mus_kmeans, kappas_kmeans] = Grab_Cut(data, segment_init_kmeans, k, 10, 40, mus_kmeans, kappas_kmeans, 5, 5, 4);

dsc_rand = SimilarityScore(gt, final_segm_rand, k);
dsc_mle = SimilarityScore(gt, final_segm_mle, k);
dsc_kmeans = SimilarityScore(gt, final_segm_kmeans, k);
